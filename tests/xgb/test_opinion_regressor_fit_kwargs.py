#!/usr/bin/env python
import types
from typing import Any, List

import numpy as np
import pytest


@pytest.mark.xgb
def test_train_regressor_older_xgb_prunes_callbacks_and_uses_early_stopping(monkeypatch):
    """
    Ensure `_train_regressor` avoids passing unsupported `callbacks` to older
    XGBoost versions and falls back to `early_stopping_rounds` when available.
    """

    import xgb.opinion as opinion
    from xgb.model import XGBoostBoosterParams

    recorded: dict = {}

    class FakeRegressorOld:
        def __init__(self, **kwargs: Any) -> None:  # accepts ctor kwargs
            self.ctor_kwargs = kwargs
            self.set_params_kwargs = None
            self.fit_received = None

        def set_params(self, **kwargs: Any) -> None:
            self.set_params_kwargs = kwargs

        # Simulate older XGBoost signature without `callbacks` support
        def fit(self, X, y, eval_set=None, verbose=False, early_stopping_rounds=None):
            # Record the kwargs we actually received
            self.fit_received = {
                "eval_set": eval_set,
                "verbose": verbose,
                "early_stopping_rounds": early_stopping_rounds,
            }
            # A hard failure here would indicate that `callbacks` leaked through
            recorded["fit_called"] = True

        def evals_result(self) -> dict:
            return {"validation_1": {"mae": [0.6, 0.5, 0.4]}}

    # Replace the estimator used by opinion with a fake (older) implementation
    monkeypatch.setattr(opinion, "XGBRegressor", FakeRegressorOld, raising=True)

    # Minimal training arrays
    X_train = np.zeros((4, 3), dtype=np.float32)
    y_train = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    X_val = np.ones((2, 3), dtype=np.float32)
    y_val = np.array([0.2, 0.5], dtype=np.float32)

    config = opinion.OpinionTrainConfig(booster=XGBoostBoosterParams())

    reg, history = opinion._train_regressor(  # type: ignore[attr-defined]
        features=X_train,
        targets=y_train,
        config=config,
        eval_features=X_val,
        eval_targets=y_val,
    )

    assert recorded.get("fit_called"), "fit should be invoked"
    # Confirm that callbacks were not passed (would have raised) and legacy
    # early stopping was used instead.
    assert reg.fit_received is not None
    assert reg.fit_received["early_stopping_rounds"] == 50
    assert isinstance(history, dict) and "validation_1" in history


@pytest.mark.xgb
def test_train_regressor_newer_xgb_keeps_callbacks_and_omits_legacy_kw(monkeypatch):
    """
    Ensure `_train_regressor` passes `callbacks` when supported and does not use
    `early_stopping_rounds` when an EarlyStopping callback is attached.
    """

    import importlib
    import xgb.opinion as opinion
    from xgb.model import XGBoostBoosterParams

    # Stub xgboost.callback module so opinion attaches an EarlyStopping callback
    real_import = importlib.import_module

    def fake_import_module(name: str, package=None):
        if name == "xgboost.callback":
            mod = types.ModuleType("xgboost.callback")

            class TrainingCallback:  # minimal base class for subclassing
                pass

            class EarlyStopping:
                def __init__(self, *args, **kwargs):  # accept any signature
                    self.args = args
                    self.kwargs = kwargs

            mod.TrainingCallback = TrainingCallback  # type: ignore[attr-defined]
            mod.EarlyStopping = EarlyStopping  # type: ignore[attr-defined]
            return mod
        return real_import(name, package)  # fallback to the real importer

    monkeypatch.setattr(importlib, "import_module", fake_import_module, raising=True)

    class FakeRegressorNew:
        def __init__(self, **kwargs: Any) -> None:
            self.ctor_kwargs = kwargs
            self.set_params_kwargs = None
            self.fit_received = None

        def set_params(self, **kwargs: Any) -> None:
            self.set_params_kwargs = kwargs

        # Simulate newer XGBoost signature with `callbacks` kw-only param
        def fit(self, X, y, *, eval_set=None, verbose=False, callbacks=None):
            self.fit_received = {
                "eval_set": eval_set,
                "verbose": verbose,
                "callbacks": callbacks,
            }

        def evals_result(self) -> dict:
            return {"validation_1": {"mae": [0.4, 0.35, 0.33]}}

    monkeypatch.setattr(opinion, "XGBRegressor", FakeRegressorNew, raising=True)

    X_train = np.zeros((4, 3), dtype=np.float32)
    y_train = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    X_val = np.ones((2, 3), dtype=np.float32)
    y_val = np.array([0.2, 0.5], dtype=np.float32)

    config = opinion.OpinionTrainConfig(booster=XGBoostBoosterParams())

    reg, history = opinion._train_regressor(  # type: ignore[attr-defined]
        features=X_train,
        targets=y_train,
        config=config,
        eval_features=X_val,
        eval_targets=y_val,
    )

    assert reg.fit_received is not None
    # Callbacks should be present and include an EarlyStopping-like object
    callbacks = reg.fit_received.get("callbacks")
    assert isinstance(callbacks, list) and callbacks, "callbacks should be attached"
    assert any("EarlyStopping" in type(cb).__name__ for cb in callbacks)
    # Ensure legacy kwarg is not passed (would raise due to signature)
    assert "early_stopping_rounds" not in reg.fit_received
    assert isinstance(history, dict) and "validation_1" in history
