"""Compatibility tests for XGBoost early stopping handling.

These tests do not require the real ``xgboost`` package. They monkeypatch the
classifier/regressor classes and provide a stub ``xgboost.callback`` module so
we can validate that:

- When EarlyStopping callback is available, we do not pass the legacy
  ``early_stopping_rounds`` kwarg to ``fit``.
- When EarlyStopping is unavailable but the estimator's ``fit`` supports the
  legacy kwarg, we fall back to providing it.
"""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

from xgb import model as model_module
from xgb import opinion as opinion_module


def _install_callback_stub(monkeypatch, with_early_stopping: bool) -> None:
    """Install a minimal ``xgboost.callback`` stub into ``sys.modules``."""
    mod = ModuleType("xgboost.callback")
    # Always provide TrainingCallback base type
    class TrainingCallback:  # pragma: no cover - container type
        pass

    setattr(mod, "TrainingCallback", TrainingCallback)

    if with_early_stopping:
        class EarlyStopping:  # pragma: no cover - simple stub
            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs

        setattr(mod, "EarlyStopping", EarlyStopping)

    monkeypatch.setitem(sys.modules, "xgboost.callback", mod)


def test_classifier_uses_callback_when_available(monkeypatch) -> None:
    """Ensure classifier attaches EarlyStopping callback, not legacy kwarg."""

    _install_callback_stub(monkeypatch, with_early_stopping=True)

    class DummyClassifier:
        def __init__(self, **kwargs) -> None:
            self._params = dict(kwargs)
            self.fit_kwargs = None

        def get_xgb_params(self):
            # Simulate single eval metric; code will add missing ones.
            return {"eval_metric": "mlogloss"}

        def set_params(self, **kwargs):  # pragma: no cover - trivial passthrough
            self._params.update(kwargs)
            return self

        def fit(self, X, y, *, eval_set=None, verbose=False, callbacks=None):
            self.fit_kwargs = {"eval_set": eval_set, "verbose": verbose, "callbacks": callbacks}
            return self

        def evals_result(self):
            return {"validation_0": {"mlogloss": [1.0]}}

    monkeypatch.setattr(model_module, "XGBClassifier", DummyClassifier)

    X = np.random.RandomState(0).randn(10, 4).astype(np.float32)
    y = np.array([0, 1] * 5)
    Xv = np.random.RandomState(1).randn(6, 4).astype(np.float32)
    yv = np.array([0, 1, 0, 1, 0, 1])
    batch = model_module.TrainingBatch(
        train=model_module.EncodedDataset(matrix=X, labels=y),
        evaluation=model_module.EncodedDataset(matrix=Xv, labels=yv),
    )

    booster, _history = model_module._train_booster(
        train_config=model_module.XGBoostTrainConfig(), batch=batch, collect_history=True
    )

    assert isinstance(booster, DummyClassifier)
    assert booster.fit_kwargs is not None
    fit_kwargs = booster.fit_kwargs
    # Legacy kwarg must not be present when EarlyStopping callback is used.
    assert "early_stopping_rounds" not in fit_kwargs  # type: ignore[operator]
    callbacks = fit_kwargs["callbacks"]  # type: ignore[index]
    assert callbacks and any(
        "EarlyStopping" in type(cb).__name__ for cb in callbacks
    ), "Expected EarlyStopping callback to be attached"


def test_classifier_falls_back_to_legacy_kwarg(monkeypatch) -> None:
    """Ensure classifier uses legacy kwarg when callback is unavailable."""

    _install_callback_stub(monkeypatch, with_early_stopping=False)

    class DummyClassifier:
        def __init__(self, **kwargs) -> None:
            self._params = dict(kwargs)
            self.fit_kwargs = None

        def get_xgb_params(self):
            return {"eval_metric": "mlogloss"}

        def set_params(self, **kwargs):  # pragma: no cover - trivial passthrough
            self._params.update(kwargs)
            return self

        # Include legacy kwarg to trigger fallback path.
        def fit(
            self,
            X,
            y,
            *,
            eval_set=None,
            verbose=False,
            callbacks=None,
            early_stopping_rounds=None,
        ):
            self.fit_kwargs = {
                "eval_set": eval_set,
                "verbose": verbose,
                "callbacks": callbacks,
                "early_stopping_rounds": early_stopping_rounds,
            }
            return self

        def evals_result(self):
            return {"validation_0": {"mlogloss": [1.0]}}

    monkeypatch.setattr(model_module, "XGBClassifier", DummyClassifier)

    X = np.random.RandomState(0).randn(8, 3).astype(np.float32)
    y = np.array([0, 1] * 4)
    Xv = np.random.RandomState(1).randn(4, 3).astype(np.float32)
    yv = np.array([0, 1, 0, 1])
    batch = model_module.TrainingBatch(
        train=model_module.EncodedDataset(matrix=X, labels=y),
        evaluation=model_module.EncodedDataset(matrix=Xv, labels=yv),
    )

    booster, _history = model_module._train_booster(
        train_config=model_module.XGBoostTrainConfig(), batch=batch, collect_history=True
    )

    assert isinstance(booster, DummyClassifier)
    assert booster.fit_kwargs is not None
    fit_kwargs = booster.fit_kwargs
    # Legacy kwarg should be present when EarlyStopping is unavailable.
    assert fit_kwargs.get("early_stopping_rounds") == 50  # type: ignore[union-attr]
    callbacks = fit_kwargs.get("callbacks")  # type: ignore[union-attr]
    assert callbacks is not None
    assert not any("EarlyStopping" in type(cb).__name__ for cb in callbacks)


def test_classifier_drops_callbacks_when_unsupported_and_uses_legacy_kwarg(monkeypatch) -> None:
    """Estimator without 'callbacks' support should not receive it; use legacy kwarg instead."""

    _install_callback_stub(monkeypatch, with_early_stopping=True)

    class DummyClassifierNoCallbacks:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - trivial
            self.fit_kwargs = None

        def get_xgb_params(self):  # pragma: no cover - trivial
            return {"eval_metric": "mlogloss"}

        def set_params(self, **_kwargs):  # pragma: no cover - trivial
            return self

        # Note: no 'callbacks' kwarg here; include legacy kwarg
        def fit(self, X, y, *, eval_set=None, verbose=False, early_stopping_rounds=None):
            self.fit_kwargs = {
                "eval_set": eval_set,
                "verbose": verbose,
                "early_stopping_rounds": early_stopping_rounds,
            }
            return self

        def evals_result(self):  # pragma: no cover - trivial
            return {"validation_0": {"mlogloss": [1.0]}}

    monkeypatch.setattr(model_module, "XGBClassifier", DummyClassifierNoCallbacks)

    X = np.random.RandomState(42).randn(6, 3).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])
    Xv = np.random.RandomState(43).randn(4, 3).astype(np.float32)
    yv = np.array([0, 1, 0, 1])
    batch = model_module.TrainingBatch(
        train=model_module.EncodedDataset(matrix=X, labels=y),
        evaluation=model_module.EncodedDataset(matrix=Xv, labels=yv),
    )

    booster, _ = model_module._train_booster(
        train_config=model_module.XGBoostTrainConfig(), batch=batch, collect_history=True
    )

    assert isinstance(booster, DummyClassifierNoCallbacks)
    assert booster.fit_kwargs is not None
    assert "early_stopping_rounds" in booster.fit_kwargs
    # Ensure we did not pass 'callbacks'
    assert "callbacks" not in booster.fit_kwargs


def test_regressor_compatibility_callbacks_and_fallback(monkeypatch) -> None:
    """Mirror the classification tests for the opinion regressor."""

    # First run with EarlyStopping available
    _install_callback_stub(monkeypatch, with_early_stopping=True)

    class DummyRegressor:
        def __init__(self, **_kwargs) -> None:
            self.fit_kwargs = None

        def set_params(self, **_kwargs):  # pragma: no cover - trivial
            return self

        def fit(self, X, y, *, eval_set=None, verbose=False, callbacks=None):
            self.fit_kwargs = {"eval_set": eval_set, "verbose": verbose, "callbacks": callbacks}
            return self

        def evals_result(self):
            return {"validation_0": {"mae": [1.0]}}

    monkeypatch.setattr(opinion_module, "XGBRegressor", DummyRegressor)

    X = np.random.RandomState(2).randn(10, 5).astype(np.float32)
    y = np.random.RandomState(3).randn(10).astype(np.float32)
    Xv = np.random.RandomState(4).randn(4, 5).astype(np.float32)
    yv = np.random.RandomState(5).randn(4).astype(np.float32)

    reg, hist = opinion_module._train_regressor(
        features=X,
        targets=y,
        config=opinion_module.OpinionTrainConfig(),
        eval_features=Xv,
        eval_targets=yv,
    )

    assert isinstance(reg, DummyRegressor)
    assert reg.fit_kwargs is not None
    fit_kwargs = reg.fit_kwargs
    assert "early_stopping_rounds" not in fit_kwargs  # type: ignore[operator]
    callbacks = fit_kwargs["callbacks"]  # type: ignore[index]
    assert callbacks and any("EarlyStopping" in type(cb).__name__ for cb in callbacks)
    assert hist

    # Now simulate absence of EarlyStopping and ensure fallback to legacy kwarg
    _install_callback_stub(monkeypatch, with_early_stopping=False)

    class DummyRegressorWithKwarg:
        def __init__(self, **_kwargs) -> None:
            self.fit_kwargs = None

        def set_params(self, **_kwargs):  # pragma: no cover - trivial
            return self

        def fit(
            self,
            X,
            y,
            *,
            eval_set=None,
            verbose=False,
            callbacks=None,
            early_stopping_rounds=None,
        ):
            self.fit_kwargs = {
                "eval_set": eval_set,
                "verbose": verbose,
                "callbacks": callbacks,
                "early_stopping_rounds": early_stopping_rounds,
            }
            return self

        def evals_result(self):
            return {"validation_0": {"mae": [1.0]}}

    monkeypatch.setattr(opinion_module, "XGBRegressor", DummyRegressorWithKwarg)

    reg2, hist2 = opinion_module._train_regressor(
        features=X,
        targets=y,
        config=opinion_module.OpinionTrainConfig(),
        eval_features=Xv,
        eval_targets=yv,
    )

    assert isinstance(reg2, DummyRegressorWithKwarg)
    assert reg2.fit_kwargs is not None
    fit_kwargs2 = reg2.fit_kwargs
    assert fit_kwargs2.get("early_stopping_rounds") == 50  # type: ignore[union-attr]
    callbacks2 = fit_kwargs2.get("callbacks")  # type: ignore[union-attr]
    assert callbacks2 is not None
    assert not any("EarlyStopping" in type(cb).__name__ for cb in callbacks2)
    assert hist2


def test_regressor_drops_callbacks_when_unsupported_and_uses_legacy_kwarg(monkeypatch) -> None:
    _install_callback_stub(monkeypatch, with_early_stopping=True)

    class DummyRegressorNoCallbacks:
        def __init__(self, **_kwargs) -> None:
            self.fit_kwargs = None

        def set_params(self, **_kwargs):  # pragma: no cover - trivial
            return self

        def fit(self, X, y, *, eval_set=None, verbose=False, early_stopping_rounds=None):
            self.fit_kwargs = {
                "eval_set": eval_set,
                "verbose": verbose,
                "early_stopping_rounds": early_stopping_rounds,
            }
            return self

        def evals_result(self):  # pragma: no cover - trivial
            return {"validation_0": {"mae": [1.0]}}

    monkeypatch.setattr(opinion_module, "XGBRegressor", DummyRegressorNoCallbacks)

    X = np.random.RandomState(7).randn(10, 4).astype(np.float32)
    y = np.random.RandomState(8).randn(10).astype(np.float32)
    Xv = np.random.RandomState(9).randn(4, 4).astype(np.float32)
    yv = np.random.RandomState(10).randn(4).astype(np.float32)

    reg, hist = opinion_module._train_regressor(
        features=X,
        targets=y,
        config=opinion_module.OpinionTrainConfig(),
        eval_features=Xv,
        eval_targets=yv,
    )

    assert isinstance(reg, DummyRegressorNoCallbacks)
    assert reg.fit_kwargs is not None
    assert "early_stopping_rounds" in reg.fit_kwargs
    assert "callbacks" not in reg.fit_kwargs
    assert hist


def test_gpu_failure_retries_on_cpu(monkeypatch) -> None:
    """Verify GPU training failures trigger a CPU retry with tree_method=hist."""

    _install_callback_stub(monkeypatch, with_early_stopping=False)

    # Reset GPU state for the test
    model_module._gpu_training_state["enabled"] = True

    class DummyClassifierGPUAware:
        def __init__(self, **kwargs) -> None:
            self.kwargs = dict(kwargs)
            self.init_tree_method = self.kwargs.get("tree_method")
            self.fit_invocations = 0

        def get_xgb_params(self):  # pragma: no cover - trivial
            return {"eval_metric": "mlogloss"}

        def set_params(self, **kwargs):  # pragma: no cover - trivial
            self.kwargs.update(kwargs)
            return self

        def fit(self, X, y, **_kwargs):
            self.fit_invocations += 1
            # Simulate a GPU-related failure when initialised with a GPU tree_method
            if str(self.init_tree_method or "").startswith("gpu"):
                raise RuntimeError("CUDA driver initialization failed")
            return self

        def evals_result(self):  # pragma: no cover - trivial
            return {}

    monkeypatch.setattr(model_module, "XGBClassifier", DummyClassifierGPUAware)

    X = np.random.RandomState(11).randn(6, 3).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1])
    batch = model_module.TrainingBatch(
        train=model_module.EncodedDataset(matrix=X, labels=y),
        evaluation=None,
    )

    cfg = model_module.XGBoostTrainConfig(
        booster=model_module.XGBoostBoosterParams(tree_method="gpu_hist")
    )
    booster, _ = model_module._train_booster(
        train_config=cfg,
        batch=batch,
        collect_history=False,
    )

    # The returned booster should be initialised with CPU tree method after retry
    assert isinstance(booster, DummyClassifierGPUAware)
    assert booster.init_tree_method == "hist"
    # GPU boosters should be disabled for the remainder of the process
    assert model_module._gpu_training_state["enabled"] is False
