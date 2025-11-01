#!/usr/bin/env python
"""Smoke tests for the XGB model refactor public API.

These tests exercise imports only and avoid heavy dependencies by enabling the
lightweight import mode exposed by ``xgb`` and ``xgb.core``. They validate that
the high-level ``xgb.core.model`` module re-exports objects from the new split
modules so downstream imports remain stable.
"""

from __future__ import annotations

import os


def test_model_reexports_from_split_modules(monkeypatch) -> None:
    # Ensure imports don't pull in the full pipeline or knn dependencies.
    monkeypatch.setenv("XGB_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("XGB_CORE_LIGHT_IMPORTS", "1")

    # Import the modules under the lightweight mode.
    import importlib

    model = importlib.import_module("xgb.core.model")
    model_predict = importlib.import_module("xgb.core.model_predict")
    model_config = importlib.import_module("xgb.core.model_config")
    model_types = importlib.import_module("xgb.core.model_types")

    # Public API is re-exported by xgb.core.model
    assert model.load_xgboost_model.__module__ == model_predict.__name__
    assert model.save_xgboost_model.__module__ == model_predict.__name__
    assert model.predict_among_slate.__module__ == model_predict.__name__

    assert model.XGBoostBoosterParams.__module__ == model_config.__name__
    assert model.XGBoostTrainConfig.__module__ == model_config.__name__

    assert model.XGBoostSlateModel.__module__ == model_types.__name__


def test_booster_params_accepts_flat_kwargs(monkeypatch) -> None:
    monkeypatch.setenv("XGB_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("XGB_CORE_LIGHT_IMPORTS", "1")
    import importlib

    model = importlib.import_module("xgb.core.model")

    # Using flat kwargs should be accepted and populate grouped fields.
    params = model.XGBoostBoosterParams.create(
        learning_rate=0.2,
        max_depth=4,
        n_estimators=123,
        subsample=0.9,
        colsample_bytree=0.7,
        tree_method="hist",
        reg_lambda=2.0,
        reg_alpha=0.1,
        some_unknown_kwarg=True,  # carried to extra_kwargs
    )
    assert params.learning_rate == 0.2
    assert params.max_depth == 4
    assert params.n_estimators == 123
    assert params.subsample == 0.9
    assert params.colsample_bytree == 0.7
    assert params.tree_method == "hist"
    assert params.reg_lambda == 2.0
    assert params.reg_alpha == 0.1

    # Any unknown kwargs should be preserved for estimator construction.
    assert isinstance(params.extra_kwargs, dict)
    assert params.extra_kwargs.get("some_unknown_kwarg") is True


def test_train_config_create_defaults(monkeypatch) -> None:
    monkeypatch.setenv("XGB_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("XGB_CORE_LIGHT_IMPORTS", "1")
    import importlib

    model = importlib.import_module("xgb.core.model")

    cfg = model.XGBoostTrainConfig.create(
        max_train=1000,
        seed=7,
        vectorizer_kind="tfidf",
    )
    assert cfg.max_train == 1000
    assert cfg.seed == 7
    assert cfg.vectorizer_kind == "tfidf"
    # Accessors proxy to grouped configs
    assert cfg.tfidf is not None
    assert cfg.word2vec is not None
    assert cfg.sentence_transformer is not None


def test_train_config_accepts_vectorizer_overrides(monkeypatch) -> None:
    monkeypatch.setenv("XGB_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("XGB_CORE_LIGHT_IMPORTS", "1")
    import importlib

    model = importlib.import_module("xgb.core.model")

    # Provide explicit vectorizer overrides and ensure they are threaded
    tfidf = model.TfidfConfig(max_features=123)
    w2v = model.Word2VecVectorizerConfig(vector_size=17)
    st = model.SentenceTransformerVectorizerConfig(model_name="abc/xyz")
    cfg = model.XGBoostTrainConfig.create(
        max_train=5,
        seed=11,
        vectorizer_kind="tfidf",
        tfidf=tfidf,
        word2vec=w2v,
        sentence_transformer=st,
    )
    assert cfg.max_train == 5
    assert cfg.seed == 11
    assert cfg.vectorizer_kind == "tfidf"
    assert cfg.tfidf.max_features == 123
    assert cfg.word2vec.vector_size == 17
    assert cfg.sentence_transformer.model_name == "abc/xyz"
