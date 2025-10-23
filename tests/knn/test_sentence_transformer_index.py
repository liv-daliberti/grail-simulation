"""Unit tests covering the sentence-transformer feature space for KNN."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from knn.index import (
    build_sentence_transformer_index,
    load_sentence_transformer_index,
    save_sentence_transformer_index,
)
from knn.pipeline import PipelineContext, _build_sweep_configs
from common.embeddings import SentenceTransformerConfig


def _stub_prepare_documents(*_args: Any, **_kwargs: Any) -> tuple[list[str], list[str], list[str]]:
    return (["doc-1", "doc-2"], ["vid-1", "vid-2"], ["Title One", "Title Two"])


def test_build_sentence_transformer_index(monkeypatch) -> None:
    class DummyEncoder:
        def __init__(self, config: SentenceTransformerConfig) -> None:
            self.config = config

        def encode(self, texts: Sequence[str]) -> np.ndarray:
            return np.full((len(texts), 5), 0.25, dtype=np.float32)

        def transform(self, texts: Sequence[str]) -> np.ndarray:
            return self.encode(texts)

        def embedding_dimension(self) -> int:
            return 5

    monkeypatch.setattr("knn.index.prepare_training_documents", _stub_prepare_documents)
    monkeypatch.setattr("knn.index.SentenceTransformerEncoder", DummyEncoder)

    index = build_sentence_transformer_index(
        train_ds=[],
        max_train=10,
        seed=0,
        extra_fields=("viewer_profile",),
        config=SentenceTransformerConfig(model_name="stub-model"),
    )
    assert index["feature_space"] == "sentence_transformer"
    assert index["X"].shape == (2, 5)
    transformed = index["vectorizer"].transform(["doc-3"])
    assert transformed.shape == (1, 5)


def test_save_and_load_sentence_transformer_index(monkeypatch, tmp_path: Path) -> None:
    class DummyEncoder:
        def __init__(self, config: SentenceTransformerConfig) -> None:
            self.config = config

        def encode(self, texts: Sequence[str]) -> np.ndarray:
            return np.zeros((len(texts), 3), dtype=np.float32)

        def transform(self, texts: Sequence[str]) -> np.ndarray:
            return self.encode(texts)

        def embedding_dimension(self) -> int:
            return 3

    monkeypatch.setattr("knn.index.prepare_training_documents", _stub_prepare_documents)
    monkeypatch.setattr("knn.index.SentenceTransformerEncoder", DummyEncoder)

    index = build_sentence_transformer_index(train_ds=[], config=SentenceTransformerConfig(model_name="stub"))
    save_sentence_transformer_index(index, str(tmp_path))

    # Ensure load also uses the dummy encoder
    monkeypatch.setattr("knn.index.SentenceTransformerEncoder", DummyEncoder)
    restored = load_sentence_transformer_index(str(tmp_path))
    assert restored["feature_space"] == "sentence_transformer"
    assert restored["X"].shape == (2, 3)
    assert restored["vectorizer"].transform(["alpha"]).shape == (1, 3)


def test_pipeline_sentence_transformer_sweep(tmp_path: Path) -> None:
    context = PipelineContext(
        dataset="dataset",
        out_dir=tmp_path,
        cache_dir=str(tmp_path / "cache"),
        sweep_dir=tmp_path / "sweeps",
        word2vec_model_dir=tmp_path / "word2vec",
        k_sweep="1,2",
        study_tokens=(),
        word2vec_epochs=5,
        word2vec_workers=2,
        sentence_model="stub-model",
        sentence_device=None,
        sentence_batch_size=16,
        sentence_normalize=True,
        feature_spaces=("sentence_transformer",),
        jobs=1,
    )
    configs = _build_sweep_configs(context)
    assert any(cfg.feature_space == "sentence_transformer" for cfg in configs)
