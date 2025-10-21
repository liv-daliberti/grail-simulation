"""Unit tests for the XGBoost baseline (skipped if xgboost is unavailable)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from xgb import cli
from xgb.model import (
    XGBoostSlateModel,
    XGBoostTrainConfig,
    fit_xgboost_model,
    predict_among_slate,
)
from xgb.vectorizers import (
    SentenceTransformerVectorizerConfig,
    Word2VecVectorizerConfig,
    create_vectorizer,
    load_vectorizer,
)

pytestmark = pytest.mark.xgb


class DummyDataset:
    """Minimal iterable mimicking the datasets.Dataset API."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.features = {key: None for key in rows[0].keys()} if rows else {}

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:  # pragma: no cover - trivial
        return self._rows[idx]


def _train_example(video_id: str) -> dict:
    return {
        "state_text": f"Prompt for {video_id}",
        "slate_text": f"Primary Title|{video_id}\nFallback Title|alt_{video_id}",
        "gold_id": video_id,
        "viewer_profile_sentence": "Viewer summary",
    }


def _prediction_example() -> dict:
    return {
        "state_text": "Prompt for prediction",
        "slate_items": [
            {"title": "Primary Title", "id": "video_alpha"},
            {"title": "Secondary Title", "id": "video_beta"},
        ],
        "viewer_profile_sentence": "Viewer summary",
    }


def test_fit_xgboost_model_and_predict() -> None:
    dataset = DummyDataset([
        _train_example("video_alpha"),
        _train_example("video_beta"),
    ])
    config = XGBoostTrainConfig(max_train=2, seed=0)
    model = fit_xgboost_model(dataset, config=config)
    assert isinstance(model, XGBoostSlateModel)

    example = _prediction_example()
    prediction_idx, probability_map = predict_among_slate(model, example)
    assert prediction_idx in {1, 2}
    assert probability_map


def test_cli_parser_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args([])
    assert args.out_dir.endswith("xgb")
    assert args.fit_model is False


def test_cli_parser_vectorizer_args() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["--text_vectorizer", "word2vec"])
    assert args.text_vectorizer == "word2vec"
    args = parser.parse_args(
        [
            "--text_vectorizer",
            "sentence_transformer",
            "--sentence_transformer_no_normalize",
            "--sentence_transformer_model",
            "sentence-transformers/all-distilroberta-v1",
        ]
    )
    assert args.text_vectorizer == "sentence_transformer"
    assert args.sentence_transformer_normalize is False
    assert args.sentence_transformer_model.endswith("all-distilroberta-v1")


def test_sentence_transformer_vectorizer_metadata(monkeypatch, tmp_path: Path) -> None:
    class DummyEncoder:
        def __init__(self, config) -> None:
            self.config = config

        def encode(self, texts):
            return np.full((len(texts), 4), 0.5, dtype=np.float32)

        def transform(self, texts):
            return self.encode(texts)

        def embedding_dimension(self) -> int:
            return 4

    monkeypatch.setattr("xgb.vectorizers.SentenceTransformerEncoder", DummyEncoder)
    vectorizer = create_vectorizer(
        "sentence_transformer",
        sentence_transformer=SentenceTransformerVectorizerConfig(
            model_name="stub-model",
            batch_size=8,
            normalize=False,
        ),
    )
    matrix = vectorizer.fit_transform(["alpha", "beta"])
    assert matrix.shape == (2, 4)
    assert vectorizer.feature_dimension() == 4

    save_dir = tmp_path / "vectorizer"
    vectorizer.save(save_dir)
    restored = load_vectorizer(save_dir)
    transformed = restored.transform(["gamma"])
    assert transformed.shape == (1, 4)
    assert restored.feature_dimension() == 4


def test_word2vec_vectorizer_stub(monkeypatch) -> None:
    class DummyBuilder:
        def __init__(self, config) -> None:
            self.config = config

        def train(self, corpus):
            self._corpus = list(corpus)

        def transform(self, corpus):
            docs = list(corpus)
            return np.ones((len(docs), self.config.vector_size), dtype=np.float32)

        def save(self, directory: Path) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "word2vec.model").write_bytes(b"stub")

        def load(self, directory: Path) -> None:  # pragma: no cover - simple stub
            pass

    monkeypatch.setattr("xgb.vectorizers.Word2VecFeatureBuilder", DummyBuilder)
    vectorizer = create_vectorizer(
        "word2vec",
        word2vec=Word2VecVectorizerConfig(vector_size=3, window=2, min_count=1, epochs=1, workers=1, seed=0),
    )
    matrix = vectorizer.fit_transform(["doc-one", "doc-two"])
    assert matrix.shape == (2, 3)
    assert vectorizer.feature_dimension() == 3
