"""Unit tests for the refactored knn feature and index modules."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")
pytest.importorskip("gensim")

from knn import cli
from knn.data import DEFAULT_DATASET_SOURCE
from knn.features import Word2VecConfig, prepare_training_documents
from knn.index import (
    SlateQueryConfig,
    build_tfidf_index,
    build_word2vec_index,
    knn_predict_among_slate_multi,
    load_word2vec_index,
    save_word2vec_index,
)

pytestmark = pytest.mark.knn


class DummyDataset:
    """Minimal iterable mimicking the huggingface dataset API used in tests."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.features = {"dummy": None}

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


def _example_row() -> dict:
    return {
        "issue": "minimum_wage",
        "state_text": "Viewer prompt",
        "gold_id": "video12345",
        "gold_index": 1,
        "slate_text": "Recommended Video|video12345",
    }


def test_prepare_training_documents_returns_docs() -> None:
    dataset = DummyDataset([_example_row()])
    docs, labels_id, labels_title = prepare_training_documents(
        dataset,
        max_train=10,
        seed=0,
        extra_fields=(),
    )
    assert docs and labels_id and labels_title
    assert labels_id[0] == "video12345"


def test_build_tfidf_index_and_predict() -> None:
    dataset = DummyDataset([_example_row()])
    index = build_tfidf_index(dataset, max_train=10, seed=0)
    assert index["X"].shape[0] == 1
    predictions = knn_predict_among_slate_multi(
        knn_index=index,
        example=_example_row(),
        k_values=[1, 5],
        config=SlateQueryConfig(),
    )
    assert predictions[1] == 1


def test_cli_parser_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args([])
    assert args.dataset == DEFAULT_DATASET_SOURCE
    assert Path(args.out_dir).name == "knn"
    assert args.feature_space == "tfidf"
    assert args.word2vec_size == 256
    assert args.word2vec_model_dir == ""


def test_cli_parser_hyphenated_options() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--feature-space",
            "tfidf",
            "--fit-index",
            "--knn-k",
            "13",
            "--eval-max",
            "5",
            "--issue",
            "minimum_wage",
        ]
    )
    assert args.fit_index is True
    assert args.knn_k == 13
    assert args.eval_max == 5
    assert args.issues == "minimum_wage"


def test_build_word2vec_index_roundtrip(tmp_path) -> None:
    dataset = DummyDataset([_example_row()])
    config = Word2VecConfig(
        vector_size=16,
        window=2,
        min_count=1,
        epochs=1,
        model_dir=tmp_path / "w2v_model",
    )
    index = build_word2vec_index(
        dataset,
        max_train=10,
        seed=0,
        config=config,
    )
    assert index["feature_space"] == "word2vec"
    predictions = knn_predict_among_slate_multi(
        knn_index=index,
        example=_example_row(),
        k_values=[1, 5],
        config=SlateQueryConfig(),
    )
    assert predictions[1] == 1

    save_path = tmp_path / "persisted"
    save_word2vec_index(index, save_path)
    loaded = load_word2vec_index(save_path)
    assert loaded["feature_space"] == "word2vec"
    predictions_loaded = knn_predict_among_slate_multi(
        knn_index=loaded,
        example=_example_row(),
        k_values=[1, 5],
        config=SlateQueryConfig(),
    )
    assert predictions_loaded[1] == predictions[1]
