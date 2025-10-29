"""Unit tests for the refactored knn feature and index modules."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")
pytest.importorskip("gensim")

from common.prompts.docs import DEFAULT_EXTRA_TEXT_FIELDS

from knn.cli import build_parser
from knn.core import evaluate
from knn.core.data import DEFAULT_DATASET_SOURCE
from knn.core.features import Word2VecConfig, prepare_training_documents
from knn.core.index import (
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
        extra_fields=DEFAULT_EXTRA_TEXT_FIELDS,
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
    parser = build_parser()
    args = parser.parse_args([])
    assert args.dataset == DEFAULT_DATASET_SOURCE
    assert Path(args.out_dir).name == "knn"
    assert args.feature_space == "tfidf"
    assert args.word2vec_size == 256
    assert args.word2vec_model_dir == ""
    assert args.train_curve_max == 0


def test_cli_parser_hyphenated_options() -> None:
    parser = build_parser()
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
            "--train-curve-max",
            "42",
        ]
    )
    assert args.fit_index is True
    assert args.knn_k == 13
    assert args.eval_max == 5
    assert args.issues == "minimum_wage"
    assert args.train_curve_max == 42


def test_select_best_k_prefers_elbow() -> None:
    k_values = [1, 5, 10]
    accuracy = {1: 0.2, 5: 0.5, 10: 0.52}
    assert evaluate.select_best_k(k_values, accuracy) == 5


def test_plot_elbow_creates_image(tmp_path) -> None:
    output = tmp_path / "elbow.png"
    evaluate.plot_elbow([1, 3, 5], {1: 0.1, 3: 0.6, 5: 0.4}, best_k=3, output_path=output)
    assert output.exists()


def test_compute_auc_from_curve() -> None:
    k_values = [1, 4, 7]
    accuracy = {1: 0.2, 4: 0.4, 7: 0.6}
    area, normalised = evaluate.compute_auc_from_curve(k_values, accuracy)
    assert area > 0
    assert 0.0 <= normalised <= 1.0


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
