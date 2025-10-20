"""Unit tests for the refactored knn feature and index modules."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from knn import cli
from knn.data import DEFAULT_DATASET_SOURCE
from knn.features import prepare_training_documents
from knn.index import (SlateQueryConfig, build_tfidf_index, knn_predict_among_slate_multi)

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
