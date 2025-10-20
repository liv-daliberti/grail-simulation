"""Unit tests for the XGBoost baseline (skipped if xgboost is unavailable)."""

from __future__ import annotations

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from xgb import cli
from xgb.model import XGBoostSlateModel, XGBoostTrainConfig, fit_xgboost_model, predict_among_slate

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
