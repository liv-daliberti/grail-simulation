"""Unit tests for ``knn.pipeline_utils`` helper functions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Optional

import pytest

from knn.pipeline import utils


@dataclass
class DummyTask:
    """Simple sweep task carrying just the attributes used by the helpers."""

    index: int
    config: str
    study: str
    metrics_path: Path


def test_partition_cached_tasks_reuses_entries(tmp_path: Path) -> None:
    """Cached tasks should be filtered out while preserving pending work."""

    tasks = ["task_a", "task_b", "task_c"]
    cache_locations = {task: tmp_path / f"{task}.json" for task in tasks}
    cached_payload = {"task_b": {"accuracy": 0.42}}

    cache_locations["task_b"].write_text("cached", encoding="utf-8")

    pending, cached = utils.partition_cached_tasks(
        tasks,
        reuse_existing=True,
        cache_path=lambda task: cache_locations[task],
        load_cached=lambda task: cached_payload.get(task),
    )

    assert pending == ["task_a", "task_c"]
    assert cached == [cached_payload["task_b"]]


def test_prepare_task_grid_uses_default_metrics_path(tmp_path: Path) -> None:
    """Default cache-path lookup should rely on the task's ``metrics_path`` attribute."""

    configs = ["cfg_a", "cfg_b"]
    studies = ["study1"]
    cached_indices = {0}

    for index, (config, study) in enumerate(product(configs, studies)):
        if index in cached_indices:
            (tmp_path / f"{config}_{study}.json").write_text("hit", encoding="utf-8")

    def build_task(index: int, config: str, study: str) -> DummyTask:
        metrics_path = tmp_path / f"{config}_{study}.json"
        return DummyTask(
            index=index,
            config=config,
            study=study,
            metrics_path=metrics_path,
        )

    pending, cached = utils.prepare_task_grid(
        configs=configs,
        studies=studies,
        reuse_existing=True,
        build_task=build_task,
        cache=utils.TaskCacheStrategy(
            load_cached=lambda task: {"cached_index": task.index}
            if task.index in cached_indices
            else None
        ),
    )

    assert [task.index for task in pending] == [1]
    assert pending[0].metrics_path == tmp_path / "cfg_b_study1.json"
    assert cached == [{"cached_index": 0}]


def test_prepare_task_grid_without_metrics_path_raises() -> None:
    """Tasks missing a ``metrics_path`` should trigger a helpful error."""

    @dataclass
    class MissingMetricsTask:
        index: int

    with pytest.raises(AttributeError, match="metrics_path"):
        utils.prepare_task_grid(
            configs=("cfg",),
            studies=("study",),
            reuse_existing=False,
            build_task=lambda index, config, study: MissingMetricsTask(index=index),
            cache=utils.TaskCacheStrategy(load_cached=lambda task: None),
        )


def test_formatting_helpers_cover_edge_cases() -> None:
    """Formatting utilities should gracefully handle ``None`` and boundary values."""

    assert utils.format_float(1.2349) == "1.235"
    assert utils.format_optional_float(None) == "—"
    assert utils.format_optional_float(0.125) == "0.125"
    assert utils.format_delta(None) == "—"
    assert utils.format_delta(-0.05) == "-0.050"
    assert utils.format_count(None) == "—"
    assert utils.format_count(12345) == "12,345"
    assert utils.format_k(None) == "—"
    assert utils.format_k(0) == "—"
    assert utils.format_k(9) == "9"
    assert utils.format_uncertainty_details({"n_bootstrap": 10, "seed": 3}) == " (n_bootstrap=10, seed=3)"
    assert utils.format_uncertainty_details("invalid") == ""
    assert utils.snake_to_title("opinion_shift") == "Opinion Shift"


def test_parse_ci_accepts_mappings_and_sequences() -> None:
    """Confidence-interval parsing should accept both mappings and sequences."""

    assert utils.parse_ci({"low": "0.2", "high": 0.8}) == pytest.approx((0.2, 0.8))
    assert utils.parse_ci([0.1, "0.9"]) == pytest.approx((0.1, 0.9))
    assert utils.parse_ci({"low": 0.1}) is None
    assert utils.parse_ci("invalid") is None


def test_extract_metric_summary_normalises_values() -> None:
    """Metric summaries should convert numeric payloads into typed fields."""

    payload = {
        "accuracy_overall": "0.812",
        "accuracy_ci_95": {"low": 0.7, "high": 0.9},
        "baseline_ci_95": [0.5, 0.6],
        "baseline_most_frequent_gold_index": {"accuracy": "0.45"},
        "random_baseline_expected_accuracy": "0.2",
        "best_k": "5",
        "n_total": "1000",
        "n_eligible": "850",
    }

    summary = utils.extract_metric_summary(payload)

    assert summary.accuracy == pytest.approx(0.812)
    assert summary.accuracy_ci == pytest.approx((0.7, 0.9))
    assert summary.baseline == pytest.approx(0.45)
    assert summary.baseline_ci == pytest.approx((0.5, 0.6))
    assert summary.random_baseline == pytest.approx(0.2)
    assert summary.best_k == 5
    assert summary.n_total == 1000
    assert summary.n_eligible == 850


def test_extract_opinion_summary_combines_metrics() -> None:
    """Opinion summaries should compute deltas and fallbacks from nested payloads."""

    payload = {
        "best_metrics": {
            "mae_after": "0.42",
            "rmse_after": 0.63,
            "r2_after": "0.22",
            "mae_change": "-0.08",
            "rmse_change": "-0.11",
            "direction_accuracy": "0.58",
            "calibration_slope": 1.1,
            "calibration_intercept": "-0.02",
            "calibration_ece": 0.03,
            "kl_divergence_change": 0.5,
            "eligible": "44",
        },
        "baseline": {
            "mae_using_before": "0.5",
            "direction_accuracy": 0.5,
            "rmse_change_zero": 0.0,
            "calibration_slope_change_zero": 0.9,
            "calibration_intercept_change_zero": 0.01,
            "calibration_ece_change_zero": 0.07,
            "kl_divergence_change_zero": 0.6,
        },
        "n_participants": "60",
        "best_k": "9",
        "dataset": "sim-dataset",
        "split": "validation",
    }

    summary = utils.extract_opinion_summary(payload)

    assert summary.mae == pytest.approx(0.42)
    assert summary.rmse == pytest.approx(0.63)
    assert summary.mae_change == pytest.approx(-0.08)
    assert summary.rmse_change == pytest.approx(-0.11)
    assert summary.baseline_mae == pytest.approx(0.5)
    assert summary.mae_delta == pytest.approx(0.08)
    assert summary.accuracy == pytest.approx(0.58)
    assert summary.baseline_accuracy == pytest.approx(0.5)
    assert summary.accuracy_delta == pytest.approx(0.08)
    assert summary.calibration_slope == pytest.approx(1.1)
    assert summary.baseline_calibration_slope == pytest.approx(0.9)
    assert summary.calibration_intercept == pytest.approx(-0.02)
    assert summary.baseline_calibration_intercept == pytest.approx(0.01)
    assert summary.calibration_ece == pytest.approx(0.03)
    assert summary.baseline_calibration_ece == pytest.approx(0.07)
    assert summary.kl_divergence_change == pytest.approx(0.5)
    assert summary.baseline_kl_divergence_change == pytest.approx(0.6)
    assert summary.best_k == 9
    assert summary.participants == 60
    assert summary.eligible == 44
    assert summary.dataset == "sim-dataset"
    assert summary.split == "validation"


def test_ensure_feature_selections_behaviour(caplog: pytest.LogCaptureFixture) -> None:
    """Validation helper should warn or raise depending on configuration."""

    logger = logging.getLogger("tests.knn.pipeline_utils")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        utils.ensure_feature_selections(
            selections={"tfidf": {"study_a": object()}},
            expected_keys=("study_a", "study_b"),
            options=utils.SelectionValidationOptions(
                allow_incomplete=True,
                logger=logger,
                missing_descriptor="sweep selections",
                empty_descriptor="feature spaces",
                require_selected=False,
            ),
        )
    assert "Missing sweep selections for feature=tfidf: study_b" in caplog.text

    with pytest.raises(RuntimeError, match="Missing metrics for feature=tfidf: study_b"):
        utils.ensure_feature_selections(
            selections={"tfidf": {"study_a": object()}},
            expected_keys=("study_a", "study_b"),
            options=utils.SelectionValidationOptions(
                allow_incomplete=False,
                logger=logger,
                missing_descriptor="metrics",
                empty_descriptor="feature spaces",
                require_selected=False,
            ),
        )

    with pytest.raises(RuntimeError, match="Failed to select a best configuration"):
        utils.ensure_feature_selections(
            selections={},
            expected_keys=("study_a",),
            options=utils.SelectionValidationOptions(
                allow_incomplete=True,
                logger=logger,
                missing_descriptor="metrics",
                empty_descriptor="runs",
                require_selected=True,
            ),
        )
