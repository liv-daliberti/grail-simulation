#!/usr/bin/env python
# pylint: disable=missing-function-docstring
from __future__ import annotations

import json
from pathlib import Path

import pytest

import grpo.pipeline as grpo_pipeline
from common.opinion.metrics import compute_opinion_metrics


def _write_next_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "issue": "gun_control",
            "participant_study": "study1",
            "n_options": 3,
            "position_index": 0,
            "gold_index": 1,
            "parsed_index": 1,
            "eligible": True,
            "correct": True,
            "gpt_output": "<answer>1</answer>",
        },
        {
            "issue": "minimum_wage",
            "participant_study": "study2",
            "n_options": 4,
            "position_index": 1,
            "gold_index": 2,
            "parsed_index": 2,
            "eligible": True,
            "correct": True,
            "gpt_output": "<answer>2</answer>",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_opinion_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"before": 3.0, "after": 4.0, "prediction": 4.0},
        {"before": 5.0, "after": 4.0, "prediction": 4.0},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_rebuilds_next_video_metrics_from_predictions(tmp_path: Path) -> None:
    run_dir = tmp_path / "models" / "grpo" / "next_video" / "unit"
    preds = run_dir / "predictions.jsonl"
    _write_next_predictions(preds)

    # No metrics.json present; loader should reconstruct it from predictions.
    result = grpo_pipeline._load_next_video_from_disk(run_dir)
    assert result is not None
    assert result.metrics_path.exists(), "metrics.json should be created from predictions"
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    assert pytest.approx(metrics.get("accuracy_overall"), rel=1e-6) == 1.0
    assert pytest.approx(metrics.get("parsed_rate"), rel=1e-6) == 1.0
    assert pytest.approx(metrics.get("format_rate"), rel=1e-6) == 1.0
    assert metrics.get("n_eligible") == 2
    assert metrics.get("n_total") == 2


def test_rebuilds_opinion_study_and_combined_from_predictions(tmp_path: Path) -> None:
    out_dir = tmp_path / "models" / "grpo" / "opinion" / "unit"
    study_dir = out_dir / "study1"
    preds = study_dir / "predictions.jsonl"
    _write_opinion_predictions(preds)

    # No per-study metrics.json or combined present; loader should reconstruct both.
    result = grpo_pipeline._load_opinion_from_disk(out_dir)
    assert result is not None

    # Study-level metrics.json should have been written by the rebuild.
    metrics_path = study_dir / "metrics.json"
    assert metrics_path.exists(), "study metrics.json should be created from predictions"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    study_metrics = payload.get("metrics", {})
    assert study_metrics.get("eligible") == 2
    assert study_metrics.get("participants") == 2
    assert pytest.approx(study_metrics.get("mae_after"), rel=1e-6) == 0.0
    assert pytest.approx(study_metrics.get("direction_accuracy"), rel=1e-6) == 1.0

    # Combined metrics should reflect available predictions when missing on disk.
    combined = result.combined_metrics
    expected = compute_opinion_metrics(
        truth_after=[4.0, 4.0], truth_before=[3.0, 5.0], pred_after=[4.0, 4.0]
    )
    for key in ("eligible", "mae_after", "direction_accuracy"):
        assert pytest.approx(combined.get(key), rel=1e-6) == expected.get(key)


def test_logging_summaries_emit_expected_lines(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Next-video log summary
    next_dir = tmp_path / "nv"
    next_dir.mkdir(parents=True, exist_ok=True)
    nv_metrics = {
        "accuracy_overall": 0.75,
        "parsed_rate": 0.9,
        "format_rate": 0.8,
        "n_eligible": 3,
        "n_total": 4,
        "baseline_most_frequent_gold_index": {"accuracy": 0.5, "top_index": 1, "count": 2},
        "random_baseline_expected_accuracy": 0.33,
    }
    nv = grpo_pipeline.NextVideoEvaluationResult(
        run_dir=next_dir,
        metrics_path=next_dir / "metrics.json",
        predictions_path=next_dir / "predictions.jsonl",
        qa_log_path=next_dir / "qa.log",
        metrics=nv_metrics,
    )
    with caplog.at_level("INFO", logger="grpo.pipeline"):
        grpo_pipeline._log_next_video_summary(nv)
    assert any("Next-video metrics | accuracy=0.750" in message for message in caplog.messages)

    # Opinion log summary
    study_dir = tmp_path / "op" / "study1"
    study_dir.mkdir(parents=True, exist_ok=True)
    opinion_metrics = {
        "eligible": 2,
        "mae_after": 0.1,
        "rmse_after": 0.2,
        "mae_change": 0.05,
        "rmse_change": 0.15,
        "direction_accuracy": 0.6,
    }
    files = grpo_pipeline.OpinionStudyFiles(
        metrics=study_dir / "metrics.json",
        predictions=study_dir / "predictions.jsonl",
        qa_log=study_dir / "qa.log",
    )
    summary = grpo_pipeline.OpinionStudySummary(
        metrics=opinion_metrics,
        baseline={"direction_accuracy": 0.5},
        participants=2,
        eligible=2,
    )
    study = grpo_pipeline.OpinionStudyResult(
        study=grpo_pipeline.DEFAULT_SPECS[0],
        files=files,
        summary=summary,
    )
    opinion = grpo_pipeline.OpinionEvaluationResult(
        studies=[study], combined_metrics=opinion_metrics
    )
    caplog.clear()
    with caplog.at_level("INFO", logger="grpo.pipeline"):
        grpo_pipeline._log_opinion_summary(opinion)
    assert any("Opinion metrics | direction=0.600" in message for message in caplog.messages)

