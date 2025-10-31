#!/usr/bin/env python
# pylint: disable=missing-function-docstring
"""Unit tests for the shared RLHF reporting helpers."""

import json
from pathlib import Path

from common.opinion import DEFAULT_SPECS
from grpo.next_video import NextVideoEvaluationResult
from grpo.opinion import (
    OpinionEvaluationResult,
    OpinionStudyFiles,
    OpinionStudyResult,
    OpinionStudySummary,
)
from common.rlhf.reports import ReportOptions, generate_reports


def _sample_next_video_metrics() -> dict:
    return {
        "accuracy_overall": 0.75,
        "parsed_rate": 0.9,
        "format_rate": 0.88,
        "n_eligible": 12,
        "n_total": 15,
        "group_metrics": {
            "by_issue": {
                "gun_control": {
                    "n_seen": 6,
                    "n_eligible": 5,
                    "accuracy": 0.8,
                    "parsed_rate": 1.0,
                    "format_rate": 1.0,
                }
            },
            "by_participant_study": {
                "study1": {
                    "n_seen": 6,
                    "n_eligible": 5,
                    "accuracy": 0.8,
                    "parsed_rate": 1.0,
                    "format_rate": 1.0,
                }
            },
        },
    }


def _sample_opinion_metrics() -> dict:
    return {
        "eligible": 4,
        "mae_after": 0.13,
        "mae_change": 0.27,
        "direction_accuracy": 0.7,
        "rmse_after": 0.34,
        "rmse_change": 0.45,
        "calibration_ece": 0.06,
        "participants": 4,
    }


def _build_next_video_result(tmp_path: Path) -> NextVideoEvaluationResult:
    run_dir = tmp_path / "next_video"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.jsonl"
    qa_log_path = run_dir / "qa.log"
    metrics = _sample_next_video_metrics()
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    predictions_path.write_text("", encoding="utf-8")
    qa_log_path.write_text("", encoding="utf-8")
    return NextVideoEvaluationResult(
        run_dir=run_dir,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        qa_log_path=qa_log_path,
        metrics=metrics,
    )


def _build_opinion_result(tmp_path: Path) -> OpinionEvaluationResult:
    study_dir = tmp_path / "opinion" / DEFAULT_SPECS[0].key
    study_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = study_dir / "metrics.json"
    predictions_path = study_dir / "predictions.jsonl"
    qa_log_path = study_dir / "qa.log"
    metrics = _sample_opinion_metrics()
    baseline = {"mae_after": 0.25, "direction_accuracy": 0.55, "eligible": 4}
    metrics_payload = {"metrics": metrics, "baseline": baseline}
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
    predictions_path.write_text("", encoding="utf-8")
    qa_log_path.write_text("", encoding="utf-8")
    combined = _sample_opinion_metrics()
    combined_path = tmp_path / "opinion" / "combined_metrics.json"
    combined_path.write_text(json.dumps({"metrics": combined}), encoding="utf-8")
    files = OpinionStudyFiles(
        metrics=metrics_path,
        predictions=predictions_path,
        qa_log=qa_log_path,
    )
    summary = OpinionStudySummary(
        metrics=metrics,
        baseline=baseline,
        participants=metrics["participants"],
        eligible=metrics["eligible"],
    )
    study_result = OpinionStudyResult(
        study=DEFAULT_SPECS[0],
        files=files,
        summary=summary,
    )
    return OpinionEvaluationResult(studies=[study_result], combined_metrics=combined)


def test_generate_reports_writes_customised_markdown(tmp_path):
    repo_root = tmp_path
    next_result = _build_next_video_result(tmp_path)
    opinion_result = _build_opinion_result(tmp_path)

    generate_reports(
        repo_root=repo_root,
        next_video=next_result,
        opinion=opinion_result,
        options=ReportOptions(
            reports_subdir="custom_rl",
            baseline_label="Custom RL",
            regenerate_hint="Regenerate via scripts/run-custom.sh.",
        ),
    )

    catalog = repo_root / "reports" / "custom_rl" / "README.md"
    next_readme = repo_root / "reports" / "custom_rl" / "next_video" / "README.md"
    opinion_readme = repo_root / "reports" / "custom_rl" / "opinion" / "README.md"

    assert catalog.exists()
    assert next_readme.exists()
    assert opinion_readme.exists()
    assert "Custom RL Next-Video Baseline" in next_readme.read_text(encoding="utf-8")
    assert "Regenerate via scripts/run-custom.sh." in catalog.read_text(encoding="utf-8")


def test_generate_reports_without_hint(tmp_path):
    repo_root = tmp_path
    next_result = _build_next_video_result(tmp_path)
    opinion_result = _build_opinion_result(tmp_path)

    generate_reports(
        repo_root=repo_root,
        next_video=next_result,
        opinion=opinion_result,
    )

    catalog = repo_root / "reports" / "rlhf" / "README.md"
    contents = catalog.read_text(encoding="utf-8")
    assert "Regenerate via" not in contents
    assert "RLHF Next-Video Baseline" in (repo_root / "reports" / "rlhf" / "next_video" / "README.md").read_text(
        encoding="utf-8"
    )
