#!/usr/bin/env python
# pylint: disable=missing-function-docstring
"""Regression tests for GRAIL pipeline reporting outputs."""

import json
from pathlib import Path

import grail.pipeline as grail_pipeline
from grail.reports import generate_reports as generate_grail_reports

from common.opinion import DEFAULT_SPECS
from grpo.next_video import NextVideoEvaluationResult
from grpo.opinion import (
    OpinionEvaluationResult,
    OpinionStudyFiles,
    OpinionStudyResult,
    OpinionStudySummary,
)


def _sample_next_video_metrics() -> dict:
    return {
        "accuracy_overall": 0.6,
        "parsed_rate": 1.0,
        "format_rate": 0.98,
        "n_eligible": 10,
        "n_total": 10,
        "group_metrics": {
            "by_issue": {
                "gun_control": {
                    "n_seen": 5,
                    "n_eligible": 5,
                    "accuracy": 0.8,
                    "parsed_rate": 1.0,
                    "format_rate": 1.0,
                }
            },
            "by_participant_study": {
                "study1": {
                    "n_seen": 5,
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
        "eligible": 3,
        "mae_after": 0.11,
        "mae_change": 0.22,
        "direction_accuracy": 0.67,
        "rmse_after": 0.3,
        "rmse_change": 0.4,
        "calibration_ece": 0.05,
        "participants": 3,
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
    baseline = {"mae_after": 0.2, "direction_accuracy": 0.5, "eligible": 3}
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


def test_generate_reports_writes_markdown(tmp_path):
    repo_root = tmp_path
    next_result = _build_next_video_result(tmp_path)
    opinion_result = _build_opinion_result(tmp_path)

    generate_grail_reports(
        repo_root=repo_root,
        next_video=next_result,
        opinion=opinion_result,
    )

    catalog = repo_root / "reports" / "grail" / "README.md"
    next_readme = repo_root / "reports" / "grail" / "next_video" / "README.md"
    opinion_readme = repo_root / "reports" / "grail" / "opinion" / "README.md"

    assert catalog.exists(), "Catalog README should be created."
    assert next_readme.exists(), "Next-video report should be written."
    assert opinion_readme.exists(), "Opinion report should be written."
    assert "GRAIL Next-Video Baseline" in next_readme.read_text(encoding="utf-8")
    assert "GRAIL Opinion Regression" in opinion_readme.read_text(encoding="utf-8")
    assert "Regenerate via `python -m grail.pipeline" in catalog.read_text(encoding="utf-8")


def test_pipeline_reports_stage_reads_cached_artifacts(tmp_path, monkeypatch):
    label = "demo"
    out_dir = tmp_path / "models" / "grail"
    next_dir = out_dir / "next_video" / label
    opinion_label_dir = out_dir / "opinion" / label / DEFAULT_SPECS[0].key
    next_dir.mkdir(parents=True, exist_ok=True)
    opinion_label_dir.mkdir(parents=True, exist_ok=True)

    next_metrics = _sample_next_video_metrics()
    (next_dir / "metrics.json").write_text(json.dumps(next_metrics), encoding="utf-8")
    (next_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    (next_dir / "qa.log").write_text("", encoding="utf-8")

    opinion_metrics = _sample_opinion_metrics()
    opinion_payload = {"metrics": opinion_metrics, "baseline": {"mae_after": 0.3}}
    (opinion_label_dir / "metrics.json").write_text(
        json.dumps(opinion_payload), encoding="utf-8"
    )
    (opinion_label_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    (opinion_label_dir / "qa.log").write_text("", encoding="utf-8")
    combined_path = out_dir / "opinion" / label / "combined_metrics.json"
    combined_path.write_text(json.dumps({"metrics": opinion_metrics}), encoding="utf-8")

    import grpo.config as grpo_config

    monkeypatch.setattr(grpo_config, "REPO_ROOT", tmp_path)

    grail_pipeline.main(
        [
            "--stage",
            "reports",
            "--out-dir",
            str(out_dir),
            "--label",
            label,
            "--dataset",
            str(tmp_path / "dataset"),
        ]
    )

    summary_catalog = tmp_path / "reports" / "grail" / "README.md"
    assert summary_catalog.exists(), "Reports pipeline should render catalog README."
