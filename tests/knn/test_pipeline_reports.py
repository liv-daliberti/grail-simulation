"""Regression tests for the modular KNN pipeline report builders."""

from __future__ import annotations

from pathlib import Path

from src.common.pipeline.models import StudySpec
from src.common.opinion.sweep_types import AccuracySummary, MetricsArtifact
from src.knn.pipeline.context import (
    OpinionStudySelection,
    OpinionSweepOutcome,
    ReportBundle,
    StudySelection,
    SweepConfig,
    SweepOutcome,
)
from src.knn.pipeline.reports import generate_reports
from src.knn.pipeline.reports.next_video import (
    NextVideoReportInputs,
    _build_next_video_report,
)
from src.knn.pipeline.reports.shared import parse_k_sweep


def test_parse_k_sweep_handles_strings_and_iterables() -> None:
    """Mixed delimiters and iterables should normalise into integer tuples."""
    assert parse_k_sweep("5, 10 / 15") == (5, 10, 15)
    assert parse_k_sweep([3, "7", 9.0]) == (3, 7, 9)


def test_generate_reports_writes_expected_markdown(tmp_path: Path, sample_png: Path) -> None:
    """End-to-end report generation should emit the expected Markdown skeleton."""
    repo_root = tmp_path

    study = StudySpec(key="study_a", issue="issue_a", label="Study A")
    sweep_config = SweepConfig(
        feature_space="tfidf",
        metric="cosine",
        text_fields=("title",),
    )

    next_video_metrics = {
        "dataset": "data/mock_dataset",
        "split": "validation",
        "accuracy_overall": 0.72,
        "best_k": 5,
        "n_total": 100,
        "n_eligible": 90,
        "baseline_most_frequent_gold_index": {"accuracy": 0.6},
        "random_baseline_expected_accuracy": 0.21,
    }
    sweep_outcome = SweepOutcome(
        order_index=1,
        study=study,
        feature_space="tfidf",
        config=sweep_config,
        accuracy=0.72,
        best_k=5,
        eligible=90,
        metrics_path=repo_root / "sweep_metrics.json",
        metrics=next_video_metrics,
    )
    study_selection = StudySelection(study=study, outcome=sweep_outcome)

    opinion_metrics = {
        "dataset": "data/mock_dataset",
        "split": "validation",
        "best_metrics": {
            "mae_after": 0.45,
            "mae_change": 0.04,
            "rmse_after": 0.62,
            "rmse_change": 0.31,
            "r2_after": 0.11,
            "direction_accuracy": 0.58,
            "calibration_slope": 0.92,
            "calibration_intercept": 0.01,
            "calibration_ece": 0.05,
            "kl_divergence_change": 0.15,
            "eligible": 48,
        },
        "baseline": {
            "mae_using_before": 0.5,
            "rmse_change_zero": 0.4,
            "direction_accuracy": 0.5,
            "calibration_slope_change_zero": 0.0,
            "calibration_intercept_change_zero": 0.0,
            "calibration_ece_change_zero": 0.08,
            "kl_divergence_change_zero": 0.25,
        },
        "n_participants": 50,
        "best_k": 5,
    }
    opinion_outcome = OpinionSweepOutcome(
        order_index=1,
        study=study,
        config=sweep_config,
        feature_space="tfidf",
        mae=0.45,
        rmse=0.62,
        r2_score=0.11,
        baseline_mae=0.5,
        mae_delta=0.05,
        best_k=5,
        participants=50,
        artifact=MetricsArtifact(
            path=repo_root / "opinion_metrics.json",
            payload=opinion_metrics,
        ),
        accuracy_summary=AccuracySummary(
            value=0.58,
            baseline=0.5,
            delta=0.08,
            eligible=48,
        ),
    )
    opinion_selection = OpinionStudySelection(study=study, outcome=opinion_outcome)

    plots_dir = repo_root / "reports" / "knn" / "tfidf" / "opinion"
    plots_dir_next = repo_root / "reports" / "knn" / "tfidf" / "opinion_from_next"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_dir_next.mkdir(parents=True, exist_ok=True)
    png_bytes = sample_png.read_bytes()
    for stem in ("mae_study_a", "r2_study_a", "change_heatmap_study_a"):
        (plots_dir / f"{stem}.png").write_bytes(png_bytes)
        (plots_dir_next / f"{stem}.png").write_bytes(png_bytes)

    bundle = ReportBundle(
        selections={"tfidf": {study.key: study_selection}},
        sweep_outcomes=[sweep_outcome],
        opinion_selections={"tfidf": {study.key: opinion_selection}},
        opinion_sweep_outcomes=[opinion_outcome],
        studies=[study],
        metrics_by_feature={"tfidf": {study.key: next_video_metrics}},
        opinion_metrics={"tfidf": {study.key: opinion_metrics}},
        opinion_from_next_metrics={"tfidf": {study.key: opinion_metrics}},
        k_sweep="3, 5 / 7",
        loso_metrics={"tfidf": {study.key: next_video_metrics}},
        feature_spaces=("tfidf",),
        sentence_model="sentence-transformers/all-mpnet-base-v2",
        allow_incomplete=False,
        include_next_video=True,
        include_opinion=True,
        include_opinion_from_next=True,
    )

    generate_reports(repo_root, bundle)

    catalog_path = repo_root / "reports" / "knn" / "README.md"
    hyper_path = repo_root / "reports" / "knn" / "hyperparameter_tuning" / "README.md"
    next_video_path = repo_root / "reports" / "knn" / "next_video" / "README.md"
    opinion_path = repo_root / "reports" / "knn" / "opinion" / "README.md"
    opinion_next_path = repo_root / "reports" / "knn" / "opinion_from_next" / "README.md"
    features_path = repo_root / "reports" / "knn" / "additional_features" / "README.md"

    assert catalog_path.exists()
    assert hyper_path.exists()
    assert next_video_path.exists()
    assert opinion_path.exists()
    assert opinion_next_path.exists()
    assert features_path.exists()

    assert "# KNN Report Catalog" in catalog_path.read_text()

    hyper_text = hyper_path.read_text()
    assert "# Hyper-Parameter Sweep Results" in hyper_text
    assert "Study A" in hyper_text

    next_video_text = next_video_path.read_text()
    assert "KNN Next-Video Baseline" in next_video_text
    assert "Study A" in next_video_text

    opinion_text = opinion_path.read_text()
    assert "KNN Opinion Shift Study" in opinion_text
    assert "Study A" in opinion_text
    assert "RMSE (change)" in opinion_text
    assert "calibration slope &" in opinion_text
    assert "![Mae Study A](../tfidf/opinion/mae_study_a.png)" in opinion_text

    opinion_next_text = opinion_next_path.read_text()
    assert "Next-Video Config" in opinion_next_text
    assert "![Mae Study A](../tfidf/opinion_from_next/mae_study_a.png)" in opinion_next_text

    features_text = features_path.read_text()
    assert "Additional Text Features" in features_text
    assert "`title`" in features_text
    assert "`viewer_profile`" in features_text


def test_next_video_report_writes_placeholder_when_metrics_missing(tmp_path: Path) -> None:
    """`--allow-incomplete` should emit a placeholder when slate metrics are absent."""
    inputs = NextVideoReportInputs(
        output_dir=tmp_path,
        metrics_by_feature={},
        studies=(),
        feature_spaces=(),
        allow_incomplete=True,
    )

    _build_next_video_report(inputs)

    readme = tmp_path / "README.md"
    assert readme.exists()
    text = readme.read_text()
    assert "Next-video slate metrics are not available yet." in text
    assert "allow-incomplete" in text
