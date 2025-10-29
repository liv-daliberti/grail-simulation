"""Unit tests for the XGBoost additional-features report."""

from __future__ import annotations

from pathlib import Path

from src.common.opinion.sweep_types import AccuracySummary, MetricsArtifact
from src.xgb.pipeline_context import (
    OpinionStudySelection,
    OpinionSweepOutcome,
    StudySelection,
    StudySpec,
    SweepConfig,
    SweepOutcome,
)
from src.xgb.pipeline_reports.features import _write_feature_report
from src.xgb.pipeline_reports.runner import OpinionReportData, SweepReportData


def _sweep_config() -> SweepConfig:
    return SweepConfig(
        text_vectorizer="tfidf",
        vectorizer_tag="tfidf",
        learning_rate=0.1,
        max_depth=4,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        vectorizer_cli=(),
    )


def test_feature_report_highlights_additional_fields(tmp_path: Path) -> None:
    """The feature report should list extra fields observed across stages."""
    study = StudySpec(key="study_one", issue="issue_alpha", label="Study One")
    config = _sweep_config()

    sweep_metrics = {
        "xgboost_params": {"extra_fields": ["viewer_profile", "state_text", "issue_summary"]},
    }
    sweep_outcome = SweepOutcome(
        order_index=0,
        study=study,
        config=config,
        accuracy=0.72,
        coverage=0.61,
        evaluated=100,
        metrics_path=tmp_path / "sweep.json",
        metrics=sweep_metrics,
    )
    study_selection = StudySelection(study=study, outcome=sweep_outcome)

    final_metrics = {
        "study_label": study.label,
        "issue": study.issue,
        "xgboost_params": {"extra_fields": ["viewer_profile", "state_text", "issue_summary"]},
    }

    sweeps = SweepReportData(
        outcomes=[sweep_outcome],
        selections={study.key: study_selection},
        final_metrics={study.key: final_metrics},
    )

    opinion_metrics = {
        "study_label": study.label,
        "extra_fields": ["viewer_profile", "state_text", "opinion_notes"],
    }
    opinion_outcome = OpinionSweepOutcome(
        order_index=0,
        study=study,
        config=config,
        mae=0.5,
        rmse=0.7,
        r_squared=0.2,
        artifact=MetricsArtifact(
            path=tmp_path / "opinion.json",
            payload={"extra_fields": ["viewer_profile", "state_text", "opinion_notes"]},
        ),
        accuracy_summary=AccuracySummary(),
    )
    opinion_selection = OpinionStudySelection(study=study, outcome=opinion_outcome)

    opinion = OpinionReportData(
        metrics={study.key: opinion_metrics},
        outcomes=[opinion_outcome],
        selections={study.key: opinion_selection},
    )

    output_dir = tmp_path / "reports" / "xgb" / "additional_features"
    _write_feature_report(
        output_dir,
        sweeps,
        include_next_video=True,
        opinion=opinion,
    )

    readme = output_dir / "README.md"
    assert readme.exists()
    text = readme.read_text()
    assert "Additional Text Features" in text
    assert "`issue_summary`" in text
    assert "`opinion_notes`" in text
