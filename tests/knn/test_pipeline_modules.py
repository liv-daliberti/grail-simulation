"""Unit tests for the modular KNN pipeline helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping

import pytest

from knn import pipeline_data as data
from knn import pipeline_reports as reports
from knn import pipeline_sweeps as sweeps
from knn.pipeline_context import (
    OpinionStudySelection,
    OpinionSweepOutcome,
    ReportBundle,
    StudySelection,
    StudySpec,
    SweepConfig,
    SweepOutcome,
    SweepTaskContext,
)


def _make_study(key: str, issue: str, label: str) -> StudySpec:
    return StudySpec(key=key, issue=issue, label=label)


def _make_sweep_config(feature_space: str, metric: str = "cosine") -> SweepConfig:
    return SweepConfig(feature_space=feature_space, metric=metric, text_fields=())


def test_resolve_studies_accepts_issue_tokens() -> None:
    studies = data.resolve_studies(["minimum_wage"])
    assert [spec.key for spec in studies] == ["study2", "study3"]

    all_studies = data.resolve_studies(["all"])
    assert len(all_studies) == len(data.study_specs())

    with pytest.raises(ValueError):
        data.resolve_studies(["unknown_token"])


def test_prepare_sweep_tasks_reuses_cached_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    study = _make_study("study1", "gun_control", "Study 1 – Gun Control (MTurk)")
    config = _make_sweep_config("tfidf")
    sweep_dir = tmp_path / "sweeps"
    base_cli: List[str] = ["--dataset", "stub"]
    extra_cli: List[str] = ["--eval-max", "50"]

    issue_slug = data.issue_slug_for_study(study)
    run_root = sweep_dir / config.feature_space / study.study_slug / config.label()
    metrics_path = run_root / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}", encoding="utf-8")

    stub_metrics: Mapping[str, object] = {
        "accuracy_overall": 0.78,
        "n_eligible": 512,
        "best_k": 5,
        "baseline_most_frequent_gold_index": {"accuracy": 0.40},
    }
    monkeypatch.setattr(
        sweeps,
        "load_metrics",
        lambda run_dir, slug: (stub_metrics, metrics_path),
    )

    context = SweepTaskContext(
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=sweep_dir,
        word2vec_model_base=tmp_path / "w2v",
    )

    pending, cached = sweeps.prepare_sweep_tasks(
        studies=[study],
        configs=[config],
        context=context,
        reuse_existing=True,
    )
    assert pending == []
    assert len(cached) == 1
    outcome = cached[0]
    assert isinstance(outcome, SweepOutcome)
    assert outcome.feature_space == "tfidf"
    assert outcome.metrics_path == metrics_path
    assert outcome.accuracy == pytest.approx(0.78)
    assert outcome.best_k == 5


def test_prepare_opinion_sweep_tasks_reuses_cached_metrics(tmp_path: Path) -> None:
    study = _make_study("study1", "gun_control", "Study 1 – Gun Control (MTurk)")
    config = _make_sweep_config("tfidf")
    base_cli: List[str] = ["--dataset", "stub", "--task", "opinion"]
    extra_cli: List[str] = []
    sweep_dir = tmp_path / "sweeps"

    run_root = sweep_dir / "opinion" / config.feature_space / study.study_slug / config.label()
    metrics_path = (
        run_root
        / "opinion"
        / config.feature_space
        / study.key
        / f"opinion_knn_{study.key}_validation_metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "best_metrics": {"mae_after": 0.42, "rmse_after": 0.64, "r2_after": 0.11},
        "baseline": {"mae_using_before": 0.55},
        "n_participants": 200,
        "best_k": 7,
    }
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")

    context = SweepTaskContext(
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=sweep_dir,
        word2vec_model_base=tmp_path / "w2v",
    )

    pending, cached = sweeps.prepare_opinion_sweep_tasks(
        studies=[study],
        configs=[config],
        context=context,
        reuse_existing=True,
    )
    assert pending == []
    assert len(cached) == 1
    outcome = cached[0]
    assert isinstance(outcome, OpinionSweepOutcome)
    assert outcome.mae == pytest.approx(0.42)
    assert outcome.best_k == 7


def test_generate_reports_creates_expected_sections(tmp_path: Path) -> None:
    repo_root = tmp_path
    studies = [
        _make_study("study1", "gun_control", "Study 1 – Gun Control (MTurk)"),
        _make_study("study2", "minimum_wage", "Study 2 – Minimum Wage (MTurk)"),
    ]

    def make_outcome(feature_space: str, study: StudySpec, accuracy: float, tag: str) -> SweepOutcome:
        config = SweepConfig(feature_space=feature_space, metric="cosine", text_fields=())
        return SweepOutcome(
            order_index=0,
            study=study,
            feature_space=feature_space,
            config=config,
            accuracy=accuracy,
            best_k=3,
            eligible=150,
            metrics_path=tmp_path / f"{feature_space}_{study.key}_{tag}.json",
            metrics={
                "accuracy_overall": accuracy,
                "n_eligible": 150,
                "best_k": 3,
                "baseline_most_frequent_gold_index": {"accuracy": 0.4},
                "dataset": "sim_dataset",
                "split": "validation",
            },
        )

    selections: Dict[str, Dict[str, StudySelection]] = {}
    sweep_outcomes: List[SweepOutcome] = []
    metrics_by_feature: Dict[str, Dict[str, Mapping[str, object]]] = {}
    feature_spaces = ("tfidf", "word2vec", "sentence_transformer")

    for feature in feature_spaces:
        per_study: Dict[str, StudySelection] = {}
        per_metrics: Dict[str, Mapping[str, object]] = {}
        for idx, study in enumerate(studies, start=1):
            outcome = make_outcome(feature, study, 0.75 + idx * 0.01, f"{feature}_{study.key}")
            sweep_outcomes.append(outcome)
            per_study[study.key] = StudySelection(study=study, outcome=outcome)
            per_metrics[study.key] = outcome.metrics
        selections[feature] = per_study
        metrics_by_feature[feature] = per_metrics

    opinion_metrics = {
        "tfidf": {
            "study1": {
                "label": "Study 1 – Gun Control (MTurk)",
                "n_participants": 180,
                "metrics": {"mae_after": 1.3, "rmse_after": 1.9, "r2_after": 0.42},
                "baseline": {"mae_using_before": 1.6},
            }
        }
    }

    bundle = ReportBundle(
        selections=selections,
        sweep_outcomes=sweep_outcomes,
        studies=studies,
        metrics_by_feature=metrics_by_feature,
        opinion_metrics=opinion_metrics,
        k_sweep="1,2,3",
        feature_spaces=feature_spaces,
        sentence_model="sentence-transformers/all-mpnet-base-v2",
        allow_incomplete=False,
    )

    reports.generate_reports(repo_root, bundle)

    reports_root = repo_root / "reports" / "knn"
    hyper = (reports_root / "hyperparameter_tuning" / "README.md").read_text(encoding="utf-8")
    assert "TFIDF" in hyper and "WORD2VEC" in hyper and "SENTENCE_TRANSFORMER" in hyper
    assert "### Configuration Leaderboards" in hyper
    assert "Study 1 – Gun Control (MTurk)" in hyper

    next_video = (reports_root / "next_video" / "README.md").read_text(encoding="utf-8")
    assert "## Observations" in next_video

    opinion = (reports_root / "opinion" / "README.md").read_text(encoding="utf-8")
    assert "MAE ↓" in opinion
