"""Unit tests for the modular KNN pipeline helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping

import pytest

from common.prompt_docs import DEFAULT_EXTRA_TEXT_FIELDS
from common.opinion_sweep_types import AccuracySummary, MetricsArtifact

from knn import pipeline_cli
from knn import pipeline_data as data
from knn import pipeline_evaluate as evaluate
from knn import pipeline_reports as reports
from knn import pipeline_sweeps as sweeps
from knn.pipeline_context import (
    EvaluationContext,
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
    return SweepConfig(
        feature_space=feature_space,
        metric=metric,
        text_fields=DEFAULT_EXTRA_TEXT_FIELDS,
    )


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


def test_build_pipeline_context_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    monkeypatch.setenv("KNN_REUSE_SWEEPS", "0")
    for key in (
        "DATASET",
        "OUT_DIR",
        "CACHE_DIR",
        "KNN_STUDIES",
        "KNN_PIPELINE_TASKS",
        "KNN_FEATURE_SPACES",
        "KNN_REUSE_FINAL",
        "KNN_ISSUES",
    ):
        monkeypatch.delenv(key, raising=False)

    args, extra = pipeline_cli.parse_args(["--tasks", "opinion"])
    assert extra == []

    context = pipeline_cli.build_pipeline_context(args, repo_root)

    assert context.run_next_video is False
    assert context.run_opinion is True
    assert context.reuse_sweeps is False
    assert context.reuse_final is False
    assert context.out_dir == repo_root / "models" / "knn"
    assert context.dataset == str(repo_root / "data" / "cleaned_grail")
    assert context.feature_spaces == ("tfidf", "word2vec", "sentence_transformer")


def test_prepare_sweep_tasks_word2vec_without_cached_metrics(tmp_path: Path) -> None:
    study = _make_study("study1", "gun_control", "Study 1 – Gun Control (MTurk)")
    config = SweepConfig(
        feature_space="word2vec",
        metric="cosine",
        text_fields=DEFAULT_EXTRA_TEXT_FIELDS,
        word2vec_size=128,
        word2vec_window=5,
        word2vec_min_count=1,
        word2vec_epochs=5,
        word2vec_workers=2,
    )
    sweep_dir = tmp_path / "sweeps"
    context = SweepTaskContext(
        base_cli=["--dataset", "stub"],
        extra_cli=[],
        sweep_dir=sweep_dir,
        word2vec_model_base=tmp_path / "word2vec",
    )

    pending, cached = sweeps.prepare_sweep_tasks(
        studies=[study],
        configs=[config],
        context=context,
        reuse_existing=True,
    )

    assert cached == []
    assert len(pending) == 1

    task = pending[0]
    issue_slug = data.issue_slug_for_study(study)
    expected_model_dir = context.word2vec_model_base / "sweeps" / study.study_slug / config.label()
    expected_metrics_path = (
        sweep_dir
        / config.feature_space
        / study.study_slug
        / config.label()
        / issue_slug
        / f"knn_eval_{issue_slug}_validation_metrics.json"
    )

    assert task.word2vec_model_dir == expected_model_dir
    assert task.metrics_path == expected_metrics_path


def test_sweep_outcome_from_metrics_handles_legacy_payload(tmp_path: Path) -> None:
    study = _make_study("study1", "gun_control", "Study 1 – Gun Control (MTurk)")
    config = _make_sweep_config("tfidf")
    context = SweepTaskContext(
        base_cli=["--dataset", "stub"],
        extra_cli=[],
        sweep_dir=tmp_path,
        word2vec_model_base=tmp_path / "word2vec",
    )
    pending, _ = sweeps.prepare_sweep_tasks(
        studies=[study],
        configs=[config],
        context=context,
        reuse_existing=False,
    )
    task = pending[0]

    legacy_metrics = {"accuracy_overall": 0.65, "best_k": "7", "baseline_most_frequent_gold_index": {}}
    outcome = sweeps.sweep_outcome_from_metrics(task, legacy_metrics, tmp_path / "metrics.json")
    assert outcome.accuracy == pytest.approx(0.65)
    assert outcome.best_k == 7
    assert outcome.eligible == 0  # fallback path


def test_opinion_sweep_outcome_from_metrics_handles_partial_payload(tmp_path: Path) -> None:
    study = _make_study("study1", "gun_control", "Study 1 – Gun Control (MTurk)")
    config = SweepConfig(
        feature_space="tfidf",
        metric="cosine",
        text_fields=DEFAULT_EXTRA_TEXT_FIELDS,
    )
    context = SweepTaskContext(
        base_cli=["--dataset", "stub"],
        extra_cli=[],
        sweep_dir=tmp_path,
        word2vec_model_base=tmp_path / "word2vec",
    )
    task = sweeps.build_opinion_task(
        index=0,
        config=config,
        study=study,
        context=context,
    )
    partial_metrics = {
        "best_metrics": {"rmse_after": 0.9},
        "best_k": 11,
        "n_participants": 42,
        "baseline": {},
    }
    outcome = sweeps.opinion_sweep_outcome_from_metrics(
        task,
        partial_metrics,
        tmp_path / "opinion_metrics.json",
    )
    assert outcome.best_k == 11
    assert outcome.rmse == pytest.approx(0.9)
    assert outcome.mae == float("inf")  # fallback when mae missing
    assert outcome.participants == 42


def test_select_best_configs_tie_breakers() -> None:
    study_a = _make_study("studyA", "issueA", "Study A")
    study_b = _make_study("studyB", "issueB", "Study B")
    config = _make_sweep_config("tfidf")

    def _outcome(accuracy: float, eligible: int, best_k: int, order: int, study: StudySpec) -> SweepOutcome:
        return SweepOutcome(
            order_index=order,
            study=study,
            feature_space="tfidf",
            config=config,
            accuracy=accuracy,
            best_k=best_k,
            eligible=eligible,
            metrics_path=Path("metrics.json"),
            metrics={"accuracy_overall": accuracy, "best_k": best_k, "n_eligible": eligible},
        )

    outcomes = [
        _outcome(0.6, 80, 11, 0, study_a),
        _outcome(0.6, 90, 13, 1, study_a),
        _outcome(0.7, 50, 5, 2, study_b),
        _outcome(0.69, 100, 3, 3, study_b),
    ]

    selections = sweeps.select_best_configs(
        outcomes=outcomes,
        studies=[study_a, study_b],
        allow_incomplete=False,
    )

    sel_a = selections["tfidf"][study_a.key]
    sel_b = selections["tfidf"][study_b.key]

    assert sel_a.best_k == 13
    assert sel_a.accuracy == pytest.approx(0.6)
    assert sel_b.best_k == 5


def test_select_best_configs_allow_incomplete_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    study_a = _make_study("studyA", "issueA", "Study A")
    study_b = _make_study("studyB", "issueB", "Study B")
    config = _make_sweep_config("tfidf")
    outcome = SweepOutcome(
        order_index=0,
        study=study_a,
        feature_space="tfidf",
        config=config,
        accuracy=0.5,
        best_k=7,
        eligible=10,
        metrics_path=Path("metrics.json"),
        metrics={"accuracy_overall": 0.5, "best_k": 7, "n_eligible": 10},
    )

    with caplog.at_level("WARNING", logger="knn.pipeline.sweeps"):
        selections = sweeps.select_best_configs(
            outcomes=[outcome],
            studies=[study_a, study_b],
            allow_incomplete=True,
        )
    assert study_a.key in selections["tfidf"]
    assert "Missing sweep selections" in "\n".join(caplog.messages)

    with pytest.raises(RuntimeError):
        sweeps.select_best_configs(
            outcomes=[outcome],
            studies=[study_a, study_b],
            allow_incomplete=False,
        )


def test_select_best_opinion_configs_tie_breakers() -> None:
    study = _make_study("studyA", "issueA", "Study A")
    config = _make_sweep_config("tfidf")

    def _opinion(
        *,
        mae: float,
        rmse: float,
        participants: int,
        best_k: int,
        order: int,
    ) -> OpinionSweepOutcome:
        return OpinionSweepOutcome(
            order_index=order,
            study=study,
            config=config,
            feature_space="tfidf",
            mae=mae,
            rmse=rmse,
            r2_score=0.0,
            baseline_mae=None,
            mae_delta=None,
            best_k=best_k,
            participants=participants,
            artifact=MetricsArtifact(
                path=Path("metrics.json"),
                payload={"best_metrics": {"mae_after": mae, "rmse_after": rmse}, "best_k": best_k},
            ),
            accuracy_summary=AccuracySummary(),
        )

    outcomes = [
        _opinion(mae=0.45, rmse=0.9, participants=40, best_k=9, order=0),
        _opinion(mae=0.45, rmse=0.9, participants=42, best_k=11, order=1),
        _opinion(mae=0.47, rmse=0.85, participants=50, best_k=3, order=2),
    ]

    selections = sweeps.select_best_opinion_configs(
        outcomes=outcomes,
        studies=[study],
        allow_incomplete=False,
    )
    result = selections["tfidf"][study.key]
    assert result.best_k == 11
    assert result.outcome.participants == 42


def test_select_best_opinion_configs_allow_incomplete(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    study_a = _make_study("studyA", "issueA", "Study A")
    study_b = _make_study("studyB", "issueB", "Study B")
    config = _make_sweep_config("tfidf")
    outcome = OpinionSweepOutcome(
        order_index=0,
        study=study_a,
        config=config,
        feature_space="tfidf",
        mae=0.4,
        rmse=1.0,
        r2_score=0.0,
        baseline_mae=None,
        mae_delta=None,
        best_k=5,
        participants=100,
        artifact=MetricsArtifact(
            path=Path("metrics.json"),
            payload={"best_metrics": {"mae_after": 0.4, "rmse_after": 1.0}, "best_k": 5},
        ),
        accuracy_summary=AccuracySummary(),
    )

    with caplog.at_level("WARNING", logger="knn.pipeline.sweeps"):
        selections = sweeps.select_best_opinion_configs(
            outcomes=[outcome],
            studies=[study_a, study_b],
            allow_incomplete=True,
        )
    assert study_a.key in selections["tfidf"]
    assert "Missing opinion sweep selections" in "\n".join(caplog.messages)

    with pytest.raises(RuntimeError):
        sweeps.select_best_opinion_configs(
            outcomes=[outcome],
            studies=[study_a, study_b],
            allow_incomplete=False,
        )


def test_run_final_evaluations_reuses_cached_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    study = _make_study("study1", "gun_control", "Study 1 – Gun Control (MTurk)")
    config = _make_sweep_config("word2vec")
    outcome = SweepOutcome(
        order_index=0,
        study=study,
        feature_space="word2vec",
        config=config,
        accuracy=0.75,
        best_k=15,
        eligible=120,
        metrics_path=tmp_path / "dummy.json",
        metrics={"accuracy_overall": 0.75},
    )
    selections = {"word2vec": {study.key: StudySelection(study=study, outcome=outcome)}}

    out_dir = tmp_path / "out"
    context = EvaluationContext.from_args(
        base_cli=["--dataset", "stub"],
        extra_cli=[],
        out_dir=out_dir,
        word2vec_model_dir=tmp_path / "word2vec_models",
        reuse_existing=True,
    )

    issue_slug = data.issue_slug_for_study(study)
    metrics_path = (
        out_dir / "word2vec" / study.study_slug / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}", encoding="utf-8")

    stub_metrics = {"accuracy_overall": 0.88}
    load_calls: list[tuple[Path, str]] = []

    def fake_load(run_dir: Path, slug: str) -> tuple[dict, Path]:
        load_calls.append((run_dir, slug))
        assert run_dir == out_dir / "word2vec" / study.study_slug
        assert slug == issue_slug
        return stub_metrics, metrics_path

    monkeypatch.setattr(evaluate, "load_metrics", fake_load)
    monkeypatch.setattr(evaluate, "run_knn_cli", lambda *_args, **_kwargs: pytest.fail("run_knn_cli should not be called"))

    results = evaluate.run_final_evaluations(
        selections=selections,
        studies=[study],
        context=context,
    )

    assert results == {"word2vec": {study.key: stub_metrics}}
    assert load_calls == [(out_dir / "word2vec" / study.study_slug, issue_slug)]
    assert (context.word2vec_model_dir / study.study_slug).exists()


def test_run_opinion_evaluations_reuses_cached_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    study = _make_study("study1", "gun_control", "Study 1 – Gun Control (MTurk)")
    config = _make_sweep_config("tfidf")
    outcome = OpinionSweepOutcome(
        order_index=0,
        study=study,
        config=config,
        feature_space="tfidf",
        mae=0.42,
        rmse=0.9,
        r2_score=0.1,
        baseline_mae=None,
        mae_delta=None,
        best_k=9,
        participants=200,
        artifact=MetricsArtifact(
            path=tmp_path / "dummy.json",
            payload={"best_metrics": {"mae_after": 0.42}, "best_k": 9},
        ),
        accuracy_summary=AccuracySummary(),
    )
    selections = {"tfidf": {study.key: OpinionStudySelection(study=study, outcome=outcome)}}

    context = EvaluationContext.from_args(
        base_cli=["--dataset", "stub"],
        extra_cli=[],
        out_dir=tmp_path / "out",
        word2vec_model_dir=tmp_path / "word2vec_models",
        reuse_existing=True,
    )

    cached_payload = {study.key: {"mae_after": 0.33}}
    call_counter = {"count": 0}

    def fake_load_opinion(out_dir: Path, feature_space: str) -> dict:
        call_counter["count"] += 1
        assert out_dir == context.out_dir
        assert feature_space == "tfidf"
        return cached_payload

    monkeypatch.setattr(evaluate, "load_opinion_metrics", fake_load_opinion)
    monkeypatch.setattr(evaluate, "run_knn_cli", lambda *_args, **_kwargs: pytest.fail("run_knn_cli should not be called"))

    results = evaluate.run_opinion_evaluations(
        selections=selections,
        studies=[study],
        context=context,
    )

    assert call_counter["count"] == 2
    assert results == {"tfidf": cached_payload}


def test_run_cross_study_evaluations_single_study_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    study = _make_study("study1", "gun_control", "Study 1 – Gun Control (MTurk)")
    config = _make_sweep_config("tfidf")
    outcome = SweepOutcome(
        order_index=0,
        study=study,
        feature_space="tfidf",
        config=config,
        accuracy=0.6,
        best_k=7,
        eligible=50,
        metrics_path=tmp_path / "dummy.json",
        metrics={"accuracy_overall": 0.6},
    )
    selections = {"tfidf": {study.key: StudySelection(study=study, outcome=outcome)}}
    context = EvaluationContext.from_args(
        base_cli=["--dataset", "stub"],
        extra_cli=[],
        out_dir=tmp_path / "out",
        word2vec_model_dir=tmp_path / "word2vec_models",
        reuse_existing=True,
    )

    monkeypatch.setattr(evaluate, "load_loso_metrics_from_disk", lambda **_kwargs: {})
    monkeypatch.setattr(evaluate, "run_knn_cli", lambda *_args, **_kwargs: pytest.fail("run_knn_cli should not be called"))
    monkeypatch.setattr(evaluate, "load_metrics", lambda *_args, **_kwargs: pytest.fail("load_metrics should not be called"))

    with caplog.at_level("WARNING", logger="knn.pipeline.evaluate"):
        results = evaluate.run_cross_study_evaluations(
            selections=selections,
            studies=[study],
            context=context,
        )

    assert results == {}
    assert any("no alternate studies" in message.lower() for message in caplog.messages)


def test_run_cross_study_evaluations_reuses_cached_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    study_a = _make_study("studyA", "issueA", "Study A")
    study_b = _make_study("studyB", "issueB", "Study B")
    config = _make_sweep_config("tfidf")
    outcome_a = SweepOutcome(
        order_index=0,
        study=study_a,
        feature_space="tfidf",
        config=config,
        accuracy=0.7,
        best_k=9,
        eligible=100,
        metrics_path=tmp_path / "a.json",
        metrics={"accuracy_overall": 0.7},
    )
    outcome_b = SweepOutcome(
        order_index=1,
        study=study_b,
        feature_space="tfidf",
        config=config,
        accuracy=0.68,
        best_k=11,
        eligible=95,
        metrics_path=tmp_path / "b.json",
        metrics={"accuracy_overall": 0.68},
    )

    selections = {
        "tfidf": {
            study_a.key: StudySelection(study=study_a, outcome=outcome_a),
            study_b.key: StudySelection(study=study_b, outcome=outcome_b),
        }
    }

    cached_metrics = {
        "tfidf": {
            study_a.key: {"accuracy_overall": 0.65},
            study_b.key: {"accuracy_overall": 0.62},
        }
    }

    context = EvaluationContext.from_args(
        base_cli=["--dataset", "stub"],
        extra_cli=[],
        out_dir=tmp_path / "out",
        word2vec_model_dir=tmp_path / "word2vec_models",
        reuse_existing=True,
    )

    monkeypatch.setattr(evaluate, "load_loso_metrics_from_disk", lambda **_kwargs: cached_metrics)
    monkeypatch.setattr(evaluate, "run_knn_cli", lambda *_args, **_kwargs: pytest.fail("run_knn_cli should not be called"))
    monkeypatch.setattr(evaluate, "load_metrics", lambda *_args, **_kwargs: pytest.fail("load_metrics should not be called"))

    results = evaluate.run_cross_study_evaluations(
        selections=selections,
        studies=[study_a, study_b],
        context=context,
    )

    assert results == cached_metrics


def test_generate_reports_creates_expected_sections(tmp_path: Path) -> None:
    repo_root = tmp_path
    studies = [
        _make_study("study1", "gun_control", "Study 1 – Gun Control (MTurk)"),
        _make_study("study2", "minimum_wage", "Study 2 – Minimum Wage (MTurk)"),
    ]

    def make_outcome(feature_space: str, study: StudySpec, accuracy: float, tag: str) -> SweepOutcome:
        config = SweepConfig(
            feature_space=feature_space,
            metric="cosine",
            text_fields=DEFAULT_EXTRA_TEXT_FIELDS,
        )
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
