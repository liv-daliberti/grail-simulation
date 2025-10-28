"""High-level integration-style tests for the KNN and XGBoost pipelines.

These tests patch the expensive IO/CLI layers so we can exercise the pipeline
entry points without launching heavy sweeps or evaluations. They focus on the
control flow that ties together stage handling, task selection, and report
generation wiring.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import pytest

from common.prompt_docs import DEFAULT_EXTRA_TEXT_FIELDS
from common.opinion_sweep_types import AccuracySummary, MetricsArtifact

from common.pipeline_stage import prepare_sweep_execution as real_prepare_sweep_execution
from knn import pipeline as knn_pipeline
from knn.pipeline_context import (
    OpinionStudySelection as KnnOpinionStudySelection,
    OpinionSweepOutcome as KnnOpinionSweepOutcome,
    OpinionSweepTask as KnnOpinionSweepTask,
    ReportBundle,
    StudySelection as KnnStudySelection,
    StudySpec as KnnStudySpec,
    SweepConfig as KnnSweepConfig,
    SweepOutcome as KnnSweepOutcome,
    SweepTask as KnnSweepTask,
    SweepTaskContext as KnnSweepTaskContext,
)
from knn.pipeline_data import issue_slug_for_study as knn_issue_slug_for_study
from xgb import pipeline as xgb_pipeline
from xgb.pipeline_context import (
    FinalEvalContext,
    OpinionStageConfig,
    OpinionStudySelection as XgbOpinionStudySelection,
    OpinionSweepOutcome as XgbOpinionSweepOutcome,
    OpinionSweepRunContext,
    OpinionSweepTask as XgbOpinionSweepTask,
    StudySelection as XgbStudySelection,
    StudySpec as XgbStudySpec,
    SweepConfig as XgbSweepConfig,
    SweepOutcome as XgbSweepOutcome,
    SweepRunContext,
    SweepTask as XgbSweepTask,
)
from xgb.pipeline_sweeps import OPINION_FEATURE_SPACE


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _knn_make_sweep_task(
    *,
    study: KnnStudySpec,
    config: KnnSweepConfig,
    context: KnnSweepTaskContext,
    index: int = 0,
) -> KnnSweepTask:
    issue_slug = knn_issue_slug_for_study(study)
    run_root = context.sweep_dir / config.feature_space / study.study_slug / config.label()
    metrics_path = run_root / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}", encoding="utf-8")
    return KnnSweepTask(
        index=index,
        study=study,
        config=config,
        base_cli=tuple(context.base_cli),
        extra_cli=tuple(context.extra_cli),
        run_root=run_root,
        word2vec_model_dir=context.word2vec_model_base
        / "sweeps"
        / study.study_slug
        / config.label(),
        issue=study.issue,
        issue_slug=issue_slug,
        metrics_path=metrics_path,
    )


def _knn_make_opinion_task(
    *,
    study: KnnStudySpec,
    config: KnnSweepConfig,
    context: KnnSweepTaskContext,
    index: int = 0,
) -> KnnOpinionSweepTask:
    run_root = (
        context.sweep_dir
        / "opinion"
        / config.feature_space
        / study.study_slug
        / config.label()
    )
    metrics_path = (
        run_root
        / "opinion"
        / config.feature_space
        / study.key
        / f"opinion_knn_{study.key}_validation_metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}", encoding="utf-8")
    return KnnOpinionSweepTask(
        index=index,
        study=study,
        config=config,
        base_cli=tuple(context.base_cli),
        extra_cli=tuple(context.extra_cli),
        run_root=run_root,
        word2vec_model_dir=context.word2vec_model_base
        / "sweeps_opinion"
        / study.study_slug
        / config.label(),
        metrics_path=metrics_path,
    )


def _xgb_make_sweep_task(
    *,
    study: XgbStudySpec,
    config: XgbSweepConfig,
    context: SweepRunContext,
    index: int = 0,
) -> XgbSweepTask:
    run_root = context.sweep_dir / study.issue_slug / study.study_slug / config.label()
    metrics_path = run_root / study.evaluation_slug / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}", encoding="utf-8")
    return XgbSweepTask(
        index=index,
        study=study,
        config=config,
        base_cli=tuple(context.base_cli),
        extra_cli=tuple(context.extra_cli),
        run_root=run_root,
        tree_method=context.tree_method,
        metrics_path=metrics_path,
    )


def _xgb_make_opinion_task(
    *,
    study: XgbStudySpec,
    config: XgbSweepConfig,
    context: OpinionSweepRunContext,
    index: int = 0,
) -> XgbOpinionSweepTask:
    run_root = context.sweep_dir / study.issue_slug / study.study_slug / config.label()
    metrics_path = (
        run_root
        / OPINION_FEATURE_SPACE
        / study.key
        / f"opinion_xgb_{study.key}_validation_metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}", encoding="utf-8")
    request_args: Dict[str, object] = {
        "dataset": context.dataset,
        "cache_dir": context.cache_dir,
        "out_dir": str(run_root),
        "feature_space": OPINION_FEATURE_SPACE,
        "extra_fields": tuple(context.extra_fields),
        "max_participants": int(context.max_participants),
        "seed": int(context.seed),
        "max_features": context.max_features,
        "tree_method": context.tree_method,
        "overwrite": True,
    }
    return XgbOpinionSweepTask(
        index=index,
        study=study,
        config=config,
        request_args=request_args,
        metrics_path=metrics_path,
    )


# ---------------------------------------------------------------------------
# KNN pipeline tests
# ---------------------------------------------------------------------------


def test_knn_pipeline_plan_and_sweeps_reuse(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The plan stage should enumerate tasks, and the sweeps stage should skip when metrics exist."""

    study = KnnStudySpec(key="study1", issue="gun_control", label="Study 1 – Gun Control")
    config = KnnSweepConfig(
        feature_space="tfidf",
        metric="cosine",
        text_fields=DEFAULT_EXTRA_TEXT_FIELDS,
    )
    captured_plan: Dict[str, Tuple[Sequence[Any], Sequence[Any]]] = {}
    sweep_calls: Dict[str, Any] = {}

    def fake_resolve_studies(_tokens: Sequence[str]) -> List[KnnStudySpec]:
        return [study]

    def fake_build_sweep_configs(_context) -> List[KnnSweepConfig]:
        return [config]

    def fake_prepare_sweep_tasks(
        *,
        studies: Sequence[KnnStudySpec],
        configs: Sequence[KnnSweepConfig],
        context: KnnSweepTaskContext,
        reuse_existing: bool,
    ) -> Tuple[List[KnnSweepTask], List[KnnSweepOutcome]]:
        task = _knn_make_sweep_task(study=studies[0], config=configs[0], context=context)
        sweep_calls.setdefault("sweep_tasks", []).append(task)
        return [task], []

    def fake_prepare_opinion_tasks(
        *,
        studies: Sequence[KnnStudySpec],
        configs: Sequence[KnnSweepConfig],
        context: KnnSweepTaskContext,
        reuse_existing: bool,
    ) -> Tuple[List[KnnOpinionSweepTask], List[KnnOpinionSweepOutcome]]:
        task = _knn_make_opinion_task(study=studies[0], config=configs[0], context=context)
        sweep_calls.setdefault("opinion_tasks", []).append(task)
        return [task], []

    def fake_emit_combined(*, slate_tasks, opinion_tasks) -> None:
        captured_plan["tasks"] = (tuple(slate_tasks), tuple(opinion_tasks))

    monkeypatch.setattr(knn_pipeline, "_resolve_studies", fake_resolve_studies)
    monkeypatch.setattr(knn_pipeline, "_build_sweep_configs", fake_build_sweep_configs)
    monkeypatch.setattr(knn_pipeline, "_prepare_sweep_tasks", fake_prepare_sweep_tasks)
    monkeypatch.setattr(knn_pipeline, "_prepare_opinion_sweep_tasks", fake_prepare_opinion_tasks)
    monkeypatch.setattr(knn_pipeline, "_emit_combined_sweep_plan", fake_emit_combined)
    monkeypatch.setattr(knn_pipeline, "prepare_sweep_execution", lambda **_: 0)
    monkeypatch.setattr(knn_pipeline, "_execute_sweep_task", lambda task: pytest.fail(f"Unexpected sweep execution {task}"))
    monkeypatch.setattr(
        knn_pipeline,
        "_execute_opinion_sweep_task",
        lambda task: pytest.fail(f"Unexpected opinion sweep execution {task}"),
    )

    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    out_dir = tmp_path / "out"
    w2v_dir = tmp_path / "w2v"

    # Plan stage
    knn_pipeline.main(
        [
            "--stage",
            "plan",
            "--dataset",
            str(dataset_path),
            "--out-dir",
            str(out_dir),
            "--word2vec-model-dir",
            str(w2v_dir),
        ]
    )
    assert "tasks" in captured_plan
    slate_tasks, opinion_tasks = captured_plan["tasks"]
    assert len(slate_tasks) == 1
    assert len(opinion_tasks) == 1

    # Sweeps stage – metrics already exist so the execution path should short-circuit
    knn_pipeline.main(
        [
            "--stage",
            "sweeps",
            "--dataset",
            str(dataset_path),
            "--out-dir",
            str(out_dir),
            "--word2vec-model-dir",
            str(w2v_dir),
        ]
    )
    # ensure the helper produced paths during both preparations
    assert sweep_calls["sweep_tasks"][0].metrics_path.exists()
    assert sweep_calls["opinion_tasks"][0].metrics_path.exists()


def test_knn_pipeline_reports_stage_uses_stubbed_data(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Stage=reports should assemble a ReportBundle using cached outcomes and generated metrics."""

    study = KnnStudySpec(key="study1", issue="gun_control", label="Study 1 – Gun Control")
    config = KnnSweepConfig(
        feature_space="tfidf",
        metric="cosine",
        text_fields=DEFAULT_EXTRA_TEXT_FIELDS,
    )
    sweep_metrics_path = tmp_path / "sweep_metrics.json"
    sweep_outcome = KnnSweepOutcome(
        order_index=0,
        study=study,
        feature_space="tfidf",
        config=config,
        accuracy=0.81,
        best_k=5,
        eligible=120,
        metrics_path=sweep_metrics_path,
        metrics={"accuracy_overall": 0.81, "best_k": 5},
    )
    study_selection = KnnStudySelection(study=study, outcome=sweep_outcome)

    opinion_metrics_path = tmp_path / "opinion_metrics.json"
    opinion_outcome = KnnOpinionSweepOutcome(
        order_index=0,
        study=study,
        config=config,
        feature_space="tfidf",
        mae=0.42,
        rmse=0.61,
        r2_score=0.15,
        baseline_mae=0.5,
        mae_delta=-0.08,
        best_k=7,
        participants=80,
        artifact=MetricsArtifact(
            path=opinion_metrics_path,
            payload={"best_metrics": {"mae_after": 0.42}, "best_k": 7},
        ),
        accuracy_summary=AccuracySummary(eligible=80),
    )
    opinion_selection = KnnOpinionStudySelection(study=study, outcome=opinion_outcome)

    captured_bundle: Dict[str, ReportBundle] = {}

    def fake_generate_reports(_root, bundle: ReportBundle) -> None:
        captured_bundle["bundle"] = bundle

    monkeypatch.setattr(knn_pipeline, "_resolve_studies", lambda _tokens: [study])
    monkeypatch.setattr(knn_pipeline, "_build_sweep_configs", lambda _context: [config])
    monkeypatch.setattr(knn_pipeline, "_prepare_sweep_tasks", lambda **_: ([], []))
    monkeypatch.setattr(knn_pipeline, "_prepare_opinion_sweep_tasks", lambda **_: ([], []))
    monkeypatch.setattr(knn_pipeline, "_merge_sweep_outcomes", lambda cached, executed: [sweep_outcome])
    monkeypatch.setattr(
        knn_pipeline,
        "_merge_opinion_sweep_outcomes",
        lambda cached, executed: [opinion_outcome],
    )
    monkeypatch.setattr(
        knn_pipeline,
        "_select_best_configs",
        lambda outcomes, studies, allow_incomplete: {"tfidf": {study.key: study_selection}},
    )
    monkeypatch.setattr(
        knn_pipeline,
        "_select_best_opinion_configs",
        lambda outcomes, studies, allow_incomplete: {"tfidf": {study.key: opinion_selection}},
    )
    monkeypatch.setattr(
        knn_pipeline,
        "_load_final_metrics_from_disk",
        lambda **_: {"tfidf": {study.key: {"accuracy_overall": 0.9}}},
    )
    monkeypatch.setattr(knn_pipeline, "_load_loso_metrics_from_disk", lambda **_: {})
    monkeypatch.setattr(
        knn_pipeline,
        "_load_opinion_metrics",
        lambda out_dir, feature_space: {study.key: {"mae_after": 0.39}}
        if feature_space == "tfidf"
        else {},
    )
    monkeypatch.setattr(knn_pipeline, "_generate_reports", fake_generate_reports)

    out_dir = tmp_path / "out"
    knn_pipeline.main(
        [
            "--stage",
            "reports",
            "--dataset",
            str(tmp_path / "dataset"),
            "--out-dir",
            str(out_dir),
            "--word2vec-model-dir",
            str(tmp_path / "w2v"),
        ]
    )

    bundle = captured_bundle["bundle"]
    assert bundle.feature_spaces == ("tfidf", "word2vec", "sentence_transformer")
    assert "tfidf" in bundle.selections
    assert bundle.metrics_by_feature["tfidf"][study.key]["accuracy_overall"] == 0.9
    assert bundle.opinion_metrics["tfidf"][study.key]["mae_after"] == 0.39


def test_knn_pipeline_opinion_only(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """When only the opinion task is selected, the slate sweep helper must not run."""

    study = KnnStudySpec(key="study1", issue="gun_control", label="Study 1 – Gun Control")
    config = KnnSweepConfig(
        feature_space="tfidf",
        metric="cosine",
        text_fields=DEFAULT_EXTRA_TEXT_FIELDS,
    )
    call_log: Dict[str, int] = {"slate": 0, "opinion": 0}

    def fake_resolve(_tokens: Sequence[str]) -> List[KnnStudySpec]:
        return [study]

    def fake_build(_context) -> List[KnnSweepConfig]:
        return [config]

    def slate_stub(**_kwargs):
        call_log["slate"] += 1
        return [], []

    def opinion_stub(**_kwargs):
        call_log["opinion"] += 1
        return [], []

    monkeypatch.setattr(knn_pipeline, "_resolve_studies", fake_resolve)
    monkeypatch.setattr(knn_pipeline, "_build_sweep_configs", fake_build)
    monkeypatch.setattr(knn_pipeline, "_prepare_sweep_tasks", slate_stub)
    monkeypatch.setattr(knn_pipeline, "_prepare_opinion_sweep_tasks", opinion_stub)
    monkeypatch.setattr(knn_pipeline, "_emit_combined_sweep_plan", lambda **_kwargs: None)

    knn_pipeline.main(
        [
            "--stage",
            "plan",
            "--tasks",
            "opinion",
            "--dataset",
            str(tmp_path / "dataset"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert call_log["slate"] == 0
    assert call_log["opinion"] == 1


def test_knn_pipeline_finalize_emits_reports(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Stage=finalize should also refresh the Markdown reports."""

    study = KnnStudySpec(key="study1", issue="gun_control", label="Study 1 – Gun Control")
    config = KnnSweepConfig(
        feature_space="tfidf",
        metric="cosine",
        text_fields=DEFAULT_EXTRA_TEXT_FIELDS,
    )
    sweep_outcome = KnnSweepOutcome(
        order_index=0,
        study=study,
        feature_space="tfidf",
        config=config,
        accuracy=0.9,
        best_k=5,
        eligible=100,
        metrics_path=tmp_path / "sweep_metrics.json",
        metrics={"accuracy_overall": 0.9, "best_k": 5},
    )
    study_selection = KnnStudySelection(study=study, outcome=sweep_outcome)

    opinion_outcome = KnnOpinionSweepOutcome(
        order_index=0,
        study=study,
        config=config,
        feature_space="tfidf",
        mae=0.42,
        rmse=0.6,
        r2_score=0.12,
        baseline_mae=0.5,
        mae_delta=-0.08,
        best_k=7,
        participants=75,
        artifact=MetricsArtifact(
            path=tmp_path / "opinion_metrics.json",
            payload={"best_metrics": {"mae_after": 0.42}, "best_k": 7},
        ),
        accuracy_summary=AccuracySummary(eligible=75),
    )
    opinion_selection = KnnOpinionStudySelection(study=study, outcome=opinion_outcome)

    captured_bundle: Dict[str, ReportBundle] = {}

    monkeypatch.setattr(knn_pipeline, "_resolve_studies", lambda _tokens: [study])
    monkeypatch.setattr(knn_pipeline, "_build_sweep_configs", lambda _context: [config])
    monkeypatch.setattr(knn_pipeline, "_prepare_sweep_tasks", lambda **_: ([], []))
    monkeypatch.setattr(knn_pipeline, "_prepare_opinion_sweep_tasks", lambda **_: ([], []))
    monkeypatch.setattr(
        knn_pipeline,
        "_merge_sweep_outcomes",
        lambda cached, executed: [sweep_outcome],
    )
    monkeypatch.setattr(
        knn_pipeline,
        "_merge_opinion_sweep_outcomes",
        lambda cached, executed: [opinion_outcome],
    )
    monkeypatch.setattr(
        knn_pipeline,
        "_select_best_configs",
        lambda outcomes, studies, allow_incomplete: {"tfidf": {study.key: study_selection}},
    )
    monkeypatch.setattr(
        knn_pipeline,
        "_select_best_opinion_configs",
        lambda outcomes, studies, allow_incomplete: {"tfidf": {study.key: opinion_selection}},
    )
    monkeypatch.setattr(
        knn_pipeline,
        "_run_final_evaluations",
        lambda *, selections, studies, context: {"tfidf": {study.key: {"accuracy_overall": 0.88}}},
    )
    monkeypatch.setattr(
        knn_pipeline,
        "_run_cross_study_evaluations",
        lambda *, selections, studies, context: {},
    )
    monkeypatch.setattr(
        knn_pipeline,
        "_run_opinion_evaluations",
        lambda *, selections, studies, context: {
            "tfidf": {study.key: {"best_metrics": {"mae_after": 0.4}}}
        },
    )

    def fake_generate_reports(_root, bundle: ReportBundle) -> None:
        captured_bundle["bundle"] = bundle

    monkeypatch.setattr(knn_pipeline, "_generate_reports", fake_generate_reports)

    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    out_dir = tmp_path / "out"

    knn_pipeline.main(
        [
            "--stage",
            "finalize",
            "--dataset",
            str(dataset_path),
            "--out-dir",
            str(out_dir),
            "--feature-spaces",
            "tfidf",
        ]
    )

    bundle = captured_bundle["bundle"]
    assert bundle.metrics_by_feature["tfidf"][study.key]["accuracy_overall"] == 0.88
    assert bundle.opinion_metrics["tfidf"][study.key]["best_metrics"]["mae_after"] == 0.4


# ---------------------------------------------------------------------------
# XGBoost pipeline tests
# ---------------------------------------------------------------------------


def test_xgb_pipeline_finalize_writes_reports(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Finalize stage should emit reports without a separate pass."""

    study = XgbStudySpec(key="study1", issue="gun_control", label="Study 1 – Gun Control")
    config = XgbSweepConfig(
        text_vectorizer="tfidf",
        vectorizer_tag="tfidf",
        learning_rate=0.1,
        max_depth=4,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        vectorizer_cli=(),
    )
    sweep_outcome = XgbSweepOutcome(
        order_index=0,
        study=study,
        config=config,
        accuracy=0.83,
        coverage=0.6,
        evaluated=150,
        metrics_path=tmp_path / "xgb_sweep.json",
        metrics={"accuracy": 0.83},
    )
    study_selection = XgbStudySelection(study=study, outcome=sweep_outcome)
    opinion_outcome = XgbOpinionSweepOutcome(
        order_index=0,
        study=study,
        config=config,
        mae=0.41,
        rmse=0.65,
        r_squared=0.2,
        artifact=MetricsArtifact(
            path=tmp_path / "xgb_opinion.json",
            payload={"metrics": {"mae_after": 0.41}},
        ),
        accuracy_summary=AccuracySummary(),
    )
    opinion_selection = XgbOpinionStudySelection(study=study, outcome=opinion_outcome)

    captured_reports: Dict[str, Any] = {}

    monkeypatch.setattr(xgb_pipeline, "_resolve_study_specs", lambda **_: [study])
    monkeypatch.setattr(xgb_pipeline, "_build_sweep_configs", lambda _args: [config])
    monkeypatch.setattr(xgb_pipeline, "_prepare_sweep_tasks", lambda **_: ([], []))
    monkeypatch.setattr(xgb_pipeline, "_prepare_opinion_sweep_tasks", lambda **_: ([], []))
    monkeypatch.setattr(
        xgb_pipeline,
        "_merge_sweep_outcomes",
        lambda cached, executed: [sweep_outcome],
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_merge_opinion_sweep_outcomes",
        lambda cached, executed: [opinion_outcome],
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_select_best_configs",
        lambda outcomes: {study.key: study_selection},
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_select_best_opinion_configs",
        lambda outcomes: {study.key: opinion_selection},
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_run_final_evaluations",
        lambda *, selections, context: {study.key: {"accuracy": 0.9}},
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_run_opinion_stage",
        lambda *, selections, config: {study.key: {"mae_after": 0.34}},
    )

    def fake_write_reports(reports_dir, **kwargs) -> None:  # pylint: disable=unused-argument
        captured_reports["dir"] = reports_dir
        captured_reports["kwargs"] = kwargs

    monkeypatch.setattr(xgb_pipeline, "_write_reports", fake_write_reports)

    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    out_dir = tmp_path / "out"
    reports_dir = tmp_path / "reports"

    xgb_pipeline.main(
        [
            "--stage",
            "finalize",
            "--dataset",
            str(dataset_path),
            "--out-dir",
            str(out_dir),
            "--reports-dir",
            str(reports_dir),
        ]
    )

    assert captured_reports["dir"] == reports_dir
    sweeps_report = captured_reports["kwargs"]["sweeps"]
    assert sweeps_report.final_metrics[study.key]["accuracy"] == 0.9
    sections = captured_reports["kwargs"]["sections"]
    assert sections.opinion is not None
    assert sections.opinion.metrics[study.key]["mae_after"] == 0.34


def test_xgb_pipeline_sweeps_and_finalize(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Exercise the sweeps and finalize stages with cached artefacts."""

    study = XgbStudySpec(key="study1", issue="gun_control", label="Study 1 – Gun Control")
    config = XgbSweepConfig(
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
    call_log: Dict[str, Any] = {}

    def fake_resolve(**_kwargs) -> List[XgbStudySpec]:
        return [study]

    def fake_build(_args) -> List[XgbSweepConfig]:
        return [config]

    def fake_prepare_slate(
        *,
        studies: Sequence[XgbStudySpec],
        configs: Sequence[XgbSweepConfig],
        context: SweepRunContext,
        reuse_existing: bool,
    ) -> Tuple[List[XgbSweepTask], List[XgbSweepOutcome]]:
        task = _xgb_make_sweep_task(study=studies[0], config=configs[0], context=context)
        call_log.setdefault("sweep_tasks", []).append(task)
        return [task], []

    def fake_prepare_opinion(
        *,
        studies: Sequence[XgbStudySpec],
        configs: Sequence[XgbSweepConfig],
        context: OpinionStageConfig,
        reuse_existing: bool,
    ) -> Tuple[List[XgbOpinionSweepTask], List[XgbOpinionSweepOutcome]]:
        task = _xgb_make_opinion_task(study=studies[0], config=configs[0], context=context, index=0)
        call_log.setdefault("opinion_tasks", []).append(task)
        return [task], []

    monkeypatch.setattr(xgb_pipeline, "_resolve_study_specs", fake_resolve)
    monkeypatch.setattr(xgb_pipeline, "_build_sweep_configs", fake_build)
    monkeypatch.setattr(xgb_pipeline, "_prepare_sweep_tasks", fake_prepare_slate)
    monkeypatch.setattr(xgb_pipeline, "_prepare_opinion_sweep_tasks", fake_prepare_opinion)
    monkeypatch.setattr(xgb_pipeline, "prepare_sweep_execution", lambda **_: 0)
    monkeypatch.setattr(
        xgb_pipeline,
        "_execute_sweep_tasks",
        lambda tasks, jobs=1: pytest.fail(f"Unexpected sweep execution {tasks}"),
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_execute_opinion_sweep_tasks",
        lambda tasks, jobs=1: pytest.fail(f"Unexpected opinion execution {tasks}"),
    )

    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    out_dir = tmp_path / "out"

    # Sweeps stage: should skip because metrics already exist
    xgb_pipeline.main(
        [
            "--stage",
            "sweeps",
            "--dataset",
            str(dataset_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert call_log["sweep_tasks"][0].metrics_path.exists()
    assert call_log["opinion_tasks"][0].metrics_path.exists()

    # Finalize stage: reuse cached results and emit reports via stubs
    sweep_outcome = XgbSweepOutcome(
        order_index=0,
        study=study,
        config=config,
        accuracy=0.82,
        coverage=0.63,
        evaluated=140,
        metrics_path=tmp_path / "final_metrics.json",
        metrics={"accuracy": 0.82},
    )
    study_selection = XgbStudySelection(study=study, outcome=sweep_outcome)
    opinion_outcome = XgbOpinionSweepOutcome(
        order_index=0,
        study=study,
        config=config,
        mae=0.45,
        rmse=0.68,
        r_squared=0.22,
        artifact=MetricsArtifact(
            path=tmp_path / "opinion_metrics.json",
            payload={"metrics": {"mae_after": 0.45}},
        ),
        accuracy_summary=AccuracySummary(),
    )
    opinion_selection = XgbOpinionStudySelection(study=study, outcome=opinion_outcome)

    monkeypatch.setattr(
        xgb_pipeline,
        "_merge_sweep_outcomes",
        lambda cached, executed: [sweep_outcome],
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_merge_opinion_sweep_outcomes",
        lambda cached, executed: [opinion_outcome],
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_select_best_configs",
        lambda outcomes: {study.key: study_selection},
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_select_best_opinion_configs",
        lambda outcomes: {study.key: opinion_selection},
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_run_final_evaluations",
        lambda *, selections, context: {study.key: {"accuracy": 0.9}},
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_run_opinion_stage",
        lambda *, selections, config: {study.key: {"mae_after": 0.4}},
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_load_opinion_metrics_from_disk",
        lambda **_: {study.key: {"mae_after": 0.4}},
    )
    monkeypatch.setattr(
        xgb_pipeline,
        "_write_reports",
        lambda reports_dir, **_: reports_dir.mkdir(parents=True, exist_ok=True),
    )

    xgb_pipeline.main(
        [
            "--stage",
            "reports",
            "--dataset",
            str(dataset_path),
            "--out-dir",
            str(out_dir),
        ]
    )


def test_xgb_pipeline_opinion_only(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Opinion-only stage should avoid preparing slate tasks."""

    study = XgbStudySpec(key="study1", issue="gun_control", label="Study 1 – Gun Control")
    config = XgbSweepConfig(
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
    call_log = {"slate": 0, "opinion": 0}

    monkeypatch.setattr(xgb_pipeline, "_resolve_study_specs", lambda **_: [study])
    monkeypatch.setattr(xgb_pipeline, "_build_sweep_configs", lambda _args: [config])
    def forbid_slate(**_kwargs):
        call_log["slate"] += 1
        pytest.fail("Slate sweep tasks should not be prepared in opinion-only mode")

    def fake_prepare_opinion(**_kwargs):
        call_log["opinion"] += 1
        return [], []

    monkeypatch.setattr(xgb_pipeline, "_prepare_sweep_tasks", forbid_slate)
    monkeypatch.setattr(xgb_pipeline, "_prepare_opinion_sweep_tasks", fake_prepare_opinion)
    monkeypatch.setattr(xgb_pipeline, "_emit_combined_sweep_plan", lambda **_kwargs: None)

    xgb_pipeline.main(
        [
            "--stage",
            "plan",
            "--tasks",
            "opinion",
            "--dataset",
            str(tmp_path / "dataset"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert call_log["opinion"] == 1


# ---------------------------------------------------------------------------
# Utility to restore prepare_sweep_execution when needed
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_prepare_sweep_execution(monkeypatch: pytest.MonkeyPatch):
    """Ensure we don't leak a patched prepare_sweep_execution across tests."""

    monkeypatch.setattr(knn_pipeline, "prepare_sweep_execution", real_prepare_sweep_execution, raising=False)
    monkeypatch.setattr(xgb_pipeline, "prepare_sweep_execution", real_prepare_sweep_execution, raising=False)
    yield
