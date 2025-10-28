"""Unit tests for the refactored XGBoost pipeline helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import pytest

from common.prompt_docs import DEFAULT_EXTRA_TEXT_FIELDS
from common.opinion_sweep_types import AccuracySummary, MetricsArtifact

from xgb import pipeline as pipeline_module
from xgb import pipeline_cli as cli
from xgb import pipeline_evaluate as evaluate
from xgb import pipeline_reports as reports
from xgb import pipeline_sweeps as sweeps
from xgb.pipeline_context import (
    FinalEvalContext,
    OpinionStageConfig,
    OpinionStudySelection,
    OpinionSweepOutcome,
    OpinionSweepRunContext,
    StudySelection,
    StudySpec,
    SweepConfig,
    SweepOutcome,
    SweepRunContext,
    SweepTask,
)
from xgb.vectorizers import (
    SentenceTransformerVectorizerConfig,
    TfidfConfig,
    Word2VecVectorizerConfig,
)


def _make_study_spec() -> StudySpec:
    return StudySpec(key="study1", issue="gun_control", label="Study 1 – Gun Control (MTurk)")


def _make_sweep_config(tag: str = "tfidf") -> SweepConfig:
    return SweepConfig(
        text_vectorizer="tfidf",
        vectorizer_tag=tag,
        learning_rate=0.1,
        max_depth=4,
        n_estimators=200,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        vectorizer_cli=(),
    )


def test_split_tokens_trims_and_filters() -> None:
    assert cli._split_tokens(" alpha , beta ,, gamma ") == ["alpha", "beta", "gamma"]
    assert cli._split_tokens("") == []


def test_sanitize_token_replaces_special_characters() -> None:
    assert cli._sanitize_token("Model/Name 1.0") == "Model_Name_1p0"


def test_build_sweep_configs_supports_multiple_vectorisers() -> None:
    args, _ = cli._parse_args(
        [
            "--learning-rate-grid",
            "0.1",
            "--max-depth-grid",
            "4",
            "--n-estimators-grid",
            "200",
            "--subsample-grid",
            "0.9",
            "--colsample-grid",
            "0.8",
            "--reg-lambda-grid",
            "1.0",
            "--reg-alpha-grid",
            "0.0",
            "--text-vectorizer-grid",
            "tfidf,word2vec,sentence_transformer",
            "--word2vec-size",
            "128",
            "--word2vec-window",
            "4",
            "--word2vec_min_count",
            "2",
            "--word2vec-epochs",
            "5",
            "--word2vec-workers",
            "2",
            "--sentence-transformer-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--sentence-transformer-batch-size",
            "16",
        ]
    )
    configs = cli._build_sweep_configs(args)
    assert len(configs) == 3

    vectorizers = {config.text_vectorizer: config for config in configs}
    assert vectorizers["tfidf"].vectorizer_cli == ()
    assert vectorizers["word2vec"].vectorizer_tag == "w2v128"
    assert "--word2vec_size" in vectorizers["word2vec"].vectorizer_cli

    sentence_config = vectorizers["sentence_transformer"]
    assert sentence_config.vectorizer_tag.startswith("st_")
    assert "--sentence_transformer_model" in sentence_config.vectorizer_cli
    assert "--sentence_transformer_normalize" in sentence_config.vectorizer_cli


def test_parse_args_accepts_tasks() -> None:
    args, extra = cli._parse_args(["--tasks", "next_video"])
    assert args.tasks == "next_video"
    assert extra == []


def test_resolve_study_specs_filters_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_dataset_source", lambda dataset, cache_dir: {"dataset": dataset})
    monkeypatch.setattr(
        cli,
        "issues_in_dataset",
        lambda _dataset: ["gun_control", "minimum_wage", "climate_change"],
    )

    specs = cli._resolve_study_specs(
        dataset="stub",
        cache_dir="/tmp/cache",
        requested_issues=["gun_control"],
        requested_studies=[],
        allow_incomplete=False,
    )
    assert [spec.issue for spec in specs] == ["gun_control"]

    with pytest.raises(ValueError):
        cli._resolve_study_specs(
            dataset="stub",
            cache_dir="/tmp/cache",
            requested_issues=["unknown_issue"],
            requested_studies=[],
            allow_incomplete=False,
        )

    specs = cli._resolve_study_specs(
        dataset="stub",
        cache_dir="/tmp/cache",
        requested_issues=[],
        requested_studies=["study1"],
        allow_incomplete=True,
    )
    assert [spec.key for spec in specs] == ["study1"]


def test_run_sweeps_collects_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: List[Path] = []

    def fake_run_xgb_cli(args: List[str]) -> None:
        out_dir = Path(args[args.index("--out_dir") + 1])
        issue_name = args[args.index("--issues") + 1]
        study_name = args[args.index("--participant_studies") + 1]
        evaluation_slug = f"{issue_name.replace(' ', '_')}_{study_name.replace(' ', '_')}"
        metrics_path = out_dir / evaluation_slug / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"accuracy": 0.81, "coverage": 0.56, "evaluated": 321}),
            encoding="utf-8",
        )
        recorded.append(metrics_path)

    monkeypatch.setattr(sweeps, "_run_xgb_cli", fake_run_xgb_cli)

    study = _make_study_spec()
    config = _make_sweep_config()
    context = SweepRunContext(
        base_cli=["--dataset", "stub"],
        extra_cli=["--eval_max", "100"],
        sweep_dir=tmp_path,
        tree_method="hist",
        jobs=1,
    )

    outcomes = sweeps._run_sweeps(studies=[study], configs=[config], context=context)
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.metrics_path in recorded
    assert outcome.accuracy == pytest.approx(0.81)
    assert outcome.metrics["evaluated"] == 321


def test_sweeps_stage_skips_cached_indices_without_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="xgb.pipeline")
    study = _make_study_spec()
    config = _make_sweep_config()

    cached_metrics_path = tmp_path / "cached" / "metrics.json"
    cached_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cached_metrics_path.write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")
    cached_outcome = SweepOutcome(
        order_index=0,
        study=study,
        config=config,
        accuracy=0.9,
        coverage=0.5,
        evaluated=123,
        metrics_path=cached_metrics_path,
        metrics={"accuracy": 0.9},
    )

    run_root = tmp_path / "pending"
    task = SweepTask(
        index=1,
        study=study,
        config=config,
        base_cli=("--dataset", "stub"),
        extra_cli=(),
        run_root=run_root,
        tree_method="hist",
        metrics_path=run_root / "eval" / "metrics.json",
        train_participant_studies=(),
    )

    monkeypatch.setattr(pipeline_module, "_resolve_study_specs", lambda **_: [study])
    monkeypatch.setattr(pipeline_module, "_build_sweep_configs", lambda args: [config])
    monkeypatch.setattr(
        pipeline_module,
        "_prepare_sweep_tasks",
        lambda **_: ([task], [cached_outcome]),
    )
    monkeypatch.setattr(pipeline_module, "_prepare_opinion_sweep_tasks", lambda **_: ([], []))
    monkeypatch.setattr(pipeline_module, "_execute_opinion_sweep_tasks", lambda *_, **__: [])

    execute_calls: List[List[SweepTask]] = []

    def fake_execute(tasks: List[SweepTask], *, jobs: int) -> List[SweepOutcome]:
        execute_calls.append(list(tasks))
        assert jobs == 1
        executed_task = tasks[0]
        executed_task.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        executed_task.metrics_path.write_text(json.dumps({"accuracy": 0.7}), encoding="utf-8")
        return [
            SweepOutcome(
                order_index=executed_task.index,
                study=executed_task.study,
                config=executed_task.config,
                accuracy=0.7,
                coverage=0.4,
                evaluated=111,
                metrics_path=executed_task.metrics_path,
                metrics={"accuracy": 0.7},
            )
        ]

    monkeypatch.setattr(pipeline_module, "_execute_sweep_tasks", fake_execute)

    pipeline_module.main(
        [
            "--stage",
            "sweeps",
            "--sweep-task-id",
            "0",
            "--sweep-task-count",
            "2",
            "--tasks",
            "next_video",
            "--dataset",
            str(tmp_path / "dataset"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out-dir",
            str(tmp_path / "out"),
            "--sweep-dir",
            str(tmp_path / "sweeps"),
        ]
    )

    assert execute_calls == []
    assert "Skipping sweep task 0" in caplog.text
    assert "Sweep task count mismatch" not in caplog.text


def test_sweeps_stage_executes_pending_task_with_cached_offsets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="xgb.pipeline")
    study = _make_study_spec()
    config = _make_sweep_config()

    cached_metrics_path = tmp_path / "cached" / "metrics.json"
    cached_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cached_metrics_path.write_text(json.dumps({"accuracy": 0.8}), encoding="utf-8")
    cached_outcome = SweepOutcome(
        order_index=0,
        study=study,
        config=config,
        accuracy=0.8,
        coverage=0.55,
        evaluated=100,
        metrics_path=cached_metrics_path,
        metrics={"accuracy": 0.8},
    )

    run_root = tmp_path / "pending"
    task = SweepTask(
        index=1,
        study=study,
        config=config,
        base_cli=("--dataset", "stub"),
        extra_cli=(),
        run_root=run_root,
        tree_method="hist",
        metrics_path=run_root / "eval" / "metrics.json",
        train_participant_studies=(),
    )

    monkeypatch.setattr(pipeline_module, "_resolve_study_specs", lambda **_: [study])
    monkeypatch.setattr(pipeline_module, "_build_sweep_configs", lambda args: [config])
    monkeypatch.setattr(
        pipeline_module,
        "_prepare_sweep_tasks",
        lambda **_: ([task], [cached_outcome]),
    )
    monkeypatch.setattr(pipeline_module, "_prepare_opinion_sweep_tasks", lambda **_: ([], []))
    monkeypatch.setattr(pipeline_module, "_execute_opinion_sweep_tasks", lambda *_, **__: [])

    execute_calls: List[List[SweepTask]] = []

    def fake_execute(tasks: List[SweepTask], *, jobs: int) -> List[SweepOutcome]:
        execute_calls.append(list(tasks))
        executed_task = tasks[0]
        executed_task.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        executed_task.metrics_path.write_text(json.dumps({"accuracy": 0.75}), encoding="utf-8")
        return [
            SweepOutcome(
                order_index=executed_task.index,
                study=executed_task.study,
                config=executed_task.config,
                accuracy=0.75,
                coverage=0.5,
                evaluated=99,
                metrics_path=executed_task.metrics_path,
                metrics={"accuracy": 0.75},
            )
        ]

    monkeypatch.setattr(pipeline_module, "_execute_sweep_tasks", fake_execute)

    pipeline_module.main(
        [
            "--stage",
            "sweeps",
            "--sweep-task-id",
            "1",
            "--sweep-task-count",
            "2",
            "--tasks",
            "next_video",
            "--dataset",
            str(tmp_path / "dataset"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out-dir",
            str(tmp_path / "out"),
            "--sweep-dir",
            str(tmp_path / "sweeps"),
        ]
    )

    assert len(execute_calls) == 1
    assert execute_calls[0][0].index == 1
    assert "Completed sweep task 1" in caplog.text


def test_select_best_configs_prefers_accuracy_then_coverage_then_support(tmp_path: Path) -> None:
    study = _make_study_spec()

    def make_outcome(order_index: int, accuracy: float, coverage: float, evaluated: int, tag: str) -> SweepOutcome:
        config = _make_sweep_config(tag)
        return SweepOutcome(
            order_index=order_index,
            study=study,
            config=config,
            accuracy=accuracy,
            coverage=coverage,
            evaluated=evaluated,
            metrics_path=tmp_path / f"{tag}.json",
            metrics={"accuracy": accuracy, "coverage": coverage, "evaluated": evaluated},
        )

    outcomes = [
        make_outcome(0, 0.80, 0.60, 200, "a"),
        make_outcome(1, 0.82, 0.55, 150, "b"),
        make_outcome(2, 0.82, 0.60, 100, "c"),
        make_outcome(3, 0.82, 0.60, 250, "d"),
    ]
    selection = sweeps._select_best_configs(outcomes)
    assert selection[study.key].outcome.config.vectorizer_tag == "d"


def test_select_best_opinion_configs_prefers_mae_then_rmse_then_r2(tmp_path: Path) -> None:
    study = _make_study_spec()

    def make_outcome(
        order_index: int,
        mae: float,
        rmse: float,
        r_squared: float,
        tag: str,
    ) -> OpinionSweepOutcome:
        config = _make_sweep_config(tag)
        return OpinionSweepOutcome(
            order_index=order_index,
            study=study,
            config=config,
            mae=mae,
            rmse=rmse,
            r_squared=r_squared,
            artifact=MetricsArtifact(
                path=tmp_path / f"{tag}.json",
                payload={
                    "metrics": {
                        "mae_after": mae,
                        "rmse_after": rmse,
                        "r2_after": r_squared,
                    }
                },
            ),
            accuracy_summary=AccuracySummary(),
        )

    outcomes = [
        make_outcome(0, 0.40, 0.60, 0.20, "a"),
        make_outcome(1, 0.38, 0.65, 0.18, "b"),
        make_outcome(2, 0.38, 0.61, 0.22, "c"),
        make_outcome(3, 0.38, 0.61, 0.25, "d"),
    ]
    selection = sweeps._select_best_opinion_configs(outcomes)
    assert selection[study.key].outcome.config.vectorizer_tag == "d"


def test_run_final_evaluations_reads_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    train_study = StudySpec(
        key="study2",
        issue="minimum_wage",
        label="Study 2 – Minimum Wage (YouGov)",
    )

    def fake_run_xgb_cli(args: List[str]) -> None:
        out_dir = Path(args[args.index("--out_dir") + 1])
        issue_name = args[args.index("--issues") + 1]
        study_name = args[args.index("--participant_studies") + 1]
        train_arg = args[args.index("--train_participant_studies") + 1]
        assert train_arg == train_study.key
        evaluation_slug = f"{issue_name.replace(' ', '_')}_{study_name.replace(' ', '_')}"
        metrics_path = out_dir / evaluation_slug / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"accuracy": 0.77, "coverage": 0.61, "evaluated": 128}),
            encoding="utf-8",
        )

    monkeypatch.setattr(evaluate, "_run_xgb_cli", fake_run_xgb_cli)

    study = _make_study_spec()
    outcome = SweepOutcome(
        order_index=0,
        study=study,
        config=_make_sweep_config(),
        accuracy=0.8,
        coverage=0.6,
        evaluated=200,
        metrics_path=tmp_path / "metrics.json",
        metrics={"accuracy": 0.8},
    )
    selections = {study.key: StudySelection(study=study, outcome=outcome)}
    context = FinalEvalContext(
        base_cli=["--dataset", "stub"],
        extra_cli=["--seed", "13"],
        out_dir=tmp_path,
        tree_method="hist",
        save_model_dir=None,
        reuse_existing=False,
    )

    metrics = evaluate._run_final_evaluations(
        selections=selections,
        studies=[study, train_study],
        context=context,
    )
    assert metrics[study.key]["accuracy"] == 0.77
    assert metrics[study.key]["evaluated"] == 128


def test_run_final_evaluations_uses_save_model_and_sets_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    study = _make_study_spec()
    config = _make_sweep_config()
    outcome = SweepOutcome(
        order_index=0,
        study=study,
        config=config,
        accuracy=0.8,
        coverage=0.6,
        evaluated=200,
        metrics_path=tmp_path / "metrics.json",
        metrics={"accuracy": 0.8},
    )
    selections = {study.key: StudySelection(study=study, outcome=outcome)}
    save_model_dir = tmp_path / "models"
    train_study = StudySpec(
        key="study2",
        issue="minimum_wage",
        label="Study 2 – Minimum Wage (YouGov)",
    )
    context = FinalEvalContext(
        base_cli=["--dataset", "stub"],
        extra_cli=["--extra", "flag"],
        out_dir=tmp_path / "out",
        tree_method="hist",
        save_model_dir=save_model_dir,
        reuse_existing=False,
    )

    call_details: dict[str, object] = {}

    def fake_run(args: List[str]) -> None:
        call_details["args"] = list(args)
        out_dir_index = args.index("--out_dir") + 1
        out_dir = Path(args[out_dir_index])
        evaluation_slug = study.evaluation_slug
        metrics_path = out_dir / evaluation_slug / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({"accuracy": 0.83}), encoding="utf-8")
        if "--save_model" in args:
            model_dir = Path(args[args.index("--save_model") + 1])
            model_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(evaluate, "_run_xgb_cli", fake_run)

    metrics = evaluate._run_final_evaluations(
        selections=selections,
        studies=[study, train_study],
        context=context,
    )

    assert metrics == {
        study.key: {
            "accuracy": 0.83,
            "issue": study.issue,
            "issue_label": "Gun Control",
            "study": study.key,
            "study_label": study.label,
        }
    }
    assert save_model_dir.exists()
    assert "--save_model" in call_details["args"]
    assert str(save_model_dir) in call_details["args"]
    assert "--train_participant_studies" in call_details["args"]


def test_run_opinion_stage_invokes_matching_studies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: Dict[str, object] = {}

    def fake_run_opinion_eval(*, request, studies):
        captured["studies"] = list(studies)
        captured["learning_rate"] = request.train_config.booster.learning_rate
        return {studies[0]: {"label": "Study One"}}

    monkeypatch.setattr(evaluate, "run_opinion_eval", fake_run_opinion_eval)

    study = _make_study_spec()
    config = _make_sweep_config()
    outcome = OpinionSweepOutcome(
        order_index=0,
        study=study,
        config=config,
        mae=0.5,
        rmse=0.7,
        r_squared=0.2,
        artifact=MetricsArtifact(
            path=tmp_path / "opinion.json",
            payload={"metrics": {"mae_after": 0.5, "rmse_after": 0.7, "r2_after": 0.2}},
        ),
        accuracy_summary=AccuracySummary(),
    )
    selections = {study.key: OpinionStudySelection(study=study, outcome=outcome)}
    stage_config = OpinionStageConfig(
        dataset="dataset",
        cache_dir="cache",
        base_out_dir=tmp_path / "opinions",
        extra_fields=DEFAULT_EXTRA_TEXT_FIELDS,
        studies=("study1",),
        max_participants=50,
        seed=999,
        max_features=10,
        tree_method="hist",
        overwrite=True,
        tfidf_config=TfidfConfig(max_features=10),
        word2vec_config=Word2VecVectorizerConfig(),
        sentence_transformer_config=SentenceTransformerVectorizerConfig(),
        word2vec_model_base=None,
        reuse_existing=False,
    )

    results = evaluate._run_opinion_stage(selections=selections, config=stage_config)
    assert captured["studies"] == ["study1"]
    assert captured["learning_rate"] == pytest.approx(0.1)
    assert "study1" in results


def test_run_opinion_from_next_stage_uses_slate_booster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    study = _make_study_spec()
    selection = StudySelection(
        study=study,
        outcome=SweepOutcome(
            order_index=0,
            study=study,
            config=_make_sweep_config(),
            accuracy=0.72,
            coverage=0.63,
            evaluated=512,
            metrics_path=tmp_path / "metrics.json",
            metrics={"accuracy": 0.72},
        ),
    )
    selections = {study.key: selection}

    stage_config = OpinionStageConfig(
        dataset="dataset",
        cache_dir="cache",
        base_out_dir=tmp_path / "opinions",
        extra_fields=DEFAULT_EXTRA_TEXT_FIELDS,
        studies=(study.key,),
        max_participants=100,
        seed=42,
        max_features=None,
        tree_method="hist",
        overwrite=True,
        tfidf_config=TfidfConfig(max_features=None),
        word2vec_config=Word2VecVectorizerConfig(),
        sentence_transformer_config=SentenceTransformerVectorizerConfig(),
        word2vec_model_base=None,
        reuse_existing=False,
    )

    captured: Dict[str, object] = {}

    def fake_run_opinion_eval(*, request, studies):
        captured["studies"] = list(studies)
        captured["learning_rate"] = request.train_config.booster.learning_rate
        metrics_path = (
            stage_config.base_out_dir
            / "from_next"
            / "tfidf"
            / studies[0]
            / f"opinion_xgb_{studies[0]}_validation_metrics.json"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({"metrics": {}}), encoding="utf-8")
        return {study.key: {"label": study.label}}

    monkeypatch.setattr(evaluate, "run_opinion_eval", fake_run_opinion_eval)

    results = evaluate._run_opinion_from_next_stage(
        selections=selections,
        studies=[study],
        config=stage_config,
    )

    assert captured["studies"] == [study.key]
    assert captured["learning_rate"] == pytest.approx(0.1)
    assert study.key in results


def test_run_opinion_stage_reuses_metrics_and_warns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    study = _make_study_spec()
    config = _make_sweep_config()
    selection = OpinionStudySelection(
        study=study,
        outcome=OpinionSweepOutcome(
            order_index=0,
            study=study,
            config=config,
            mae=0.5,
            rmse=0.7,
            r_squared=0.2,
            artifact=MetricsArtifact(path=tmp_path / "cached.json", payload={}),
            accuracy_summary=AccuracySummary(),
        ),
    )
    selections = {study.key: selection}

    reuse_dir = tmp_path / "out" / "tfidf" / study.key
    reuse_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reuse_dir / f"opinion_xgb_{study.key}_validation_metrics.json"
    cached_metrics = {"mae_after": 0.41}
    metrics_path.write_text(json.dumps(cached_metrics), encoding="utf-8")

    call_counter = {"count": 0}

    def fake_run_opinion_eval(*args, **kwargs):
        call_counter["count"] += 1
        return {}

    monkeypatch.setattr(evaluate, "run_opinion_eval", fake_run_opinion_eval)

    config_stage = OpinionStageConfig(
        dataset="dataset",
        cache_dir="cache",
        base_out_dir=tmp_path / "out",
        extra_fields=DEFAULT_EXTRA_TEXT_FIELDS,
        studies=(study.key, "missing_study"),
        max_participants=0,
        seed=0,
        max_features=None,
        tree_method="hist",
        overwrite=False,
        tfidf_config=TfidfConfig(max_features=None),
        word2vec_config=Word2VecVectorizerConfig(),
        sentence_transformer_config=SentenceTransformerVectorizerConfig(),
        word2vec_model_base=None,
        reuse_existing=True,
    )

    with caplog.at_level("WARNING", logger="xgb.pipeline.finalize"):
        results = evaluate._run_opinion_stage(selections=selections, config=config_stage)

    assert results == {study.key: cached_metrics}
    assert call_counter["count"] == 0
    assert any("Skipping opinion study for study=missing_study" in message for message in caplog.messages)


def test_run_cross_study_evaluations_executes_loso(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    study_a = _make_study_spec()
    study_b = StudySpec(key="study2", issue="gun_control", label="Study 2 – Gun Control")
    config = _make_sweep_config()
    selections = {
        study_a.key: StudySelection(
            study=study_a,
            outcome=SweepOutcome(
                order_index=0,
                study=study_a,
                config=config,
                accuracy=0.8,
                coverage=0.6,
                evaluated=200,
                metrics_path=tmp_path / "a.json",
                metrics={},
            ),
        ),
        study_b.key: StudySelection(
            study=study_b,
            outcome=SweepOutcome(
                order_index=1,
                study=study_b,
                config=config,
                accuracy=0.75,
                coverage=0.55,
                evaluated=180,
                metrics_path=tmp_path / "b.json",
                metrics={},
            ),
        ),
    }

    context = FinalEvalContext(
        base_cli=["--dataset", "stub"],
        extra_cli=[],
        out_dir=tmp_path,
        tree_method="hist",
        save_model_dir=None,
        reuse_existing=False,
    )

    call_args: list[list[str]] = []

    def fake_run(args: List[str]) -> None:
        call_args.append(list(args))
        out_dir = Path(args[args.index("--out_dir") + 1])
        issue = args[args.index("--issues") + 1]
        study_key = args[args.index("--participant_studies") + 1]
        spec = study_a if study_key == study_a.key else study_b
        metrics_path = out_dir / spec.evaluation_slug / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({"accuracy": 0.65}), encoding="utf-8")

    monkeypatch.setattr(evaluate, "_run_xgb_cli", fake_run)

    results = evaluate._run_cross_study_evaluations(
        selections=selections,
        studies=[study_a, study_b],
        context=context,
    )

    assert set(results.keys()) == {study_a.key, study_b.key}
    assert all(payload["accuracy"] == 0.65 for payload in results.values())
    assert any("--train_participant_studies" in args for args in call_args)


def test_prepare_opinion_sweep_tasks_reuses_cached_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    study = _make_study_spec()
    config = _make_sweep_config()
    context = OpinionSweepRunContext(
        dataset="dataset",
        cache_dir="cache",
        sweep_dir=tmp_path,
        extra_fields=DEFAULT_EXTRA_TEXT_FIELDS,
        max_participants=25,
        seed=123,
        max_features=None,
        tree_method="hist",
        overwrite=False,
        tfidf_config=TfidfConfig(max_features=None),
        word2vec_config=Word2VecVectorizerConfig(),
        sentence_transformer_config=SentenceTransformerVectorizerConfig(),
        word2vec_model_base=None,
    )

    run_root = context.sweep_dir / study.issue_slug / study.study_slug / config.label()
    metrics_path = (
        run_root
        / sweeps.DEFAULT_OPINION_FEATURE_SPACE
        / study.key
        / f"opinion_xgb_{study.key}_validation_metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        sweeps,
        "_load_metrics",
        lambda path: {"metrics": {"mae_after": 0.4, "rmse_after": 0.6, "r2_after": 0.2}},
    )

    pending, cached = sweeps._prepare_opinion_sweep_tasks(
        studies=[study],
        configs=[config],
        context=context,
        reuse_existing=True,
    )

    assert pending == []
    assert len(cached) == 1
    cached_outcome = cached[0]
    assert isinstance(cached_outcome, OpinionSweepOutcome)
    assert cached_outcome.metrics_path == metrics_path
    assert cached_outcome.mae == pytest.approx(0.4)


def test_iter_opinion_sweep_tasks_respects_feature_space(tmp_path: Path) -> None:
    study = _make_study_spec()
    config = SweepConfig(
        text_vectorizer="word2vec",
        vectorizer_tag="w2v256",
        learning_rate=0.1,
        max_depth=4,
        n_estimators=200,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        vectorizer_cli=(),
    )
    context = OpinionSweepRunContext(
        dataset="dataset",
        cache_dir="cache",
        sweep_dir=tmp_path,
        extra_fields=DEFAULT_EXTRA_TEXT_FIELDS,
        max_participants=25,
        seed=42,
        max_features=None,
        tree_method="hist",
        overwrite=False,
        tfidf_config=TfidfConfig(max_features=None),
        word2vec_config=Word2VecVectorizerConfig(),
        sentence_transformer_config=SentenceTransformerVectorizerConfig(),
        word2vec_model_base=None,
    )

    tasks = list(
        sweeps._iter_opinion_sweep_tasks(
            studies=[study],
            configs=[config],
            context=context,
        )
    )
    assert len(tasks) == 1

    task = tasks[0]
    assert task.feature_space == "word2vec"
    assert "word2vec" in task.metrics_path.parts

    request_args = task.request_args
    vectorizer = request_args["vectorizer"]
    train_config = request_args["train_config"]
    assert vectorizer.feature_space == "word2vec"
    assert vectorizer.word2vec is not None
    assert train_config.max_features is None


def test_prepare_sweep_tasks_reuses_cached_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    study = _make_study_spec()
    config = _make_sweep_config()
    context = SweepRunContext(
        base_cli=["--dataset", "stub"],
        extra_cli=["--seed", "1"],
        sweep_dir=tmp_path,
        tree_method="hist",
        jobs=1,
    )

    run_root = context.sweep_dir / study.issue_slug / study.study_slug / config.label()
    metrics_path = run_root / study.evaluation_slug / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}", encoding="utf-8")

    stub_metrics = {"accuracy": 0.91, "coverage": 0.63, "evaluated": 512}
    call_counter = {"count": 0}

    def fake_load(path: Path) -> dict:
        call_counter["count"] += 1
        assert path == metrics_path
        return stub_metrics

    monkeypatch.setattr(sweeps, "_load_metrics", fake_load)

    pending, cached = sweeps._prepare_sweep_tasks(
        studies=[study],
        configs=[config],
        context=context,
        reuse_existing=True,
    )

    assert pending == []
    assert len(cached) == 1
    outcome = cached[0]
    assert outcome.metrics_path == metrics_path
    assert outcome.accuracy == pytest.approx(0.91)
    assert outcome.metrics["evaluated"] == 512
    assert call_counter["count"] == 1


def test_gpu_tree_method_supported_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    # Case 1: xgboost unavailable
    monkeypatch.setattr(sweeps, "xgboost", None)
    assert sweeps._gpu_tree_method_supported() is False

    # Case 2: modern build with _has_cuda_support returning truthy
    class ModernCore:
        def __init__(self) -> None:
            self._calls: int = 0

        def _has_cuda_support(self) -> bool:
            self._calls += 1
            return True

    modern = type("ModernXGB", (), {"core": ModernCore()})()
    monkeypatch.setattr(sweeps, "xgboost", modern)
    assert sweeps._gpu_tree_method_supported() is True
    assert modern.core._calls == 1  # type: ignore[attr-defined]

    # Case 3: _has_cuda_support raises -> expect False
    class FaultyCore:
        def _has_cuda_support(self) -> bool:
            raise RuntimeError("boom")

    faulty = type("FaultyXGB", (), {"core": FaultyCore()})()
    monkeypatch.setattr(sweeps, "xgboost", faulty)
    assert sweeps._gpu_tree_method_supported() is False

    # Case 4: legacy build exposing _LIB symbol lookup
    class LegacyLib:
        def __init__(self, has_symbol: bool) -> None:
            self._has_symbol = has_symbol

        def __getattr__(self, item: str) -> object:
            if item == "XGBoosterPredictFromDeviceDMatrix" and self._has_symbol:
                return object()
            raise AttributeError(item)

    legacy_with_symbol = type("LegacyXGB", (), {"core": type("Core", (), {"_LIB": LegacyLib(True)})()})()
    monkeypatch.setattr(sweeps, "xgboost", legacy_with_symbol)
    assert sweeps._gpu_tree_method_supported() is True

    legacy_without_symbol = type("LegacyXGBNoSymbol", (), {"core": type("Core", (), {"_LIB": LegacyLib(False)})()})()
    monkeypatch.setattr(sweeps, "xgboost", legacy_without_symbol)
    assert sweeps._gpu_tree_method_supported() is False


def test_format_float_three_decimal_precision() -> None:
    assert reports._format_float(0.12349) == "0.123"
    assert reports._format_float(0.12354) == "0.124"


def test_write_reports_generates_expected_readmes(tmp_path: Path, sample_png: Path) -> None:
    study = _make_study_spec()
    outcome = SweepOutcome(
        order_index=0,
        study=study,
        config=_make_sweep_config(),
        accuracy=0.82,
        coverage=0.64,
        evaluated=256,
        metrics_path=tmp_path / "metrics.json",
        metrics={"accuracy": 0.82, "coverage": 0.64, "evaluated": 256},
    )
    selections = {study.key: StudySelection(study=study, outcome=outcome)}
    final_metrics = {
        study.key: {
            "accuracy": 0.81,
            "coverage": 0.62,
            "evaluated": 200,
            "issue_label": "Gun Control",
            "study_label": study.label,
            "dataset": "sim_dataset",
        }
    }
    opinion_metrics = {
        "study1": {
            "label": "Study One",
            "n_participants": 123,
            "metrics": {"mae_after": 1.2, "rmse_after": 1.8, "r2_after": 0.4},
            "baseline": {"mae_before": 1.5},
            "dataset": "sim_dataset",
            "split": "validation",
        }
    }
    opinion_from_next_metrics = {
        "study1": {
            "label": "Study One",
            "n_participants": 98,
            "metrics": {"mae_after": 1.1, "rmse_after": 1.6, "r2_after": 0.45},
            "baseline": {"mae_before": 1.45},
            "dataset": "sim_dataset",
            "split": "validation",
        }
    }

    loso_metrics = {
        study.key: {
            "accuracy": 0.78,
            "coverage": 0.60,
            "evaluated": 150,
            "correct": 117,
            "known_candidate_hits": 80,
            "known_candidate_total": 110,
            "avg_probability": 0.52,
            "issue": study.issue,
            "issue_label": "Gun Control",
            "study": study.key,
            "study_label": study.label,
        }
    }

    plots_dir = tmp_path / "tfidf" / "opinion"
    plots_dir.mkdir(parents=True, exist_ok=True)
    png_bytes = sample_png.read_bytes()
    for stem in ("mae_study1", "r2_study1", "change_heatmap_study1"):
        (plots_dir / f"{stem}.png").write_bytes(png_bytes)

    sweep_report = reports.SweepReportData(
        outcomes=[outcome],
        selections=selections,
        final_metrics=final_metrics,
        loso_metrics=loso_metrics,
    )
    opinion_report = reports.OpinionReportData(metrics=opinion_metrics)
    opinion_next_report = reports.OpinionReportData(
        metrics=opinion_from_next_metrics,
        title="XGBoost Opinion Regression (Next-Video Config)",
        description_lines=[
            "This section reuses the selected next-video configuration to "
            "estimate post-study opinion change."
        ],
    )
    reports._write_reports(
        reports_dir=tmp_path,
        sweeps=sweep_report,
        allow_incomplete=False,
        sections=reports.ReportSections(
            include_next_video=True,
            opinion=opinion_report,
            opinion_from_next=opinion_next_report,
        ),
    )

    catalog = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "XGBoost Report Catalog" in catalog

    hyper = (tmp_path / "hyperparameter_tuning" / "README.md").read_text(encoding="utf-8")
    assert "| Config | Accuracy" in hyper
    assert "**tfidf_lr0p1_depth4_estim200_sub0p9_col0p8_l21_l10**" in hyper

    next_video = (tmp_path / "next_video" / "README.md").read_text(encoding="utf-8")
    assert "XGBoost Next-Video Baseline" in next_video
    assert "Study 1 – Gun Control (MTurk)" in next_video
    assert "| Study | Issue | Acc (eligible) ↑ | Baseline ↑ | Random ↑ |" in next_video
    assert "Cross-Study Holdouts" in next_video

    opinion = (tmp_path / "opinion" / "README.md").read_text(encoding="utf-8")
    assert "Study One" in opinion
    assert "![Mae Study1]" in opinion
    opinion_next = (tmp_path / "opinion_from_next" / "README.md").read_text(encoding="utf-8")
    assert "Next-Video Config" in opinion_next
    assert "reuses the selected next-video configuration" in opinion_next


def test_load_final_metrics_from_disk_adds_defaults(tmp_path: Path) -> None:
    study = _make_study_spec()
    next_video_dir = tmp_path / "final"
    metrics_path = next_video_dir / study.evaluation_slug / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"accuracy": 0.71}), encoding="utf-8")

    result = sweeps._load_final_metrics_from_disk(
        next_video_dir=next_video_dir,
        studies=[study],
    )

    assert study.key in result
    metrics = result[study.key]
    assert metrics["accuracy"] == 0.71
    assert metrics["issue"] == study.issue
    assert metrics["issue_label"] == "Gun Control"
    assert metrics["study"] == study.key
    assert metrics["study_label"] == study.label


def test_load_opinion_metrics_from_disk_supports_feature_space_layout(tmp_path: Path) -> None:
    study = _make_study_spec()
    feature_dir = tmp_path / sweeps.DEFAULT_OPINION_FEATURE_SPACE / study.key
    feature_dir.mkdir(parents=True, exist_ok=True)
    preferred_payload = {"metrics": {"mae_after": 0.42}}
    metrics_path = (
        feature_dir / f"opinion_xgb_{study.key}_validation_metrics.json"
    )
    metrics_path.write_text(json.dumps(preferred_payload), encoding="utf-8")

    legacy_dir = tmp_path / study.key
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = legacy_dir / f"opinion_xgb_{study.key}_validation_metrics.json"
    legacy_path.write_text(json.dumps({"metrics": {"mae_after": 0.99}}), encoding="utf-8")

    result = sweeps._load_opinion_metrics_from_disk(
        opinion_dir=tmp_path,
        studies=[study],
    )

    assert study.key in result
    assert result[study.key]["metrics"]["mae_after"] == pytest.approx(0.42)


def test_load_opinion_metrics_from_disk_falls_back_to_legacy(tmp_path: Path) -> None:
    study = _make_study_spec()
    legacy_dir = tmp_path / study.key
    legacy_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": {"mae_after": 0.37}}
    metrics_path = legacy_dir / f"opinion_xgb_{study.key}_validation_metrics.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    result = sweeps._load_opinion_metrics_from_disk(
        opinion_dir=tmp_path,
        studies=[study],
    )

    assert study.key in result
    assert result[study.key]["metrics"]["mae_after"] == pytest.approx(0.37)


def test_load_opinion_from_next_metrics_from_disk(tmp_path: Path) -> None:
    study = _make_study_spec()
    base_dir = tmp_path / "from_next" / sweeps.DEFAULT_OPINION_FEATURE_SPACE / study.key
    base_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": {"mae_after": 0.44}}
    metrics_path = base_dir / f"opinion_xgb_{study.key}_validation_metrics.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    result = sweeps._load_opinion_from_next_metrics_from_disk(
        opinion_dir=tmp_path,
        studies=[study],
    )

    assert result == {study.key: payload}


def test_load_loso_metrics_from_disk(tmp_path: Path) -> None:
    study = _make_study_spec()
    loso_dir = tmp_path / "loso" / study.evaluation_slug
    loso_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"accuracy": 0.59}
    (loso_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    result = sweeps._load_loso_metrics_from_disk(
        next_video_dir=tmp_path,
        studies=[study],
    )

    assert result[study.key]["accuracy"] == pytest.approx(0.59)


def test_write_reports_handles_disabled_sections(tmp_path: Path) -> None:
    sweep_report = reports.SweepReportData()
    reports._write_reports(
        reports_dir=tmp_path,
        sweeps=sweep_report,
        allow_incomplete=False,
        sections=reports.ReportSections(include_next_video=False),
    )

    catalog = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "next_video/README.md" not in catalog

    hyper = (tmp_path / "hyperparameter_tuning" / "README.md").read_text(encoding="utf-8")
    assert "disabled" in hyper.lower()

    opinion = (tmp_path / "opinion" / "README.md").read_text(encoding="utf-8")
    assert "disabled" in opinion.lower()
