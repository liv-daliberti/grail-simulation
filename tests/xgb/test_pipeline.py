"""Unit tests for the refactored XGBoost pipeline helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

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
            metrics_path=tmp_path / f"{tag}.json",
            metrics={
                "metrics": {
                    "mae_after": mae,
                    "rmse_after": rmse,
                    "r2_after": r_squared,
                }
            },
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
    def fake_run_xgb_cli(args: List[str]) -> None:
        out_dir = Path(args[args.index("--out_dir") + 1])
        issue_name = args[args.index("--issues") + 1]
        study_name = args[args.index("--participant_studies") + 1]
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

    metrics = evaluate._run_final_evaluations(selections=selections, context=context)
    assert metrics[study.key]["accuracy"] == 0.77
    assert metrics[study.key]["evaluated"] == 128


def test_run_final_evaluations_reuses_metrics_and_sets_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    context = FinalEvalContext(
        base_cli=["--dataset", "stub"],
        extra_cli=["--extra", "flag"],
        out_dir=tmp_path / "out",
        tree_method="hist",
        save_model_dir=save_model_dir,
        reuse_existing=True,
    )

    metrics_path = context.out_dir / study.evaluation_slug / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {"accuracy": 0.83}
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")

    monkeypatch.setattr(
        evaluate,
        "_run_xgb_cli",
        lambda *_args, **_kwargs: pytest.fail("_run_xgb_cli should not be called when metrics cached"),
    )

    metrics = evaluate._run_final_evaluations(selections=selections, context=context)

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
        metrics_path=tmp_path / "opinion.json",
        metrics={"metrics": {"mae_after": 0.5, "rmse_after": 0.7, "r2_after": 0.2}},
    )
    selections = {study.key: OpinionStudySelection(study=study, outcome=outcome)}
    stage_config = OpinionStageConfig(
        dataset="dataset",
        cache_dir="cache",
        base_out_dir=tmp_path,
        extra_fields=("viewer_profile",),
        studies=("study1",),
        max_participants=50,
        seed=999,
        max_features=10,
        tree_method="hist",
        overwrite=True,
        reuse_existing=False,
    )

    results = evaluate._run_opinion_stage(selections=selections, config=stage_config)
    assert captured["studies"] == ["study1"]
    assert captured["learning_rate"] == pytest.approx(0.1)
    assert "study1" in results


def test_prepare_opinion_sweep_tasks_reuses_cached_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    study = _make_study_spec()
    config = _make_sweep_config()
    context = OpinionSweepRunContext(
        dataset="dataset",
        cache_dir="cache",
        sweep_dir=tmp_path,
        extra_fields=("viewer_profile",),
        max_participants=25,
        seed=123,
        max_features=None,
        tree_method="hist",
        overwrite=False,
    )

    run_root = context.sweep_dir / study.issue_slug / study.study_slug / config.label()
    metrics_path = (
        run_root
        / sweeps.OPINION_FEATURE_SPACE
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


def test_format_float_three_decimal_precision() -> None:
    assert reports._format_float(0.12349) == "0.123"
    assert reports._format_float(0.12354) == "0.124"


def test_write_reports_generates_expected_readmes(tmp_path: Path) -> None:
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

    reports._write_reports(
        reports_dir=tmp_path,
        outcomes=[outcome],
        selections=selections,
        final_metrics=final_metrics,
        opinion_metrics=opinion_metrics,
        allow_incomplete=False,
    )

    catalog = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "XGBoost Report Catalog" in catalog

    hyper = (tmp_path / "hyperparameter_tuning" / "README.md").read_text(encoding="utf-8")
    assert "| Config | Accuracy" in hyper
    assert "**tfidf_lr0p1_depth4_estim200_sub0p9_col0p8_l21_l10**" in hyper

    next_video = (tmp_path / "next_video" / "README.md").read_text(encoding="utf-8")
    assert "XGBoost Next-Video Baseline" in next_video
    assert "Study 1 – Gun Control (MTurk)" in next_video

    opinion = (tmp_path / "opinion" / "README.md").read_text(encoding="utf-8")
    assert "Study One" in opinion


def test_write_reports_handles_disabled_sections(tmp_path: Path) -> None:
    reports._write_reports(
        reports_dir=tmp_path,
        outcomes=[],
        selections={},
        final_metrics={},
        opinion_metrics={},
        allow_incomplete=False,
        include_next_video=False,
        include_opinion=False,
    )

    catalog = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "next_video" not in catalog.lower()

    hyper = (tmp_path / "hyperparameter_tuning" / "README.md").read_text(encoding="utf-8")
    assert "disabled" in hyper.lower()

    opinion = (tmp_path / "opinion" / "README.md").read_text(encoding="utf-8")
    assert "disabled" in opinion.lower()
