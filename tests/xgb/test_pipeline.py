"""Unit tests for the high-level XGBoost pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from xgb import pipeline


def test_split_tokens_trims_and_filters() -> None:
    assert pipeline._split_tokens(" alpha , beta ,, gamma ") == ["alpha", "beta", "gamma"]
    assert pipeline._split_tokens("") == []


def test_sanitize_token_replaces_special_characters() -> None:
    assert pipeline._sanitize_token("Model/Name 1.0") == "Model_Name_1p0"


def test_build_sweep_configs_supports_multiple_vectorisers() -> None:
    args, _ = pipeline._parse_args(
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
    configs = pipeline._build_sweep_configs(args)
    assert len(configs) == 3

    vectorizers = {config.text_vectorizer: config for config in configs}
    assert vectorizers["tfidf"].vectorizer_cli == ()
    assert vectorizers["word2vec"].vectorizer_tag == "w2v128"
    assert "--word2vec_size" in vectorizers["word2vec"].vectorizer_cli

    sentence_config = vectorizers["sentence_transformer"]
    assert sentence_config.vectorizer_tag.startswith("st_")
    assert "--sentence_transformer_model" in sentence_config.vectorizer_cli
    assert "--sentence_transformer_normalize" in sentence_config.vectorizer_cli


def test_resolve_issue_specs_filters_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "load_dataset_source", lambda dataset, cache_dir: {"dataset": dataset})
    monkeypatch.setattr(
        pipeline,
        "issues_in_dataset",
        lambda _dataset: ["gun_control", "minimum_wage", "climate_change"],
    )

    specs = pipeline._resolve_issue_specs(dataset="stub", cache_dir="/tmp/cache", requested=["gun_control"])
    assert [spec.name for spec in specs] == ["gun_control"]

    with pytest.raises(ValueError):
        pipeline._resolve_issue_specs(dataset="stub", cache_dir="/tmp/cache", requested=["unknown_issue"])


def test_run_sweeps_collects_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: List[Path] = []

    def fake_run_xgb_cli(args: List[str]) -> None:
        out_dir = Path(args[args.index("--out_dir") + 1])
        issue_name = args[args.index("--issues") + 1]
        metrics_path = out_dir / issue_name.replace(" ", "_") / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"accuracy": 0.81, "coverage": 0.56, "evaluated": 321}),
            encoding="utf-8",
        )
        recorded.append(metrics_path)

    monkeypatch.setattr(pipeline, "_run_xgb_cli", fake_run_xgb_cli)

    issue = pipeline.IssueSpec(name="gun_control", label="Gun Control")
    config = pipeline.SweepConfig(
        text_vectorizer="tfidf",
        vectorizer_tag="tfidf",
        learning_rate=0.1,
        max_depth=4,
        n_estimators=200,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        vectorizer_cli=(),
    )
    context = pipeline.SweepRunContext(
        base_cli=["--dataset", "stub"],
        extra_cli=["--eval_max", "100"],
        sweep_dir=tmp_path,
        tree_method="hist",
    )

    outcomes = pipeline._run_sweeps(issues=[issue], configs=[config], context=context)
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.metrics_path in recorded
    assert outcome.accuracy == pytest.approx(0.81)
    assert outcome.metrics["evaluated"] == 321


def test_select_best_configs_prefers_accuracy_then_coverage_then_support(tmp_path: Path) -> None:
    issue = pipeline.IssueSpec(name="gun_control", label="Gun Control")

    def make_outcome(accuracy: float, coverage: float, evaluated: int, tag: str) -> pipeline.SweepOutcome:
        config = pipeline.SweepConfig(
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
        return pipeline.SweepOutcome(
            issue=issue,
            config=config,
            accuracy=accuracy,
            coverage=coverage,
            evaluated=evaluated,
            metrics_path=tmp_path / f"{tag}.json",
            metrics={"accuracy": accuracy, "coverage": coverage, "evaluated": evaluated},
        )

    outcomes = [
        make_outcome(0.80, 0.60, 200, "a"),
        make_outcome(0.82, 0.55, 150, "b"),
        make_outcome(0.82, 0.60, 100, "c"),
        make_outcome(0.82, 0.60, 250, "d"),
    ]
    selection = pipeline._select_best_configs(outcomes)
    assert selection[issue.name].outcome.config.vectorizer_tag == "d"


def test_run_final_evaluations_reads_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run_xgb_cli(args: List[str]) -> None:
        out_dir = Path(args[args.index("--out_dir") + 1])
        issue_name = args[args.index("--issues") + 1]
        metrics_path = out_dir / issue_name.replace(" ", "_") / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"accuracy": 0.77, "coverage": 0.61, "evaluated": 128}),
            encoding="utf-8",
        )

    monkeypatch.setattr(pipeline, "_run_xgb_cli", fake_run_xgb_cli)

    issue = pipeline.IssueSpec(name="gun_control", label="Gun Control")
    outcome = pipeline.SweepOutcome(
        issue=issue,
        config=pipeline.SweepConfig(
            text_vectorizer="tfidf",
            vectorizer_tag="tfidf",
            learning_rate=0.1,
            max_depth=4,
            n_estimators=200,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            vectorizer_cli=(),
        ),
        accuracy=0.8,
        coverage=0.6,
        evaluated=200,
        metrics_path=tmp_path / "metrics.json",
        metrics={"accuracy": 0.8},
    )
    selection = pipeline.IssueSelection(issue=issue, outcome=outcome)
    context = pipeline.FinalEvalContext(
        base_cli=["--dataset", "stub"],
        extra_cli=["--seed", "13"],
        out_dir=tmp_path,
        tree_method="hist",
        save_model_dir=None,
    )

    metrics = pipeline._run_final_evaluations(selections={issue.name: selection}, context=context)
    assert metrics[issue.name]["accuracy"] == 0.77
    assert metrics[issue.name]["evaluated"] == 128


def test_group_requested_studies_defaults_to_all() -> None:
    grouped_default = pipeline._group_requested_studies(())
    default_keys = {spec.key for spec in pipeline.DEFAULT_SPECS}
    assert set().union(*[set(keys) for keys in grouped_default.values()]) == default_keys

    grouped_subset = pipeline._group_requested_studies(("study1",))
    assert set(grouped_subset.keys()) == {"gun_control"}
    assert grouped_subset["gun_control"] == ["study1"]


def test_run_opinion_stage_invokes_matching_studies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: Dict[str, Dict[str, object]] = {}

    def fake_run_opinion_eval(*, request, studies):
        captured["studies"] = list(studies)
        captured["booster_lr"] = request.train_config.booster.learning_rate
        return {studies[0]: {"label": "Study One"}}

    monkeypatch.setattr(pipeline, "run_opinion_eval", fake_run_opinion_eval)

    issue = pipeline.IssueSpec(name="gun_control", label="Gun Control")
    config = pipeline.SweepConfig(
        text_vectorizer="tfidf",
        vectorizer_tag="tfidf",
        learning_rate=0.15,
        max_depth=6,
        n_estimators=300,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        reg_alpha=0.1,
        vectorizer_cli=(),
    )
    outcome = pipeline.SweepOutcome(
        issue=issue,
        config=config,
        accuracy=0.8,
        coverage=0.6,
        evaluated=200,
        metrics_path=tmp_path / "metrics.json",
        metrics={"accuracy": 0.8},
    )
    selections = {issue.name: pipeline.IssueSelection(issue=issue, outcome=outcome)}
    stage_config = pipeline.OpinionStageConfig(
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
    )

    results = pipeline._run_opinion_stage(selections=selections, config=stage_config)
    assert captured["studies"] == ["study1"]
    assert captured["booster_lr"] == pytest.approx(0.15)
    assert "study1" in results


def test_format_float_three_decimal_precision() -> None:
    assert pipeline._format_float(0.12349) == "0.123"
    assert pipeline._format_float(0.12354) == "0.124"


def test_write_reports_generates_expected_readmes(tmp_path: Path) -> None:
    issue = pipeline.IssueSpec(name="gun_control", label="Gun Control")
    config = pipeline.SweepConfig(
        text_vectorizer="tfidf",
        vectorizer_tag="tfidf",
        learning_rate=0.1,
        max_depth=4,
        n_estimators=200,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        vectorizer_cli=(),
    )
    outcome = pipeline.SweepOutcome(
        issue=issue,
        config=config,
        accuracy=0.82,
        coverage=0.64,
        evaluated=256,
        metrics_path=tmp_path / "metrics.json",
        metrics={"accuracy": 0.82, "coverage": 0.64, "evaluated": 256},
    )
    selections = {issue.name: pipeline.IssueSelection(issue=issue, outcome=outcome)}
    final_metrics = {
        issue.name: {"accuracy": 0.81, "coverage": 0.62, "evaluated": 200},
    }
    opinion_metrics = {
        "study1": {
            "label": "Study One",
            "n_participants": 123,
            "metrics": {"mae_after": 1.2, "rmse_after": 1.8, "r2_after": 0.4},
            "baseline": {"mae_before": 1.5},
        }
    }

    pipeline._write_reports(
        reports_dir=tmp_path,
        outcomes=[outcome],
        selections=selections,
        final_metrics=final_metrics,
        opinion_metrics=opinion_metrics,
    )

    catalog = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "XGBoost Report Catalog" in catalog

    hyper = (tmp_path / "hyperparameter_tuning" / "README.md").read_text(encoding="utf-8")
    assert "| Config | Accuracy" in hyper
    assert "**tfidf_lr0p1_depth4_estim200_sub0p9_col0p8_l21_l10**" in hyper
    assert "### Configuration Leaderboards" in hyper
    assert "| 1 | **tfidf_lr0p1_depth4_estim200_sub0p9_col0p8_l21_l10** | 0.820 | 0.000 |" in hyper

    next_video = (tmp_path / "next_video" / "README.md").read_text(encoding="utf-8")
    assert "XGBoost Next-Video Baseline" in next_video
    assert "Gun Control" in next_video

    opinion = (tmp_path / "opinion" / "README.md").read_text(encoding="utf-8")
    assert "Study One" in opinion
