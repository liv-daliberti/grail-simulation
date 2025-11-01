import os
from pathlib import Path

import pytest


def test_reexports_importable():
    from xgb.pipeline.context import SweepConfig, NextVideoMetricSummary, OpinionSummary

    assert SweepConfig.__name__ == "SweepConfig"
    assert NextVideoMetricSummary.__name__ == "NextVideoMetricSummary"
    assert OpinionSummary.__name__ == "OpinionSummary"


def test_sweep_config_label_and_cli_and_booster():
    from xgb.pipeline.context import SweepConfig

    cfg = SweepConfig(
        text_vectorizer="tfidf",
        vectorizer_tag="tfidf",
        learning_rate=0.1,
        max_depth=3,
        n_estimators=50,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
    )
    assert cfg.label() == "tfidf_lr0p1_depth3_estim50_sub1_col1_l21_l10"

    cli = cfg.cli_args("hist")
    assert "--xgb_learning_rate" in cli and "0.1" in cli
    assert "--xgb_tree_method" in cli and "hist" in cli

    params = cfg.booster_params("gpu_hist")
    assert params.learning_rate == pytest.approx(0.1)
    assert params.max_depth == 3
    assert params.n_estimators == 50
    assert params.subsample == pytest.approx(1.0)
    assert params.colsample_bytree == pytest.approx(1.0)
    assert params.reg_lambda == pytest.approx(1.0)
    assert params.reg_alpha == pytest.approx(0.0)


def test_opinion_sweep_outcome_from_kwargs_and_forwarding(tmp_path: Path):
    from common.pipeline.types import StudySpec
    from common.opinion.sweep_types import MetricsArtifact, AccuracySummary
    from xgb.pipeline.context import SweepConfig, OpinionSweepOutcome

    study = StudySpec(key="study-A", issue="issue-1", label="Study A")
    cfg = SweepConfig(
        text_vectorizer="tfidf",
        vectorizer_tag="tfidf",
        learning_rate=0.2,
        max_depth=4,
        n_estimators=25,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.1,
    )
    artifact = MetricsArtifact(path=tmp_path / "metrics.json", payload={"ok": True})
    acc = AccuracySummary(value=0.9, baseline=0.8, delta=0.1, eligible=200)

    outcome = OpinionSweepOutcome(
        order_index=7,
        study=study,
        config=cfg,
        mae=1.1,
        rmse=2.2,
        artifact=artifact,
        accuracy_summary=acc,
        r_squared=0.42,
    )

    assert outcome.order_index == 7
    assert outcome.study == study
    assert outcome.config == cfg
    assert outcome.mae == pytest.approx(1.1)
    assert outcome.rmse == pytest.approx(2.2)
    assert outcome.metrics_path == artifact.path
    assert outcome.metrics == artifact.payload
    assert outcome.accuracy == pytest.approx(0.9)
    assert outcome.baseline_accuracy == pytest.approx(0.8)
    assert outcome.accuracy_delta == pytest.approx(0.1)
    assert outcome.eligible == 200
    assert outcome.r_squared == pytest.approx(0.42)


def test_next_video_metric_summary_create():
    from xgb.pipeline.context import NextVideoMetricSummary

    summary = NextVideoMetricSummary.create(
        accuracy=0.75, coverage=0.9, evaluated=17, dataset="ds1", study_label="S1"
    )
    assert summary.accuracy == pytest.approx(0.75)
    assert summary.coverage == pytest.approx(0.9)
    assert summary.evaluated == 17
    assert summary.dataset == "ds1"
    assert summary.study_label == "S1"


def test_opinion_summary_from_kwargs():
    from xgb.pipeline.context import OpinionSummary

    s = OpinionSummary.from_kwargs(
        mae_after=0.8,
        mae_change=-0.1,
        rmse_after=1.2,
        r2_after=0.5,
        rmse_change=-0.2,
        accuracy_after=0.7,
        baseline_mae=0.9,
        baseline_rmse_change=-0.1,
        baseline_accuracy=0.6,
        calibration_slope=0.95,
        calibration_intercept=0.05,
        calibration_ece=0.02,
        kl_divergence_change=-0.01,
        mae_delta=-0.1,
        accuracy_delta=0.1,
        participants=100,
        eligible=80,
        dataset="dset",
        split="test",
        label="L1",
    )
    assert s.mae_after == pytest.approx(0.8)
    assert s.rmse_after == pytest.approx(1.2)
    assert s.r2_after == pytest.approx(0.5)
    assert s.baseline_mae == pytest.approx(0.9)
    assert s.calibration_slope == pytest.approx(0.95)
    assert s.mae_delta == pytest.approx(-0.1)
    assert s.accuracy_delta == pytest.approx(0.1)
    assert s.participants == 100
    assert s.dataset == "dset"
    assert s.label == "L1"


def test_opinion_sweep_run_context_legacy_kwargs():
    from xgb.pipeline.context import OpinionSweepRunContext
    from xgb.core.vectorizers import (
        TfidfConfig,
        Word2VecVectorizerConfig,
        SentenceTransformerVectorizerConfig,
    )

    ctx = OpinionSweepRunContext(
        sweep_dir=Path("."),
        dataset="ds",
        cache_dir="cache",
        extra_fields=("a", "b"),
        max_participants=50,
        seed=1337,
        max_features=1000,
        tree_method="hist",
        overwrite=True,
    )
    assert ctx.dataset == "ds"
    assert ctx.cache_dir == "cache"
    assert tuple(ctx.extra_fields) == ("a", "b")
    assert ctx.max_participants == 50
    assert ctx.seed == 1337
    assert ctx.max_features == 1000
    # Vectorizer configs default to instances
    assert isinstance(ctx.tfidf_config, TfidfConfig)
    assert isinstance(ctx.word2vec_config, Word2VecVectorizerConfig)
    assert isinstance(ctx.sentence_transformer_config, SentenceTransformerVectorizerConfig)
    assert ctx.tree_method == "hist"
    assert ctx.overwrite is True
