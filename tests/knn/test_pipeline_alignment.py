"""Tests for sentence-transformer context alignment in the KNN pipeline.

These tests ensure that `_align_sentence_transformer_context` updates the
immutable `PipelineContext` via its property setters without raising
`FrozenInstanceError`, and that guard conditions behave as expected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knn.pipeline import _align_sentence_transformer_context
from knn.pipeline.context import PipelineContext


def _make_context(
    *,
    tmp_path: Path,
    feature_spaces: tuple[str, ...] = ("tfidf", "sentence_transformer"),
    sentence_device: str | None = "cpu",
    sentence_batch_size: int = 16,
    sentence_normalize: bool = False,
    reuse_sweeps: bool = True,
    reuse_final: bool = False,
) -> PipelineContext:
    out_dir = tmp_path / "out"
    paths = {
        "dataset": "stub",
        "out_dir": out_dir,
        "cache_dir": str(tmp_path / "cache"),
        "sweep_dir": out_dir / "next_video" / "sweeps",
        "opinion_sweep_dir": out_dir / "opinions" / "sweeps",
        "word2vec_model_dir": out_dir / "next_video" / "word2vec_models",
        "opinion_word2vec_dir": out_dir / "opinions" / "word2vec_models",
        "next_video_dir": out_dir / "next_video",
        "opinion_dir": out_dir / "opinions",
    }
    settings = {
        "k_sweep": "1,2,5",
        "study_tokens": tuple(),
        "word2vec_epochs": 1,
        "word2vec_workers": 1,
        "sentence_model": "sentence-transformers/all-mpnet-base-v2",
        "sentence_device": sentence_device,
        "sentence_batch_size": sentence_batch_size,
        "sentence_normalize": sentence_normalize,
        "feature_spaces": feature_spaces,
        "jobs": 1,
        "reuse_sweeps": reuse_sweeps,
        "reuse_final": reuse_final,
        "allow_incomplete": True,
        "run_next_video": True,
        "run_opinion": True,
    }
    return PipelineContext.from_mappings(paths=paths, settings=settings)


def _mk_config_dir(root: Path, label: str) -> None:
    # Create a directory structure discoverable by `_iter_sentence_transformer_config_dirs`:
    #   <root>/sentence_transformer/<study>/<label>/
    cfg = root / "sentence_transformer" / "study1" / label
    cfg.mkdir(parents=True, exist_ok=True)


def test_align_updates_via_property_setters(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path=tmp_path)

    # Seed cached config under sweep_dir that encodes cuda, bs64, norm
    _mk_config_dir(ctx.sweep_dir, "device-cuda_bs64_norm")

    # Should not raise FrozenInstanceError and should update values
    _align_sentence_transformer_context(ctx, stage="finalize")

    assert ctx.sentence_device == "cuda"
    assert ctx.sentence_batch_size == 64
    assert ctx.sentence_normalize is True


def test_align_considers_opinion_sweep_dir(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path=tmp_path)

    # Only create cached config under the opinion sweep root
    _mk_config_dir(ctx.opinion_sweep_dir, "device-cpu_bs8_nonorm")

    _align_sentence_transformer_context(ctx, stage="finalize")

    assert ctx.sentence_device == "cpu"
    assert ctx.sentence_batch_size == 8
    assert ctx.sentence_normalize is False


def test_align_skips_when_feature_space_disabled(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path=tmp_path, feature_spaces=("tfidf",))

    # Create a config; it should be ignored because feature space is disabled
    _mk_config_dir(ctx.sweep_dir, "device-cuda_bs64_norm")

    _align_sentence_transformer_context(ctx, stage="finalize")

    # Unchanged from defaults
    assert ctx.sentence_device == "cpu"
    assert ctx.sentence_batch_size == 16
    assert ctx.sentence_normalize is False


def test_align_skips_when_not_considering_reuse(tmp_path: Path) -> None:
    ctx = _make_context(
        tmp_path=tmp_path,
        reuse_sweeps=False,
        reuse_final=False,
    )

    _mk_config_dir(ctx.sweep_dir, "device-cuda_bs64_norm")

    # Stage 'sweeps' with no reuse flags should skip alignment
    _align_sentence_transformer_context(ctx, stage="sweeps")

    assert ctx.sentence_device == "cpu"
    assert ctx.sentence_batch_size == 16
    assert ctx.sentence_normalize is False

