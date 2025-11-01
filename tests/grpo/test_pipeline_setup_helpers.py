#!/usr/bin/env python
# pylint: disable=missing-function-docstring
from __future__ import annotations

from pathlib import Path

import grpo.pipeline as pipeline
from grpo.pipeline_setup import (
    PipelineContext,
    PipelinePrompts,
    StageSelection,
    _build_context,
    _derive_label,
    _load_prompts,
    _resolve_out_dir,
    _resolve_stage_selection,
)


class _Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_derive_label_prefers_explicit_and_falls_back_to_model() -> None:
    assert _derive_label(_Args(label="run42", model="some/model")) == "run42"
    # Model path -> basename -> spaces/slashes normalised to underscore
    assert _derive_label(_Args(label=None, model="/a/b/Qwen 1.5B")) == "Qwen_1.5B"
    assert _derive_label(_Args()) == "grpo"


def test_resolve_out_dir_uses_repo_root_when_unset(monkeypatch, tmp_path: Path) -> None:
    # Point grpo.pipeline._repo_root to a temporary directory to avoid touching the real tree.
    monkeypatch.setattr(pipeline, "_repo_root", lambda: tmp_path, raising=False)
    out = _resolve_out_dir(_Args())
    assert out == tmp_path / "models" / "grpo"

    # Explicit out_dir should be respected as-is.
    custom = _resolve_out_dir(_Args(out_dir=str(tmp_path / "custom")))
    assert custom == tmp_path / "custom"


def test_load_prompts_uses_files_or_fallbacks(tmp_path: Path) -> None:
    sys_p = tmp_path / "sys.txt"
    op_p = tmp_path / "op.txt"
    sys_p.write_text("SYS", encoding="utf-8")
    op_p.write_text("OPN", encoding="utf-8")

    prompts = _load_prompts(_Args(system_prompt_file=str(sys_p), opinion_prompt_file=str(op_p)))
    assert isinstance(prompts, PipelinePrompts)
    assert prompts.system == "SYS"
    assert prompts.opinion == "OPN"


def test_resolve_stage_selection_inverts_negated_flags() -> None:
    sel = _resolve_stage_selection(_Args(stage="evaluate", no_next_video=False, no_opinion=True))
    assert isinstance(sel, StageSelection)
    assert sel.stage == "evaluate"
    assert sel.run_next_video is True
    assert sel.run_opinion is False
    assert sel.run_evaluations is True
    assert sel.run_reports is False


def test_build_context_populates_paths(monkeypatch, tmp_path: Path) -> None:
    # Patch repo root to avoid reliance on working tree
    monkeypatch.setattr(pipeline, "_repo_root", lambda: tmp_path, raising=False)
    args = _Args(out_dir=None, model="some/model", label=None)
    ctx = _build_context(args)
    assert isinstance(ctx, PipelineContext)
    assert ctx.repo_root == tmp_path
    assert ctx.out_dir == tmp_path / "models" / "grpo"
    assert ctx.next_video_root == ctx.out_dir / "next_video"
    assert ctx.opinion_root == ctx.out_dir / "opinion"
    assert ctx.label == "model"
    # Convenience properties
    assert ctx.next_video_run_dir == ctx.next_video_root / ctx.label
    assert ctx.opinion_run_dir == ctx.opinion_root / ctx.label

