#!/usr/bin/env python
# pylint: disable=missing-function-docstring
from __future__ import annotations

from pathlib import Path

import types

import grpo.pipeline_runner as pr
import grpo.pipeline as gp


class _Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Sel:
    def __init__(self, *, run_next_video: bool, run_opinion: bool):
        self.run_next_video = run_next_video
        self.run_opinion = run_opinion


class _Ctx:
    def __init__(self, root: Path, label: str):
        self.repo_root = root
        self.out_dir = root / "models" / "grpo"
        self.next_video_root = self.out_dir / "next_video"
        self.opinion_root = self.out_dir / "opinion"
        self.label = label

    @property
    def next_video_run_dir(self) -> Path:
        return self.next_video_root / self.label

    @property
    def opinion_run_dir(self) -> Path:
        return self.opinion_root / self.label


def _base_args() -> _Args:
    return _Args(
        dataset="local_ds",
        split="validation",
        cache_dir=None,
        model="stub-model",
        revision=None,
        dtype="auto",
        temperature=0.0,
        top_p=None,
        max_new_tokens=8,
        solution_key=None,
        max_history=1,
        eval_max=0,
        flush_interval=0,
        overwrite=False,
        issues="",
        studies="",
        opinion_studies="",
        opinion_max_participants=0,
        direction_tolerance=1e-6,
        reports_subdir="grpo",
        baseline_label="GRPO",
        regenerate_hint="",
    )


def test_run_evaluations_calls_selected_stages(monkeypatch, tmp_path: Path) -> None:
    args = _base_args()
    ctx = _Ctx(tmp_path, label="unit")
    sel = _Sel(run_next_video=True, run_opinion=False)

    # Avoid heavy deps
    monkeypatch.setattr(pr, "load_tokenizer_and_model", lambda **_: (object(), object()))

    captured = {}  # type: ignore[var-annotated]

    def fake_run_nv(tokenizer, model, settings, config_label, out_dir):  # noqa: ARG001
        captured.update({
            "nv_settings": settings,
            "nv_label": config_label,
            "nv_out": out_dir,
        })
        return types.SimpleNamespace(metrics={"ok": True}, run_dir=out_dir, metrics_path=out_dir / "m.json", predictions_path=out_dir / "p.jsonl", qa_log_path=out_dir / "qa.log")

    monkeypatch.setattr(pr, "run_next_video_evaluation", fake_run_nv)
    res = pr._run_evaluations(args, sel, ctx, prompts=types.SimpleNamespace(system="S", opinion="O"))

    assert getattr(res, "next_video") is not None
    assert getattr(res, "opinion") is None
    assert captured["nv_label"] == "unit"
    # Runner passes the stage root and uses config_label to create subdir
    assert captured["nv_out"] == ctx.next_video_root
    assert captured["nv_settings"].limits.max_examples == 0


def test_generate_reports_loads_cached_when_results_missing(monkeypatch, tmp_path: Path) -> None:
    ctx = _Ctx(tmp_path, label="unit")
    sel = _Sel(run_next_video=True, run_opinion=True)
    res = types.SimpleNamespace(next_video=None, opinion=None)
    args = _base_args()
    args.reports_subdir = "grpo_smoke"
    args.baseline_label = "GRPO"
    args.regenerate_hint = "hint"

    nv_obj = types.SimpleNamespace(metrics_path=ctx.next_video_run_dir / "metrics.json")
    op_obj = object()
    monkeypatch.setattr(pr, "_load_next_video_from_disk", lambda _: nv_obj)
    monkeypatch.setattr(pr, "_load_opinion_from_disk", lambda _: op_obj)

    captured = {}

    def fake_generate_reports(*, repo_root, next_video, opinion, options):  # noqa: ARG001
        captured.update({
            "repo_root": repo_root,
            "next_video": next_video,
            "opinion": opinion,
            "options": options,
        })

    monkeypatch.setattr(pr, "generate_reports", fake_generate_reports)
    # Silence summaries during test
    monkeypatch.setattr(pr, "_log_next_video_summary", lambda *_: None)
    monkeypatch.setattr(pr, "_log_opinion_summary", lambda *_: None)

    # Ensure repo root resolution honours monkeypatches
    monkeypatch.setattr(gp, "_repo_root", lambda: tmp_path, raising=False)
    pr._generate_reports_if_needed(sel, ctx, res, args)

    assert captured["repo_root"] == tmp_path
    assert captured["next_video"] is nv_obj
    assert captured["opinion"] is op_obj
    assert captured["options"].reports_subdir == "grpo_smoke"
