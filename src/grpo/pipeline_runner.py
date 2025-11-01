#!/usr/bin/env python
"""Evaluation and reporting runners for :mod:`grpo.pipeline`."""

from __future__ import annotations

import logging
from pathlib import Path
import time

from common.rlhf.reports import ReportOptions
from common.utils.repo import resolve_repo_root_from_monkeypatch

from .model import GenerationSettings, load_tokenizer_and_model
from .next_video import (
    FilterSelection,
    NextVideoDatasetSpec,
    NextVideoEvaluationLimits,
    NextVideoEvaluationSettings,
    NextVideoPromptSettings,
    run_next_video_evaluation,
)
from .opinion import (
    OpinionDatasetSpec,
    OpinionEvaluationControls,
    OpinionEvaluationSettings,
    OpinionInferenceContext,
    OpinionPromptSettings,
    run_opinion_evaluation,
)
from .pipeline_common import (
    _comma_separated,
    _status,
    _log_next_video_summary,
    _log_opinion_summary,
)
from .pipeline_loaders import _load_next_video_from_disk, _load_opinion_from_disk
from .reports import generate_reports


def _resolve_repo_root_with_patch(context_repo_root: Path) -> Path:
    """Resolve repo root honoring any grpo.pipeline monkeypatch."""
    return resolve_repo_root_from_monkeypatch("grpo.pipeline", context_repo_root)


LOGGER = logging.getLogger("grpo.pipeline")


def _run_evaluations(
    args,
    selection,
    context,
    prompts,
):
    """Run the requested evaluation stages and return results container."""

    results = type("PipelineResults", (), {})()  # duck-typed container
    results.next_video = None
    results.opinion = None

    tokenizer = None
    model = None
    generation = None

    # Shared model + generation settings for all evaluations
    tokenizer, model = load_tokenizer_and_model(
        model_name_or_path=args.model,
        revision=args.revision,
        dtype=args.dtype,
    )
    generation = GenerationSettings(
        temperature=float(args.temperature),
        top_p=float(args.top_p) if args.top_p is not None else None,
        max_new_tokens=int(args.max_new_tokens),
    )

    # Next-video evaluation
    if selection.run_next_video:
        stage_start = time.perf_counter()
        settings = NextVideoEvaluationSettings(
            model_label=str(args.model),
            dataset=NextVideoDatasetSpec(
                name=args.dataset,
                split=args.split,
                cache_dir=(str(args.cache_dir) if getattr(args, "cache_dir", None) else None),
            ),
            prompts=NextVideoPromptSettings(
                system_prompt=prompts.system,
                solution_key=args.solution_key,
                max_history=int(args.max_history),
            ),
            limits=NextVideoEvaluationLimits(
                max_examples=int(args.eval_max or 0),
                flush_every=int(args.flush_interval or 0),
            ),
            overwrite=bool(args.overwrite),
            generation=generation,
            filters=FilterSelection.from_raw(
                issues=_comma_separated(args.issues),
                studies=_comma_separated(args.studies),
            ),
        )
        run_dir = context.next_video_run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        results.next_video = run_next_video_evaluation(
            tokenizer=tokenizer,
            model=model,
            settings=settings,
            config_label=context.label,
            out_dir=run_dir,
        )
        _status("Next-video evaluation finished in %.2fs", time.perf_counter() - stage_start)

    # Opinion evaluation
    if selection.run_opinion:
        stage_start = time.perf_counter()
        opinion_settings = OpinionEvaluationSettings(
            dataset=OpinionDatasetSpec(
                name=args.dataset,
                split=args.split,
                cache_dir=(str(args.cache_dir) if getattr(args, "cache_dir", None) else None),
            ),
            prompts=OpinionPromptSettings(
                system=prompts.system,
                opinion=prompts.opinion,
                solution_key=args.solution_key,
                max_history=int(args.max_history),
            ),
            controls=OpinionEvaluationControls(
                max_participants=int(args.opinion_max_participants or 0),
                direction_tolerance=float(args.direction_tolerance),
                overwrite=bool(args.overwrite),
                flush_every=int(args.flush_interval or 0),
            ),
            include_studies=_comma_separated(args.opinion_studies),
        )
        opinion_dir = context.opinion_run_dir
        opinion_dir.mkdir(parents=True, exist_ok=True)
        results.opinion = run_opinion_evaluation(
            context=OpinionInferenceContext(
                tokenizer=tokenizer,
                model=model,
                generation=generation,
            ),
            settings=opinion_settings,
            out_dir=opinion_dir,
        )
        _status(
            "Opinion evaluation finished in %.2fs",
            time.perf_counter() - stage_start,
        )

    return results


def _generate_reports_if_needed(
    selection,
    context,
    results,
    args,
) -> None:
    """Load cached results when necessary and generate reports."""

    next_result = getattr(results, "next_video", None)
    if selection.run_next_video and next_result is None:
        next_result = _load_next_video_from_disk(context.next_video_run_dir)
        if next_result is not None:
            LOGGER.info(
                "Loaded cached next-video metrics from %s", next_result.metrics_path
            )
            _log_next_video_summary(next_result)

    opinion_result = getattr(results, "opinion", None)
    if selection.run_opinion and opinion_result is None:
        opinion_result = _load_opinion_from_disk(context.opinion_run_dir)
        if opinion_result is not None:
            LOGGER.info(
                "Loaded cached opinion metrics from %s",
                context.opinion_run_dir / "combined_metrics.json",
            )
            _log_opinion_summary(opinion_result)

    repo_root = _resolve_repo_root_with_patch(context.repo_root)
    generate_reports(
        repo_root=repo_root,
        next_video=next_result if selection.run_next_video else None,
        opinion=opinion_result if selection.run_opinion else None,
        options=ReportOptions(
            args.reports_subdir,  # reports_subdir
            args.baseline_label,  # baseline_label
            args.regenerate_hint or None,
        ),
    )
    _status(
        "Reports written under %s.",
        repo_root / "reports" / args.reports_subdir,
    )
