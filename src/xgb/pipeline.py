"""High-level orchestration for the XGBoost baselines."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from .pipeline_context import (
    FinalEvalContext,
    OpinionStageConfig,
    StudySelection,
    StudySpec,
    SweepConfig,
    SweepOutcome,
    SweepRunContext,
)
from .pipeline_cli import (
    _parse_args,
    _repo_root,
    _default_out_dir,
    _default_cache_dir,
    _default_reports_dir,
    _split_tokens,
    _build_sweep_configs,
    _resolve_study_specs,
)
from .pipeline_sweeps import (
    _run_xgb_cli,
    _load_metrics,
    _sweep_outcome_from_metrics,
    _prepare_sweep_tasks,
    _merge_sweep_outcomes,
    _opinion_sweep_outcome_from_metrics,
    _prepare_opinion_sweep_tasks,
    _merge_opinion_sweep_outcomes,
    _execute_opinion_sweep_task,
    _execute_sweep_tasks,
    _emit_sweep_plan,
    _emit_combined_sweep_plan,
    _format_sweep_task_descriptor,
    _format_opinion_sweep_task_descriptor,
    _gpu_tree_method_supported,
    _load_final_metrics_from_disk,
    _load_opinion_metrics_from_disk,
    _run_sweeps,
    _execute_sweep_task,
    _select_best_configs,
    _select_best_opinion_configs,
)
from .pipeline_evaluate import _run_final_evaluations, _run_opinion_stage
from .pipeline_reports import _write_reports

LOGGER = logging.getLogger("xgb.pipeline")

__all__ = ["main", "SweepRunContext", "OpinionStageConfig"]



def main(argv: Sequence[str] | None = None) -> None:
    """Entry point orchestrating the full XGBoost workflow."""

    args, extra_cli = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    root = _repo_root()
    dataset = args.dataset or str(root / "data" / "cleaned_grail")
    cache_dir = args.cache_dir or str(_default_cache_dir(root))
    out_dir = Path(args.out_dir or _default_out_dir(root))
    sweep_dir = Path(args.sweep_dir or (out_dir / "sweeps"))
    reports_dir = Path(args.reports_dir or _default_reports_dir(root))

    jobs_value = getattr(args, "jobs", 1) or 1
    env_jobs = os.environ.get("XGB_JOBS")
    if env_jobs:
        try:
            jobs_value = int(env_jobs)
        except ValueError:
            LOGGER.warning("Ignoring invalid XGB_JOBS value '%s'.", env_jobs)
    jobs = max(1, jobs_value)

    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    os.environ.setdefault("HF_HOME", cache_dir)

    allow_incomplete = getattr(args, "allow_incomplete", True)
    env_allow = os.environ.get("XGB_ALLOW_INCOMPLETE")
    if env_allow is not None:
        allow_incomplete = env_allow.lower() not in {"0", "false", "no"}

    issue_tokens = _split_tokens(args.issues)
    study_tokens = _split_tokens(args.studies)
    study_specs = _resolve_study_specs(
        dataset=dataset,
        cache_dir=cache_dir,
        requested_issues=issue_tokens,
        requested_studies=study_tokens,
        allow_incomplete=allow_incomplete,
    )
    extra_fields = _split_tokens(args.extra_text_fields)

    LOGGER.info("Parallel sweep jobs: %d", jobs)
    tree_method = args.tree_method or "gpu_hist"
    if tree_method == "gpu_hist" and not _gpu_tree_method_supported():
        LOGGER.warning(
            "Requested tree_method=gpu_hist but the installed XGBoost build lacks GPU support. "
            "Falling back to tree_method=hist."
        )
        tree_method = "hist"
    args.tree_method = tree_method
    LOGGER.info("Using XGBoost tree_method=%s.", tree_method)

    base_cli: List[str] = [
        "--fit_model",
        "--dataset",
        dataset,
        "--cache_dir",
        cache_dir,
        "--max_train",
        str(args.max_train),
        "--max_features",
        str(args.max_features),
        "--eval_max",
        str(args.eval_max),
        "--seed",
        str(args.seed),
    ]
    if extra_fields:
        base_cli.extend(["--extra_text_fields", ",".join(extra_fields)])
    base_cli.append("--log_level")
    base_cli.append(args.log_level.upper())
    if args.overwrite:
        base_cli.append("--overwrite")

    configs = _build_sweep_configs(args)
    stage = getattr(args, "stage", "full")
    reuse_sweeps = not args.overwrite
    reuse_final = reuse_sweeps
    if args.reuse_final is not None:
        reuse_final = args.reuse_final
    reuse_final_env = os.environ.get("XGB_REUSE_FINAL")
    if reuse_final_env is not None:
        reuse_final = reuse_final_env.lower() not in {"0", "false", "no"}

    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_context = SweepRunContext(
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=sweep_dir,
        tree_method=args.tree_method,
        jobs=jobs,
    )

    planned_tasks, cached_planned = _prepare_sweep_tasks(
        studies=study_specs,
        configs=configs,
        context=sweep_context,
        reuse_existing=reuse_sweeps,
    )

    if stage == "plan":
        LOGGER.info(
            "Planned %d sweep configurations (%d cached).",
            len(planned_tasks),
            len(cached_planned),
        )
        _emit_sweep_plan(planned_tasks)
        return

    if args.dry_run:
        LOGGER.info(
            "Dry-run mode. Pending sweep tasks: %d (cached: %d).",
            len(planned_tasks),
            len(cached_planned),
        )
        return

    if stage == "sweeps":
        total_tasks = len(planned_tasks)
        if total_tasks == 0:
            LOGGER.info("No sweep tasks pending; existing metrics cover the grid.")
            return
        task_id = args.sweep_task_id
        if task_id is None:
            env_value = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env_value is None:
                raise RuntimeError(
                    "Sweep stage requires --sweep-task-id or SLURM_ARRAY_TASK_ID to be set."
                )
            try:
                task_id = int(env_value)
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid SLURM_ARRAY_TASK_ID '{env_value}'; expected an integer."
                ) from exc
        if args.sweep_task_count is not None and args.sweep_task_count != total_tasks:
            LOGGER.warning(
                "Sweep task count mismatch: expected=%d provided=%d.",
                total_tasks,
                args.sweep_task_count,
            )
        if task_id < 0 or task_id >= total_tasks:
            raise RuntimeError(
                f"Sweep task index {task_id} outside valid range 0..{total_tasks - 1}."
            )
        task = planned_tasks[task_id]
        if reuse_sweeps and task.metrics_path.exists():
            LOGGER.info(
                "Skipping sweep task %d (%s | %s | %s); metrics already exist at %s.",
                task.index,
                task.study.key,
                task.study.issue,
                task.config.label(),
                task.metrics_path,
            )
            return
        outcome = _execute_sweep_task(task)
        LOGGER.info(
            "Completed sweep task %d (%s | %s | %s). Metrics stored at %s.",
            outcome.order_index,
            task.study.key,
            task.study.issue,
            task.config.label(),
            outcome.metrics_path,
        )
        return

    reuse_for_stage = reuse_sweeps
    if stage in {"finalize", "reports"}:
        reuse_for_stage = True
    pending_tasks, cached_outcomes = _prepare_sweep_tasks(
        studies=study_specs,
        configs=configs,
        context=sweep_context,
        reuse_existing=reuse_for_stage,
    )
    if stage in {"finalize", "reports"} and pending_tasks:
        missing = ", ".join(_format_sweep_task_descriptor(task) for task in pending_tasks[:5])
        more = "" if len(pending_tasks) <= 5 else f", … ({len(pending_tasks)} total)"
        message = (
            "Sweep metrics missing for the following tasks: "
            f"{missing}{more}. Run --stage=sweeps to populate them."
        )
        if allow_incomplete:
            LOGGER.warning("%s Continuing because allow-incomplete mode is enabled.", message)
        else:
            raise RuntimeError(message)

    executed_outcomes: List[SweepOutcome] = []
    if stage == "full":
        executed_outcomes = _execute_sweep_tasks(pending_tasks, jobs=jobs)

    outcomes = _merge_sweep_outcomes(cached_outcomes, executed_outcomes)
    if not outcomes:
        if allow_incomplete:
            LOGGER.warning(
                "No sweep outcomes available; reports will contain placeholders because allow-incomplete mode is enabled."
            )
        else:
            raise RuntimeError("No sweep outcomes available; ensure sweeps have completed.")

    selections = _select_best_configs(outcomes)
    if not selections:
        if allow_incomplete:
            LOGGER.warning(
                "Failed to select configurations for any study; downstream reports will rely on placeholders."
            )
        else:
            raise RuntimeError("Failed to select a configuration for any study.")
    else:
        LOGGER.info(
            "Selected configurations: %s",
            ", ".join(
                f"{selection.study.key} ({selection.study.issue})"
                for selection in selections.values()
            ),
        )

    final_eval_context = FinalEvalContext(
        base_cli=base_cli,
        extra_cli=extra_cli,
        out_dir=out_dir / "next_video",
        tree_method=args.tree_method,
        save_model_dir=Path(args.save_model_dir) if args.save_model_dir else None,
        reuse_existing=reuse_final,
    )

    if stage == "reports":
        final_metrics = _load_final_metrics_from_disk(
            next_video_dir=final_eval_context.out_dir,
            studies=study_specs,
        )
        if not final_metrics:
            message = (
                f"No final metrics found under {final_eval_context.out_dir}. "
                "Run --stage=finalize before generating reports."
            )
            if allow_incomplete:
                LOGGER.warning("%s Continuing because allow-incomplete mode is enabled.", message)
            else:
                raise RuntimeError(message)
        opinion_metrics = _load_opinion_metrics_from_disk(
            opinion_dir=out_dir / "opinion",
            studies=study_specs,
        )
        _write_reports(
            reports_dir=reports_dir,
            outcomes=outcomes,
            selections=selections,
            final_metrics=final_metrics,
            opinion_metrics=opinion_metrics,
            allow_incomplete=allow_incomplete,
        )
        return

    final_metrics = _run_final_evaluations(selections=selections, context=final_eval_context)

    opinion_stage_config = OpinionStageConfig(
        dataset=dataset,
        cache_dir=cache_dir,
        base_out_dir=out_dir,
        extra_fields=extra_fields,
        studies=study_tokens,
        max_participants=args.opinion_max_participants,
        seed=args.seed,
        max_features=args.max_features if args.max_features > 0 else None,
        tree_method=args.tree_method,
        overwrite=args.overwrite or not reuse_final,
        reuse_existing=reuse_final,
    )
    opinion_metrics = _run_opinion_stage(selections=selections, config=opinion_stage_config)

    if stage == "finalize":
        return

    _write_reports(
        reports_dir=reports_dir,
        outcomes=outcomes,
        selections=selections,
        final_metrics=final_metrics,
        opinion_metrics=opinion_metrics,
        allow_incomplete=allow_incomplete,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
