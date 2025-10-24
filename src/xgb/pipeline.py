#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipeline orchestration for the XGBoost baselines.

This module manages sweep execution, evaluation, and report emission for
the slate-ranking and opinion-regression XGBoost workflows."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from common.pipeline_stage import prepare_sweep_execution
from common.prompt_docs import merge_default_extra_fields

from .pipeline_context import (
    FinalEvalContext,
    OpinionStageConfig,
    OpinionStudySelection,
    OpinionSweepOutcome,
    OpinionSweepRunContext,
    OpinionSweepTask,
    StudySelection,
    SweepOutcome,
    SweepRunContext,
    SweepTask,
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
    _prepare_opinion_sweep_tasks,
    _prepare_sweep_tasks,
    _merge_opinion_sweep_outcomes,
    _merge_sweep_outcomes,
    _execute_opinion_sweep_tasks,
    _execute_sweep_tasks,
    _emit_combined_sweep_plan,
    _format_opinion_sweep_task_descriptor,
    _format_sweep_task_descriptor,
    _gpu_tree_method_supported,
    _load_final_metrics_from_disk,
    _load_opinion_metrics_from_disk,
    _select_best_configs,
    _select_best_opinion_configs,
)
from .pipeline_evaluate import _run_final_evaluations, _run_opinion_stage
from .pipeline_reports import OpinionReportData, SweepReportData, _write_reports

LOGGER = logging.getLogger("xgb.pipeline")

__all__ = ["main", "SweepRunContext", "OpinionStageConfig", "OpinionSweepRunContext"]



def main(argv: Sequence[str] | None = None) -> None:
    """
    Entry point orchestrating the full XGBoost workflow.

    :param argv: Optional override for command-line arguments
        (defaults to ``sys.argv[1:]``).
    :type argv: Sequence[str] | None
    """
    # pylint: disable=too-many-locals,too-many-branches
    # pylint: disable=too-many-return-statements,too-many-statements

    args, extra_cli = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    run_next_video = getattr(args, "run_next_video", True)
    run_opinion = getattr(args, "run_opinion", True)
    if not run_next_video and not run_opinion:
        LOGGER.warning("No tasks selected; exiting.")
        return

    root = _repo_root()
    dataset = args.dataset or str(root / "data" / "cleaned_grail")
    cache_dir = args.cache_dir or str(_default_cache_dir(root))
    out_dir = Path(args.out_dir or _default_out_dir(root))
    next_video_dir = out_dir / "next_video"
    opinion_dir = out_dir / "opinions"
    sweep_dir = Path(args.sweep_dir or (next_video_dir / "sweeps"))
    opinion_sweep_dir = Path(os.environ.get("XGB_OPINION_SWEEP_DIR") or (opinion_dir / "sweeps"))
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
    extra_fields = merge_default_extra_fields(_split_tokens(args.extra_text_fields))

    task_log: List[str] = []
    if run_next_video:
        task_log.append("next_video")
    if run_opinion:
        task_log.append("opinion")
    LOGGER.info("Selected pipeline tasks: %s.", ", ".join(task_log))

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
    sweep_context: SweepRunContext | None = None
    if run_next_video:
        sweep_context = SweepRunContext(
            base_cli=base_cli,
            extra_cli=extra_cli,
            sweep_dir=sweep_dir,
            tree_method=args.tree_method,
            jobs=jobs,
        )

    opinion_sweep_context: OpinionSweepRunContext | None = None
    if run_opinion:
        opinion_sweep_context = OpinionSweepRunContext(
            dataset=dataset,
            cache_dir=cache_dir,
            sweep_dir=opinion_sweep_dir,
            extra_fields=tuple(extra_fields),
            max_participants=args.opinion_max_participants,
            seed=args.seed,
            max_features=args.max_features if args.max_features > 0 else None,
            tree_method=args.tree_method,
            overwrite=args.overwrite,
        )
        opinion_sweep_context.sweep_dir.mkdir(parents=True, exist_ok=True)

    planned_slate_tasks: List[SweepTask] = []
    cached_slate_planned: List[SweepOutcome] = []
    if run_next_video and sweep_context is not None:
        planned_slate_tasks, cached_slate_planned = _prepare_sweep_tasks(
            studies=study_specs,
            configs=configs,
            context=sweep_context,
            reuse_existing=reuse_sweeps,
        )

    planned_opinion_tasks: List[OpinionSweepTask] = []
    cached_opinion_planned: List[OpinionSweepOutcome] = []
    if run_opinion and opinion_sweep_context is not None:
        planned_opinion_tasks, cached_opinion_planned = _prepare_opinion_sweep_tasks(
            studies=study_specs,
            configs=configs,
            context=opinion_sweep_context,
            reuse_existing=reuse_sweeps,
        )

    if stage == "plan":
        summary_bits: List[str] = []
        if run_next_video:
            summary_bits.append(
                f"next-video sweeps={len(planned_slate_tasks)} "
                f"(cached={len(cached_slate_planned)})"
            )
        if run_opinion:
            summary_bits.append(
                f"opinion sweeps={len(planned_opinion_tasks)} "
                f"(cached={len(cached_opinion_planned)})"
            )
        LOGGER.info("Planned sweep tasks: %s.", "; ".join(summary_bits))
        _emit_combined_sweep_plan(
            slate_tasks=planned_slate_tasks,
            opinion_tasks=planned_opinion_tasks,
        )
        return

    if args.dry_run:
        summary_bits: List[str] = []
        if run_next_video:
            summary_bits.append(
                f"next-video pending={len(planned_slate_tasks)} "
                f"cached={len(cached_slate_planned)}"
            )
        if run_opinion:
            summary_bits.append(
                f"opinion pending={len(planned_opinion_tasks)} "
                f"cached={len(cached_opinion_planned)}"
            )
        LOGGER.info(
            "Dry-run mode. %s.",
            "; ".join(summary_bits) if summary_bits else "No tasks selected.",
        )
        return

    if stage == "sweeps":
        slate_pending_by_index = {task.index: task for task in planned_slate_tasks}
        slate_cached_by_index = {
            outcome.order_index: outcome
            for outcome in cached_slate_planned
            if outcome.order_index not in slate_pending_by_index
        }
        slate_indices = set(slate_pending_by_index).union(slate_cached_by_index)
        slate_total = (max(slate_indices) + 1) if slate_indices else 0

        opinion_pending_by_index = {task.index: task for task in planned_opinion_tasks}
        opinion_cached_by_index = {
            outcome.order_index: outcome
            for outcome in cached_opinion_planned
            if outcome.order_index not in opinion_pending_by_index
        }
        opinion_indices = set(opinion_pending_by_index).union(opinion_cached_by_index)
        opinion_total = (max(opinion_indices) + 1) if opinion_indices else 0

        total_tasks = slate_total + opinion_total
        task_id = prepare_sweep_execution(
            total_tasks=total_tasks,
            cli_task_id=args.sweep_task_id,
            cli_task_count=args.sweep_task_count,
            logger=LOGGER,
        )
        if task_id is None:
            return
        if task_id < slate_total:
            task = slate_pending_by_index.get(task_id)
            if task is None:
                cached_outcome = slate_cached_by_index.get(task_id)
                if cached_outcome is not None:
                    LOGGER.info(
                        "Skipping sweep task %d (%s | %s | %s); metrics already exist at %s.",
                        task_id,
                        cached_outcome.study.key,
                        cached_outcome.study.issue,
                        cached_outcome.config.label(),
                        cached_outcome.metrics_path,
                    )
                    return
                LOGGER.warning(
                    "No sweep task registered for index %d; skipping.",
                    task_id,
                )
                return
            if reuse_sweeps and task.metrics_path.exists():
                LOGGER.info(
                    "Skipping sweep task %d (%s); metrics already exist at %s.",
                    task.index,
                    _format_sweep_task_descriptor(task),
                    task.metrics_path,
                )
                return
            outcome = _execute_sweep_tasks([task], jobs=1)[0]
            LOGGER.info(
                "Completed sweep task %d (%s | %s | %s). Metrics stored at %s.",
                outcome.order_index,
                task.study.key,
                task.study.issue,
                task.config.label(),
                outcome.metrics_path,
            )
        else:
            opinion_index = task_id - slate_total
            task = opinion_pending_by_index.get(opinion_index)
            if task is None:
                cached_outcome = opinion_cached_by_index.get(opinion_index)
                if cached_outcome is not None:
                    LOGGER.info(
                        "Skipping opinion sweep task %d (%s | %s); metrics already exist at %s.",
                        opinion_index,
                        cached_outcome.study.key,
                        cached_outcome.config.label(),
                        cached_outcome.metrics_path,
                    )
                    return
                LOGGER.warning(
                    "No opinion sweep task registered for index %d; skipping.",
                    opinion_index,
                )
                return
            if reuse_sweeps and task.metrics_path.exists():
                LOGGER.info(
                    "Skipping opinion sweep task %d (%s); metrics already exist at %s.",
                    task.index,
                    _format_opinion_sweep_task_descriptor(task),
                    task.metrics_path,
                )
                return
            outcome = _execute_opinion_sweep_tasks([task], jobs=1)[0]
            LOGGER.info(
                "Completed opinion sweep task %d (%s | %s). Metrics stored at %s.",
                outcome.order_index,
                task.study.key,
                task.config.label(),
                outcome.metrics_path,
            )
        return

    reuse_for_stage = reuse_sweeps
    if stage in {"finalize", "reports"}:
        reuse_for_stage = True
    pending_slate_tasks: List[SweepTask] = []
    cached_slate_outcomes: List[SweepOutcome] = []
    if run_next_video and sweep_context is not None:
        pending_slate_tasks, cached_slate_outcomes = _prepare_sweep_tasks(
            studies=study_specs,
            configs=configs,
            context=sweep_context,
            reuse_existing=reuse_for_stage,
        )

    pending_opinion_tasks: List[OpinionSweepTask] = []
    cached_opinion_outcomes: List[OpinionSweepOutcome] = []
    if run_opinion and opinion_sweep_context is not None:
        pending_opinion_tasks, cached_opinion_outcomes = _prepare_opinion_sweep_tasks(
            studies=study_specs,
            configs=configs,
            context=opinion_sweep_context,
            reuse_existing=reuse_for_stage,
        )
    if run_next_video and stage in {"finalize", "reports"} and pending_slate_tasks:
        missing = ", ".join(
            _format_sweep_task_descriptor(task) for task in pending_slate_tasks[:5]
        )
        more = ""
        if len(pending_slate_tasks) > 5:
            more = f", ... ({len(pending_slate_tasks)} total)"
        message = (
            "Sweep metrics missing for the following tasks: "
            f"{missing}{more}. Run --stage=sweeps to populate them."
        )
        if allow_incomplete:
            LOGGER.warning("%s Continuing because allow-incomplete mode is enabled.", message)
        else:
            raise RuntimeError(message)
    if run_opinion and stage in {"finalize", "reports"} and pending_opinion_tasks:
        missing = ", ".join(
            _format_opinion_sweep_task_descriptor(task)
            for task in pending_opinion_tasks[:5]
        )
        more = ""
        if len(pending_opinion_tasks) > 5:
            more = f", ... ({len(pending_opinion_tasks)} total)"
        message = (
            "Opinion sweep metrics missing for the following tasks: "
            f"{missing}{more}. Run --stage=sweeps to populate them."
        )
        if allow_incomplete:
            LOGGER.warning("%s Continuing because allow-incomplete mode is enabled.", message)
        else:
            raise RuntimeError(message)

    executed_slate_outcomes: List[SweepOutcome] = []
    executed_opinion_outcomes: List[OpinionSweepOutcome] = []
    if stage == "full":
        if run_next_video:
            executed_slate_outcomes = _execute_sweep_tasks(pending_slate_tasks, jobs=jobs)
        if run_opinion:
            executed_opinion_outcomes = _execute_opinion_sweep_tasks(
                pending_opinion_tasks,
                jobs=jobs,
            )

    outcomes: List[SweepOutcome] = []
    if run_next_video:
        outcomes = _merge_sweep_outcomes(cached_slate_outcomes, executed_slate_outcomes)
        if not outcomes:
            if allow_incomplete:
                warning = (
                    "No sweep outcomes available; reports will contain placeholders "
                    "because allow-incomplete mode is enabled."
                )
                LOGGER.warning(warning)
            else:
                raise RuntimeError("No sweep outcomes available; ensure sweeps have completed.")

    opinion_sweep_outcomes: List[OpinionSweepOutcome] = []
    if run_opinion:
        opinion_sweep_outcomes = _merge_opinion_sweep_outcomes(
            cached_opinion_outcomes,
            executed_opinion_outcomes,
        )
        if not opinion_sweep_outcomes:
            if allow_incomplete:
                warning = (
                    "No opinion sweep outcomes available; opinion reports will contain "
                    "placeholders because allow-incomplete mode is enabled."
                )
                LOGGER.warning(warning)
            else:
                raise RuntimeError(
                    "No opinion sweep outcomes available; ensure opinion "
                    "sweeps have completed."
                )

    selections: Dict[str, StudySelection] = {}
    if run_next_video:
        selections = _select_best_configs(outcomes)
        if not selections:
            if allow_incomplete:
                warning = (
                    "Failed to select configurations for any study; downstream "
                    "reports will rely on placeholders."
                )
                LOGGER.warning(warning)
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

    opinion_selections: Dict[str, OpinionStudySelection] = {}
    if run_opinion:
        opinion_selections = _select_best_opinion_configs(opinion_sweep_outcomes)
        if not opinion_selections:
            if allow_incomplete:
                warning = (
                    "Failed to select opinion configurations for any study; "
                    "downstream opinion metrics will rely on placeholders."
                )
                LOGGER.warning(warning)
            else:
                raise RuntimeError("Failed to select an opinion configuration for any study.")
        else:
            LOGGER.info(
                "Selected opinion configurations: %s",
                ", ".join(
                    f"{selection.study.key} ({selection.study.issue})"
                    for selection in opinion_selections.values()
                ),
            )

    final_eval_context: FinalEvalContext | None = None
    if run_next_video:
        final_eval_context = FinalEvalContext(
            base_cli=base_cli,
            extra_cli=extra_cli,
            out_dir=next_video_dir,
            tree_method=args.tree_method,
            save_model_dir=Path(args.save_model_dir) if args.save_model_dir else None,
            reuse_existing=reuse_final,
        )

    if stage == "reports":
        final_metrics: Dict[str, Mapping[str, object]] = {}
        if run_next_video and final_eval_context is not None:
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
                    LOGGER.warning(
                        "%s Continuing because allow-incomplete mode is enabled.",
                        message,
                    )
                else:
                    raise RuntimeError(message)
        opinion_metrics: Dict[str, Dict[str, object]] = {}
        if run_opinion:
            opinion_metrics = _load_opinion_metrics_from_disk(
                opinion_dir=opinion_dir,
                studies=study_specs,
            )
        sweep_report = SweepReportData(
            outcomes=outcomes,
            selections=selections,
            final_metrics=final_metrics,
        )
        opinion_report = (
            OpinionReportData(
                metrics=opinion_metrics,
                outcomes=opinion_sweep_outcomes,
                selections=opinion_selections,
            )
            if run_opinion
            else None
        )
        _write_reports(
            reports_dir=reports_dir,
            sweeps=sweep_report,
            allow_incomplete=allow_incomplete,
            include_next_video=run_next_video,
            opinion=opinion_report,
        )
        return

    final_metrics: Dict[str, Mapping[str, object]] = {}
    if run_next_video and final_eval_context is not None:
        final_metrics = _run_final_evaluations(selections=selections, context=final_eval_context)

    opinion_metrics: Dict[str, Dict[str, object]] = {}
    if run_opinion:
        opinion_stage_config = OpinionStageConfig(
            dataset=dataset,
            cache_dir=cache_dir,
            base_out_dir=opinion_dir,
            extra_fields=extra_fields,
            studies=study_tokens,
            max_participants=args.opinion_max_participants,
            seed=args.seed,
            max_features=args.max_features if args.max_features > 0 else None,
            tree_method=args.tree_method,
            overwrite=args.overwrite or not reuse_final,
            reuse_existing=reuse_final,
        )
        opinion_metrics = _run_opinion_stage(
            selections=opinion_selections,
            config=opinion_stage_config,
        )

    if stage == "finalize":
        return

    sweep_report = SweepReportData(
        outcomes=outcomes,
        selections=selections,
        final_metrics=final_metrics,
    )
    opinion_report = (
        OpinionReportData(
            metrics=opinion_metrics,
            outcomes=opinion_sweep_outcomes,
            selections=opinion_selections,
        )
        if run_opinion
        else None
    )
    _write_reports(
        reports_dir=reports_dir,
        sweeps=sweep_report,
        allow_incomplete=allow_incomplete,
        include_next_video=run_next_video,
        opinion=opinion_report,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
