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

from common.pipeline_stage import (
    SweepPartitionExecutors,
    dispatch_cli_partitions,
    emit_stage_dry_run_summary,
    build_standard_sweeps_partitions,
    prepare_sweep_execution as _prepare_sweep_execution,
)
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
    _load_loso_metrics_from_disk,
    _load_opinion_metrics_from_disk,
    _load_opinion_from_next_metrics_from_disk,
    _select_best_configs,
    _select_best_opinion_configs,
)
from .pipeline_evaluate import (
    _run_cross_study_evaluations,
    _run_final_evaluations,
    _run_opinion_from_next_stage,
    _run_opinion_stage,
)
from .vectorizers import (
    SentenceTransformerVectorizerConfig,
    TfidfConfig,
    Word2VecVectorizerConfig,
)
from .pipeline_reports import (
    OpinionReportData,
    ReportSections,
    SweepReportData,
    _write_reports,
)

LOGGER = logging.getLogger("xgb.pipeline")

__all__ = ["main", "SweepRunContext", "OpinionStageConfig", "OpinionSweepRunContext"]

prepare_sweep_execution = _prepare_sweep_execution



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
    study_tokens_tuple = tuple(study_tokens)
    study_specs = _resolve_study_specs(
        dataset=dataset,
        cache_dir=cache_dir,
        requested_issues=issue_tokens,
        requested_studies=study_tokens,
        allow_incomplete=allow_incomplete,
    )
    extra_fields = merge_default_extra_fields(_split_tokens(args.extra_text_fields))
    extra_fields_tuple = tuple(extra_fields)

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
    reuse_sweeps = bool(getattr(args, "reuse_sweeps", False))
    reuse_sweeps_source: str | None = "--reuse-sweeps" if reuse_sweeps else None
    reuse_sweeps_env = os.environ.get("XGB_REUSE_SWEEPS")
    if reuse_sweeps_env is not None:
        reuse_sweeps = reuse_sweeps_env.lower() not in {"0", "false", "no"}
        reuse_sweeps_source = "XGB_REUSE_SWEEPS"
    if reuse_sweeps:
        detail = f" ({reuse_sweeps_source})" if reuse_sweeps_source else ""
        LOGGER.warning(
            "Cached sweep metrics reuse enabled%s; stale artefacts will be used when present.",
            detail,
        )
    else:
        LOGGER.info("Cached sweep metrics reuse disabled; sweeps will recompute results.")
    reuse_final = reuse_sweeps
    reuse_final_source: str | None = "sweep reuse default" if reuse_final else None
    if args.reuse_final is not None:
        reuse_final = args.reuse_final
        reuse_final_source = "--reuse-final"
    reuse_final_env = os.environ.get("XGB_REUSE_FINAL")
    if reuse_final_env is not None:
        reuse_final = reuse_final_env.lower() not in {"0", "false", "no"}
        reuse_final_source = "XGB_REUSE_FINAL"
    if reuse_final:
        detail = f" ({reuse_final_source})" if reuse_final_source else ""
        LOGGER.warning(
            "Finalize-stage reuse enabled%s; cached evaluation artefacts may be consumed.",
            detail,
        )

    max_features_value = args.max_features if args.max_features > 0 else None
    tfidf_config = TfidfConfig(max_features=max_features_value)
    word2vec_model_base = (
        Path(args.word2vec_model_dir).resolve()
        if args.word2vec_model_dir
        else None
    )
    word2vec_config = Word2VecVectorizerConfig(
        vector_size=args.word2vec_size,
        window=args.word2vec_window,
        min_count=args.word2vec_min_count,
        epochs=args.word2vec_epochs,
        workers=args.word2vec_workers,
        seed=args.seed,
        model_dir=None,
    )
    sentence_transformer_config = SentenceTransformerVectorizerConfig(
        model_name=args.sentence_transformer_model,
        device=args.sentence_transformer_device or None,
        batch_size=args.sentence_transformer_batch_size,
        normalize=args.sentence_transformer_normalize,
    )

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
            extra_fields=extra_fields_tuple,
            max_participants=args.opinion_max_participants,
            seed=args.seed,
            max_features=max_features_value,
            tree_method=args.tree_method,
            overwrite=args.overwrite,
            tfidf_config=tfidf_config,
            word2vec_config=word2vec_config,
            sentence_transformer_config=sentence_transformer_config,
            word2vec_model_base=word2vec_model_base,
        )
        opinion_sweep_context.sweep_dir.mkdir(parents=True, exist_ok=True)

    opinion_stage_config: OpinionStageConfig | None = None
    if run_opinion or run_next_video:
        opinion_stage_config = OpinionStageConfig(
            dataset=dataset,
            cache_dir=cache_dir,
            base_out_dir=opinion_dir,
            extra_fields=extra_fields_tuple,
            studies=study_tokens_tuple,
            max_participants=args.opinion_max_participants,
            seed=args.seed,
            max_features=max_features_value,
            tree_method=args.tree_method,
            overwrite=args.overwrite or not reuse_final,
            tfidf_config=tfidf_config,
            word2vec_config=word2vec_config,
            sentence_transformer_config=sentence_transformer_config,
            word2vec_model_base=word2vec_model_base,
            reuse_existing=reuse_final,
        )

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
        emit_stage_dry_run_summary(
            LOGGER,
            include_next=run_next_video,
            next_label="next-video",
            next_pending=len(planned_slate_tasks),
            next_cached=len(cached_slate_planned),
            include_opinion=run_opinion,
            opinion_pending=len(planned_opinion_tasks),
            opinion_cached=len(cached_opinion_planned),
        )
        return

    if stage == "sweeps":
        # Skip execution for cached tasks so the sweeps stage only fills gaps. Removing
        # the cached artefacts before invocation forces recomputation when needed.
        reuse_cached_metrics = True

        def _describe_slate_cached(outcome):
            return f"{outcome.study.key}:{outcome.study.issue}:{outcome.config.label()}"

        def _execute_opinion_task(task):
            return _execute_opinion_sweep_tasks([task], jobs=1)[0]

        partitions = build_standard_sweeps_partitions(
            include_next=run_next_video,
            next_label="next-video",
            next_pending=planned_slate_tasks,
            next_cached=cached_slate_planned,
            next_executors=SweepPartitionExecutors(
                execute_task=lambda task: _execute_sweep_tasks([task], jobs=1)[0],
                describe_pending=_format_sweep_task_descriptor,
                describe_cached=_describe_slate_cached,
            ),
            include_opinion=run_opinion,
            opinion_pending=planned_opinion_tasks,
            opinion_cached=cached_opinion_planned,
            opinion_executors=SweepPartitionExecutors(
                execute_task=_execute_opinion_task,
                describe_pending=_format_opinion_sweep_task_descriptor,
                describe_cached=lambda o: f"{o.study.key}:{o.study.issue}:{o.config.label()}",
            ),
            reuse_existing=reuse_cached_metrics,
        )
        dispatch_cli_partitions(partitions, args=args, logger=LOGGER)
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
        loso_metrics: Dict[str, Mapping[str, object]] = {}
        if run_next_video and final_eval_context is not None:
            loso_metrics = _load_loso_metrics_from_disk(
                next_video_dir=final_eval_context.out_dir,
                studies=study_specs,
            )
        opinion_metrics: Dict[str, Dict[str, object]] = {}
        if run_opinion:
            opinion_metrics = _load_opinion_metrics_from_disk(
                opinion_dir=opinion_dir,
                studies=study_specs,
            )
        opinion_from_next_metrics: Dict[str, Dict[str, object]] = {}
        if run_next_video and opinion_stage_config is not None:
            opinion_from_next_metrics = _load_opinion_from_next_metrics_from_disk(
                opinion_dir=opinion_dir,
                studies=study_specs,
            )
        sweep_report = SweepReportData(
            outcomes=outcomes,
            selections=selections,
            final_metrics=final_metrics,
            loso_metrics=loso_metrics,
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
        opinion_from_next_report = None
        if run_next_video and (opinion_from_next_metrics or allow_incomplete):
            description_lines = [
                "This section reuses the selected next-video configuration to "
                "estimate post-study opinion change."
            ]
            opinion_from_next_report = OpinionReportData(
                metrics=opinion_from_next_metrics,
                title="XGBoost Opinion Regression (Next-Video Config)",
                description_lines=description_lines,
            )
        _write_reports(
            reports_dir=reports_dir,
            sweeps=sweep_report,
            allow_incomplete=allow_incomplete,
            sections=ReportSections(
                include_next_video=run_next_video,
                opinion=opinion_report,
                opinion_from_next=opinion_from_next_report,
            ),
        )
        return
    final_metrics: Dict[str, Mapping[str, object]] = {}
    loso_metrics: Dict[str, Mapping[str, object]] = {}
    if run_next_video and final_eval_context is not None:
        final_metrics = _run_final_evaluations(
            selections=selections,
            studies=study_specs,
            context=final_eval_context,
        )
        loso_metrics = _run_cross_study_evaluations(
            selections=selections,
            studies=study_specs,
            context=final_eval_context,
        )

    opinion_metrics: Dict[str, Dict[str, object]] = {}
    if run_opinion and opinion_stage_config is not None:
        opinion_metrics = _run_opinion_stage(
            selections=opinion_selections,
            config=opinion_stage_config,
        )

    opinion_from_next_metrics: Dict[str, Dict[str, object]] = {}
    if run_next_video and opinion_stage_config is not None:
        opinion_from_next_metrics = _run_opinion_from_next_stage(
            selections=selections,
            studies=study_specs,
            config=opinion_stage_config,
            allow_incomplete=allow_incomplete,
        )

    sweep_report = SweepReportData(
        outcomes=outcomes,
        selections=selections,
        final_metrics=final_metrics,
        loso_metrics=loso_metrics,
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
    opinion_from_next_report = None
    if run_next_video and opinion_from_next_metrics:
        description_lines = [
            "This section reuses the selected next-video configuration to "
            "estimate post-study opinion change."
        ]
        opinion_from_next_report = OpinionReportData(
            metrics=opinion_from_next_metrics,
            title="XGBoost Opinion Regression (Next-Video Config)",
            description_lines=description_lines,
        )
    _write_reports(
        reports_dir=reports_dir,
        sweeps=sweep_report,
        allow_incomplete=allow_incomplete,
        sections=ReportSections(
            include_next_video=run_next_video,
            opinion=opinion_report,
            opinion_from_next=opinion_from_next_report,
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
