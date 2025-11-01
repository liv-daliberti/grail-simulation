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

"""Stage assembly and orchestration helpers for the XGB pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple, cast
import importlib as _importlib

from common.pipeline.stage import (
    SweepPartitionExecutors,
    dispatch_cli_partitions,
    emit_stage_dry_run_summary,
    build_standard_sweeps_partitions,
)

from .context import (
    FinalEvalContext,
    OpinionStageConfig,
    OpinionStudySelection,
    OpinionSweepOutcome,
    OpinionSweepRunContext,
    OpinionSweepTask,
    OpinionDataSettings,
    OpinionVectorizerSettings,
    OpinionXgbSettings,
    StudySelection,
    SweepOutcome,
    SweepRunContext,
    SweepTask,
)
from .reports import (
    OpinionReportData,
    ReportSections,
    SweepReportData,
    _write_reports,
)
from .sweeps import (
    _merge_opinion_sweep_outcomes,
    _merge_sweep_outcomes,
    _format_opinion_sweep_task_descriptor,
    _format_sweep_task_descriptor,
    _prepare_opinion_sweep_tasks,
    _prepare_sweep_tasks,
    _execute_sweep_tasks,
    _execute_opinion_sweep_tasks,
    _select_best_configs,
    _select_best_opinion_configs,
)

LOGGER = logging.getLogger("xgb.pipeline")


def _pipeline_attr(name: str):
    """Resolve attribute from top-level xgb.pipeline at runtime for test hooks."""
    return getattr(_importlib.import_module("xgb.pipeline"), name)

def _build_sweep_context(
    *,
    run_next_video: bool,
    args,
    paths,
    jobs: int,
    base_cli: List[str],
    extra_cli: Sequence[str],
) -> SweepRunContext | None:
    """Construct and initialise the next-video sweep context when enabled."""

    if not run_next_video:
        return None
    sweep_dir = paths.sweep_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)
    return SweepRunContext(
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=sweep_dir,
        tree_method=args.tree_method,
        jobs=jobs,
    )


def _build_opinion_sweep_context(
    *,
    run_opinion: bool,
    args,
    paths,
    extra_fields_tuple: tuple[str, ...],
    max_features_value: int | None,
    tfidf_config,
    word2vec_config,
    sentence_transformer_config,
    word2vec_model_base: Path | None,
) -> OpinionSweepRunContext | None:
    """Construct and initialise the opinion sweep context when enabled."""

    if not run_opinion:
        return None
    ctx = OpinionSweepRunContext(
        sweep_dir=paths.opinion_sweep_dir,
        data=OpinionDataSettings(
            dataset=paths.dataset,
            cache_dir=paths.cache_dir,
            extra_fields=extra_fields_tuple,
            max_participants=args.opinion_max_participants,
            seed=args.seed,
            max_features=max_features_value,
        ),
        vectorizers=OpinionVectorizerSettings(
            tfidf_config=tfidf_config,
            word2vec_config=word2vec_config,
            sentence_transformer_config=sentence_transformer_config,
            word2vec_model_base=word2vec_model_base,
        ),
        xgb=OpinionXgbSettings(
            tree_method=args.tree_method,
            overwrite=args.overwrite,
        ),
    )
    ctx.sweep_dir.mkdir(parents=True, exist_ok=True)
    return ctx


def _build_opinion_stage_config(
    *,
    enabled: bool,
    args,
    paths,
    extra_fields_tuple: tuple[str, ...],
    study_tokens_tuple: tuple[str, ...],
    max_features_value: int | None,
    tfidf_config,
    word2vec_config,
    sentence_transformer_config,
    word2vec_model_base: Path | None,
    reuse_final: bool,
) -> OpinionStageConfig | None:
    """Construct the opinion stage config used by finalize/reports stages."""

    if not enabled:
        return None
    return OpinionStageConfig(
        dataset=paths.dataset,
        cache_dir=paths.cache_dir,
        base_out_dir=paths.opinion_dir,
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


def build_contexts(
    *,
    run_next_video: bool,
    run_opinion: bool,
    args,
    paths,
    jobs: int,
    base_cli: List[str],
    extra_cli: Sequence[str],
    max_features_value: int | None,
    tfidf_config,
    word2vec_config,
    sentence_transformer_config,
    word2vec_model_base: Path | None,
    reuse_final: bool,
    extra_fields_tuple: tuple[str, ...],
    study_tokens_tuple: tuple[str, ...],
) -> tuple[SweepRunContext | None, OpinionSweepRunContext | None, OpinionStageConfig | None]:
    """Build sweep and stage contexts for the requested pipelines."""

    return (
        _build_sweep_context(
            run_next_video=run_next_video,
            args=args,
            paths=paths,
            jobs=jobs,
            base_cli=base_cli,
            extra_cli=extra_cli,
        ),
        _build_opinion_sweep_context(
            run_opinion=run_opinion,
            args=args,
            paths=paths,
            extra_fields_tuple=extra_fields_tuple,
            max_features_value=max_features_value,
            tfidf_config=tfidf_config,
            word2vec_config=word2vec_config,
            sentence_transformer_config=sentence_transformer_config,
            word2vec_model_base=word2vec_model_base,
        ),
        _build_opinion_stage_config(
            enabled=run_opinion or run_next_video,
            args=args,
            paths=paths,
            extra_fields_tuple=extra_fields_tuple,
            study_tokens_tuple=study_tokens_tuple,
            max_features_value=max_features_value,
            tfidf_config=tfidf_config,
            word2vec_config=word2vec_config,
            sentence_transformer_config=sentence_transformer_config,
            word2vec_model_base=word2vec_model_base,
            reuse_final=reuse_final,
        ),
    )


def handle_plan_or_dry_run(
    *,
    stage: str,
    args,
    run_next_video: bool,
    run_opinion: bool,
    planned_slate_tasks: Sequence[SweepTask],
    cached_slate_planned: Sequence[SweepOutcome],
    planned_opinion_tasks: Sequence[OpinionSweepTask],
    cached_opinion_planned: Sequence[OpinionSweepOutcome],
) -> bool:
    """Emit plan or dry-run summaries and stop further execution when requested."""

    if stage == "plan":
        summary_bits: List[str] = []
        if run_next_video:
            summary_bits.append(
                f"next-video sweeps={len(planned_slate_tasks)} (cached={len(cached_slate_planned)})"
            )
        if run_opinion:
            summary_bits.append(
                f"opinion sweeps={len(planned_opinion_tasks)} "
                f"(cached={len(cached_opinion_planned)})"
            )
        LOGGER.info("Planned sweep tasks: %s.", "; ".join(summary_bits))
        _pipeline_attr("_emit_combined_sweep_plan")(  # type: ignore[operator]
            slate_tasks=planned_slate_tasks,
            opinion_tasks=planned_opinion_tasks,
        )
        return True

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
        return True

    return False


def handle_sweeps_stage(
    *,
    stage: str,
    args,
    run_next_video: bool,
    run_opinion: bool,
    planned_slate_tasks: Sequence[SweepTask],
    cached_slate_planned: Sequence[SweepOutcome],
    planned_opinion_tasks: Sequence[OpinionSweepTask],
    cached_opinion_planned: Sequence[OpinionSweepOutcome],
    prepare,
) -> bool:
    """Execute the sweeps-only stage using CLI-driven partitioning."""

    if stage != "sweeps":
        return False

    reuse_cached_metrics = True

    def _describe_slate_cached(outcome):
        return f"{outcome.study.key}:{outcome.study.issue}:{outcome.config.label()}"

    def _execute_opinion_task(task):
        return _execute_opinion_sweep_tasks([task], jobs=1)[0]

    # Resolve executors dynamically via the top-level package to allow tests
    # to monkeypatch xgb.pipeline._execute_* hooks reliably.
    _exec_sweeps = cast(
        Callable[[Sequence[SweepTask]], Sequence[SweepOutcome]],
        _pipeline_attr("_execute_sweep_tasks"),
    )
    _exec_opinion_sweeps = cast(
        Callable[[Sequence[OpinionSweepTask]], Sequence[OpinionSweepOutcome]],
        _pipeline_attr("_execute_opinion_sweep_tasks"),
    )

    partitions = build_standard_sweeps_partitions(
        include_next=run_next_video,
        next_label="next-video",
        next_pending=planned_slate_tasks,
        next_cached=cached_slate_planned,
        next_executors=SweepPartitionExecutors(
            execute_task=lambda task: _exec_sweeps([task], jobs=1)[0],
            describe_pending=_format_sweep_task_descriptor,
            describe_cached=_describe_slate_cached,
        ),
        include_opinion=run_opinion,
        opinion_pending=planned_opinion_tasks,
        opinion_cached=cached_opinion_planned,
        opinion_executors=SweepPartitionExecutors(
            execute_task=lambda task: _exec_opinion_sweeps([task], jobs=1)[0],
            describe_pending=_format_opinion_sweep_task_descriptor,
            describe_cached=lambda o: f"{o.study.key}:{o.study.issue}:{o.config.label()}",
        ),
        reuse_existing=reuse_cached_metrics,
    )
    dispatch_cli_partitions(
        partitions,
        args=args,
        logger=LOGGER,
        prepare=prepare,
    )
    return True


def prepare_pending_and_cached(
    *,
    stage: str,
    run_next_video: bool,
    run_opinion: bool,
    sweep_context: SweepRunContext | None,
    opinion_sweep_context: OpinionSweepRunContext | None,
    study_specs,
    configs,
    reuse_sweeps: bool,
) -> tuple[
    List[SweepTask],
    List[SweepOutcome],
    List[OpinionSweepTask],
    List[OpinionSweepOutcome],
]:
    """Prepare stage-specific pending tasks and cached outcomes for both pipelines."""

    reuse_for_stage = reuse_sweeps
    if stage in {"finalize", "reports"}:
        reuse_for_stage = True

    pending_slate_tasks: List[SweepTask] = []
    cached_slate_outcomes: List[SweepOutcome] = []
    if run_next_video and sweep_context is not None:
        prep_sweeps = cast(
            Callable[..., Tuple[List[SweepTask], List[SweepOutcome]]],
            _prepare_sweep_tasks,
        )
        pending_slate_tasks, cached_slate_outcomes = prep_sweeps(
            studies=study_specs,
            configs=configs,
            context=sweep_context,
            reuse_existing=reuse_for_stage,
        )

    pending_opinion_tasks: List[OpinionSweepTask] = []
    cached_opinion_outcomes: List[OpinionSweepOutcome] = []
    if run_opinion and opinion_sweep_context is not None:
        prep_opinion = cast(
            Callable[..., Tuple[List[OpinionSweepTask], List[OpinionSweepOutcome]]],
            _prepare_opinion_sweep_tasks,
        )
        pending_opinion_tasks, cached_opinion_outcomes = prep_opinion(
            studies=study_specs,
            configs=configs,
            context=opinion_sweep_context,
            reuse_existing=reuse_for_stage,
        )

    return (
        pending_slate_tasks,
        cached_slate_outcomes,
        pending_opinion_tasks,
        cached_opinion_outcomes,
    )


def validate_missing_metrics(
    *,
    stage: str,
    allow_incomplete: bool,
    run_next_video: bool,
    run_opinion: bool,
    pending_slate_tasks: Sequence[SweepTask],
    pending_opinion_tasks: Sequence[OpinionSweepTask],
) -> None:
    """Ensure downstream stages have the metrics they need to proceed."""

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


def maybe_execute_full_stage(
    *,
    stage: str,
    run_next_video: bool,
    run_opinion: bool,
    pending_slate_tasks: Sequence[SweepTask],
    pending_opinion_tasks: Sequence[OpinionSweepTask],
    jobs: int,
) -> tuple[List[SweepOutcome], List[OpinionSweepOutcome]]:
    """Execute sweeps for the "full" stage when requested by the caller."""

    executed_slate_outcomes: List[SweepOutcome] = []
    executed_opinion_outcomes: List[OpinionSweepOutcome] = []
    if stage == "full":
        if run_next_video:
            exec_sweeps = cast(
                Callable[..., List[SweepOutcome]],
                _execute_sweep_tasks,
            )
            executed_slate_outcomes = exec_sweeps(
                pending_slate_tasks,
                jobs=jobs,
            )
        if run_opinion:
            exec_opinion = cast(
                Callable[..., List[OpinionSweepOutcome]],
                _execute_opinion_sweep_tasks,
            )
            executed_opinion_outcomes = exec_opinion(
                pending_opinion_tasks,
                jobs=jobs,
            )
    return executed_slate_outcomes, executed_opinion_outcomes


@dataclass(frozen=True)
class _ReportInputs:
    final_metrics: Dict[str, Mapping[str, object]]
    loso_metrics: Dict[str, Mapping[str, object]]
    opinion_metrics: Dict[str, Dict[str, object]]
    opinion_from_next_metrics: Dict[str, Dict[str, object]]


def _collect_report_inputs(
    *,
    run_next_video: bool,
    run_opinion: bool,
    allow_incomplete: bool,
    final_eval_context: FinalEvalContext | None,
    opinion_stage_config: OpinionStageConfig | None,
    study_specs,
    paths,
) -> _ReportInputs:
    """Load or collect all inputs needed to render report sections."""

    final_metrics: Dict[str, Mapping[str, object]] = {}
    if run_next_video and final_eval_context is not None:
        final_metrics = _pipeline_attr("_load_final_metrics_from_disk")(  # type: ignore[operator]
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
        loso_metrics = _pipeline_attr("_load_loso_metrics_from_disk")(  # type: ignore[operator]
            next_video_dir=final_eval_context.out_dir,
            studies=study_specs,
        )

    opinion_metrics: Dict[str, Dict[str, object]] = {}
    if run_opinion:
        load_opinion_metrics = _pipeline_attr(
            "_load_opinion_metrics_from_disk"
        )  # type: ignore[operator]
        opinion_metrics = load_opinion_metrics(
            opinion_dir=paths.opinion_dir,
            studies=study_specs,
        )

    opinion_from_next_metrics: Dict[str, Dict[str, object]] = {}
    if run_next_video and opinion_stage_config is not None:
        opinion_from_next_metrics = _pipeline_attr(
            "_load_opinion_from_next_metrics_from_disk"
        )(  # type: ignore[operator]
            opinion_dir=paths.opinion_dir,
            studies=study_specs,
        )

    return _ReportInputs(
        final_metrics=final_metrics,
        loso_metrics=loso_metrics,
        opinion_metrics=opinion_metrics,
        opinion_from_next_metrics=opinion_from_next_metrics,
    )


def _write_reports_from_inputs(
    *,
    run_next_video: bool,
    run_opinion: bool,
    allow_incomplete: bool,
    paths,
    outcomes: Sequence[SweepOutcome],
    opinion_sweep_outcomes: Sequence[OpinionSweepOutcome],
    selections: Mapping[str, StudySelection],
    opinion_selections: Mapping[str, OpinionStudySelection],
    inputs: _ReportInputs,
) -> None:
    """Assemble report payloads and write them to disk."""

    sweep_report = SweepReportData(
        outcomes=outcomes,
        selections=selections,
        final_metrics=inputs.final_metrics,
        loso_metrics=inputs.loso_metrics,
    )
    opinion_report = (
        OpinionReportData(
            metrics=inputs.opinion_metrics,
            outcomes=opinion_sweep_outcomes,
            selections=opinion_selections,
            predictions_root=paths.opinion_dir,
        )
        if run_opinion
        else None
    )
    opinion_from_next_report = None
    if run_next_video and (inputs.opinion_from_next_metrics or allow_incomplete):
        description_lines = [
            "This section reuses the selected next-video configuration to ",
            "estimate post-study opinion change.",
        ]
        opinion_from_next_report = OpinionReportData(
            metrics=inputs.opinion_from_next_metrics,
            title="XGBoost Opinion Regression (Next-Video Config)",
            description_lines=description_lines,
            predictions_root=paths.opinion_dir / "from_next",
        )
    _write_reports(  # type: ignore[operator]
        reports_dir=paths.reports_dir,
        sweeps=sweep_report,
        allow_incomplete=allow_incomplete,
        sections=ReportSections(
            include_next_video=run_next_video,
            opinion=opinion_report,
            opinion_from_next=opinion_from_next_report,
        ),
    )


def merge_and_validate_outcomes(
    *,
    run_next_video: bool,
    run_opinion: bool,
    allow_incomplete: bool,
    cached_slate_outcomes: Sequence[SweepOutcome],
    executed_slate_outcomes: Sequence[SweepOutcome],
    cached_opinion_outcomes: Sequence[OpinionSweepOutcome],
    executed_opinion_outcomes: Sequence[OpinionSweepOutcome],
) -> tuple[List[SweepOutcome], List[OpinionSweepOutcome]]:
    """Merge cached and executed outcomes and check completeness."""

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
        opinion_sweep_outcomes = _merge_opinion_sweep_outcomes(  # type: ignore[operator]
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
                    "No opinion sweep outcomes available; ensure opinion sweeps have completed."
                )
    return outcomes, opinion_sweep_outcomes


def select_and_log_configs(
    *,
    run_next_video: bool,
    run_opinion: bool,
    allow_incomplete: bool,
    outcomes: Sequence[SweepOutcome],
    opinion_sweep_outcomes: Sequence[OpinionSweepOutcome],
) -> tuple[Dict[str, StudySelection], Dict[str, OpinionStudySelection]]:
    """Choose the best configurations per-study and log selections."""

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
    return selections, opinion_selections


def maybe_reports_stage(
    *,
    stage: str,
    run_next_video: bool,
    run_opinion: bool,
    allow_incomplete: bool,
    final_eval_context: FinalEvalContext | None,
    opinion_stage_config: OpinionStageConfig | None,
    selections: Mapping[str, StudySelection],
    opinion_selections: Mapping[str, OpinionStudySelection],
    study_specs,
    paths,
    outcomes: Sequence[SweepOutcome],
    opinion_sweep_outcomes: Sequence[OpinionSweepOutcome],
) -> bool:
    """Generate reports when requested and return True when handled."""

    if stage != "reports":
        return False

    inputs = _collect_report_inputs(
        run_next_video=run_next_video,
        run_opinion=run_opinion,
        allow_incomplete=allow_incomplete,
        final_eval_context=final_eval_context,
        opinion_stage_config=opinion_stage_config,
        study_specs=study_specs,
        paths=paths,
    )
    _write_reports_from_inputs(
        run_next_video=run_next_video,
        run_opinion=run_opinion,
        allow_incomplete=allow_incomplete,
        paths=paths,
        outcomes=outcomes,
        opinion_sweep_outcomes=opinion_sweep_outcomes,
        selections=selections,
        opinion_selections=opinion_selections,
        inputs=inputs,
    )
    return True


def finalize_and_report(
    *,
    run_next_video: bool,
    run_opinion: bool,
    allow_incomplete: bool,
    final_eval_context: FinalEvalContext | None,
    opinion_stage_config: OpinionStageConfig | None,
    selections: Mapping[str, StudySelection],
    opinion_selections: Mapping[str, OpinionStudySelection],
    outcomes: Sequence[SweepOutcome],
    opinion_sweep_outcomes: Sequence[OpinionSweepOutcome],
    paths,
    study_specs,
) -> None:
    """Run finalize evaluations and emit the Markdown reports."""

    @dataclass(frozen=True)
    class _FinalizedResults:
        final_metrics: Dict[str, Mapping[str, object]]
        loso_metrics: Dict[str, Mapping[str, object]]
        opinion_metrics: Dict[str, Dict[str, object]]
        opinion_from_next_metrics: Dict[str, Dict[str, object]]

    def _execute_final_evaluations() -> _FinalizedResults:
        final_metrics: Dict[str, Mapping[str, object]] = {}
        loso_metrics: Dict[str, Mapping[str, object]] = {}
        if run_next_video and final_eval_context is not None:
            final_metrics = _pipeline_attr("_run_final_evaluations")(  # type: ignore[operator]
                selections=selections,
                studies=study_specs,
                context=final_eval_context,
            )
            loso_metrics = {}
            if not allow_incomplete:
                run_cross = _pipeline_attr(
                    "_run_cross_study_evaluations"
                )  # type: ignore[operator]
                loso_metrics = run_cross(
                    selections=selections,
                    studies=study_specs,
                    context=final_eval_context,
                )

        opinion_metrics: Dict[str, Dict[str, object]] = {}
        if run_opinion and opinion_stage_config is not None:
            opinion_metrics = _pipeline_attr("_run_opinion_stage")(  # type: ignore[operator]
                selections=opinion_selections,
                config=opinion_stage_config,
            )

        opinion_from_next_metrics: Dict[str, Dict[str, object]] = {}
        if run_next_video and opinion_stage_config is not None:
            run_opinion_from_next = _pipeline_attr(
                "_run_opinion_from_next_stage"
            )  # type: ignore[operator]
            opinion_from_next_metrics = run_opinion_from_next(
                selections=selections,
                studies=study_specs,
                config=opinion_stage_config,
                allow_incomplete=allow_incomplete,
            )
        return _FinalizedResults(
            final_metrics=final_metrics,
            loso_metrics=loso_metrics,
            opinion_metrics=opinion_metrics,
            opinion_from_next_metrics=opinion_from_next_metrics,
        )

    def _write_reports_for_finalized(results: _FinalizedResults) -> None:
        sweep_report = SweepReportData(
            outcomes=outcomes,
            selections=selections,
            final_metrics=results.final_metrics,
            loso_metrics=results.loso_metrics,
        )
        opinion_report = (
            OpinionReportData(
                metrics=results.opinion_metrics,
                outcomes=opinion_sweep_outcomes,
                selections=opinion_selections,
            )
            if run_opinion
            else None
        )
        opinion_from_next_report = None
        if run_next_video and results.opinion_from_next_metrics:
            description_lines = [
                "This section reuses the selected next-video configuration to ",
                "estimate post-study opinion change.",
            ]
            opinion_from_next_report = OpinionReportData(
                metrics=results.opinion_from_next_metrics,
                title="XGBoost Opinion Regression (Next-Video Config)",
                description_lines=description_lines,
            )
        _pipeline_attr("_write_reports")(  # type: ignore[operator]
            reports_dir=paths.reports_dir,
            sweeps=sweep_report,
            allow_incomplete=allow_incomplete,
            sections=ReportSections(
                include_next_video=run_next_video,
                opinion=opinion_report,
                opinion_from_next=opinion_from_next_report,
            ),
        )

    _write_reports_for_finalized(_execute_final_evaluations())


__all__ = [
    "build_contexts",
    "handle_plan_or_dry_run",
    "handle_sweeps_stage",
    "prepare_pending_and_cached",
    "validate_missing_metrics",
    "maybe_execute_full_stage",
    "merge_and_validate_outcomes",
    "select_and_log_configs",
    "maybe_reports_stage",
    "finalize_and_report",
]
