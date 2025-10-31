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

"""Pipeline orchestration for the Grail Simulation KNN baselines.

The module coordinates hyper-parameter sweeps, final evaluations, and
Markdown report generation for the slate-ranking and opinion-regression
KNN workflows used throughout the project."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace as _dc_replace
from typing import Dict, List, Mapping, Sequence, TYPE_CHECKING

from common.pipeline.stage import (
    SweepPartitionExecutors,
    dispatch_cli_partitions,
    emit_stage_dry_run_summary,
    build_standard_sweeps_partitions,
    prepare_sweep_execution as _prepare_sweep_execution,
)

from .context import (
    EvaluationContext,
    PipelineContext,
    ReportBundle,
    ReportSelections,
    ReportOutcomes,
    ReportMetrics,
    ReportPresentation,
    PresentationFlags,
    PredictionRoots,
    SweepTaskContext,
)
from .cli import (
    build_base_cli as _build_base_cli,
    build_pipeline_context as _build_pipeline_context,
    log_dry_run as _log_dry_run,
    log_run_configuration as _log_run_configuration,
    parse_args as _parse_args,
    repo_root as _repo_root,
)
from .data import (
    resolve_studies as _resolve_studies,
    warn_if_issue_tokens_used as _warn_if_issue_tokens_used,
)
from .io import (
    load_final_metrics_from_disk as _load_final_metrics_from_disk,
    load_loso_metrics_from_disk as _load_loso_metrics_from_disk,
    load_opinion_metrics as _load_opinion_metrics,
)
from .sweeps import (
    build_sweep_configs as _build_sweep_configs,
    emit_combined_sweep_plan as _emit_combined_sweep_plan,
    execute_opinion_sweep_tasks as _execute_opinion_sweep_tasks,
    execute_sweep_task as _execute_sweep_task,
    execute_sweep_tasks as _execute_sweep_tasks,
    format_opinion_sweep_task_descriptor as _format_opinion_sweep_task_descriptor,
    format_sweep_task_descriptor as _format_sweep_task_descriptor,
    merge_opinion_sweep_outcomes as _merge_opinion_sweep_outcomes,
    merge_sweep_outcomes as _merge_sweep_outcomes,
    prepare_opinion_sweep_tasks as _prepare_opinion_sweep_tasks,
    prepare_sweep_tasks as _prepare_sweep_tasks,
    select_best_configs as _select_best_configs,
    select_best_opinion_configs as _select_best_opinion_configs,
)
from .evaluate import (
    run_cross_study_evaluations as _run_cross_study_evaluations,
    run_final_evaluations as _run_final_evaluations,
    run_opinion_evaluations as _run_opinion_evaluations,
    run_opinion_from_next_evaluations as _run_opinion_from_next_evaluations,
)
from .reports import generate_reports as _generate_reports

if TYPE_CHECKING:
    from .context import (
        OpinionStudySelection,
        OpinionSweepOutcome,
        OpinionSweepTask,
        StudySelection,
        SweepOutcome,
        SweepTask,
    )

LOGGER = logging.getLogger(__name__)

__all__ = [
    "PipelineContext",
    "ReportBundle",
    "main",
    "_build_sweep_configs",
]

prepare_sweep_execution = _prepare_sweep_execution


@dataclass(frozen=True)
class _RunEnv:
    """Container for inputs shared across pipeline stages."""

    args: object
    extra_cli: Sequence[str]
    context: PipelineContext
    studies: Sequence[object]
    stage: str
    sweep_context: SweepTaskContext
    opinion_sweep_context: SweepTaskContext


@dataclass(frozen=True)
class _PreparedSweeps:
    """Planned sweep tasks and cached results prepared once per run."""

    planned_tasks: List["SweepTask"]
    cached_planned: List["SweepOutcome"]
    planned_opinion_tasks: List["OpinionSweepTask"]
    cached_planned_opinion: List["OpinionSweepOutcome"]


@dataclass(frozen=True)
class _StageTasks:
    """Pending tasks and cached outcomes tailored to the active stage."""

    pending_tasks: List["SweepTask"]
    cached_outcomes: List["SweepOutcome"]
    pending_opinion_tasks: List["OpinionSweepTask"]
    cached_opinion_outcomes: List["OpinionSweepOutcome"]


def _init_run(argv: Sequence[str] | None) -> _RunEnv:
    """Parse CLI, build context, and compute static inputs for this run."""

    args, extra_cli = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    root = _repo_root()
    context = _build_pipeline_context(args, root)
    _warn_if_issue_tokens_used(args)

    studies = _resolve_studies(context.study_tokens)
    if not studies:
        raise RuntimeError("No studies available for evaluation.")

    _log_run_configuration(studies, context)

    stage = getattr(args, "stage", "full")
    _align_sentence_transformer_context(context, stage=stage)

    sweep_context = SweepTaskContext(
        base_cli=_build_base_cli(context, extra_cli),
        extra_cli=extra_cli,
        sweep_dir=context.sweep_dir,
        word2vec_model_base=context.word2vec_model_dir,
    )
    opinion_sweep_context = SweepTaskContext(
        base_cli=_build_base_cli(context, extra_cli),
        extra_cli=extra_cli,
        sweep_dir=context.opinion_sweep_dir,
        word2vec_model_base=context.opinion_word2vec_dir,
    )

    return _RunEnv(
        args=args,
        extra_cli=extra_cli,
        context=context,
        studies=studies,
        stage=stage,
        sweep_context=sweep_context,
        opinion_sweep_context=opinion_sweep_context,
    )


def _prepare_initial(env: _RunEnv) -> _PreparedSweeps:
    """Compute the sweep plan once so it can be reused across stages."""

    planned_tasks: List[SweepTask] = []
    cached_planned: List[SweepOutcome] = []
    if env.context.run_next_video:
        planned_tasks, cached_planned = _prepare_sweep_tasks(
            studies=env.studies,
            configs=_build_sweep_configs(env.context),
            context=env.sweep_context,
            reuse_existing=env.context.reuse_sweeps,
        )

    planned_opinion_tasks: List[OpinionSweepTask] = []
    cached_planned_opinion: List[OpinionSweepOutcome] = []
    if env.context.run_opinion:
        planned_opinion_tasks, cached_planned_opinion = _prepare_opinion_sweep_tasks(
            studies=env.studies,
            configs=_build_sweep_configs(env.context),
            context=env.opinion_sweep_context,
            reuse_existing=env.context.reuse_sweeps,
        )

    return _PreparedSweeps(
        planned_tasks=planned_tasks,
        cached_planned=cached_planned,
        planned_opinion_tasks=planned_opinion_tasks,
        cached_planned_opinion=cached_planned_opinion,
    )


def _emit_plan(env: _RunEnv, prepared: _PreparedSweeps) -> None:
    """Emit a compact plan summary for both pipelines and exit."""

    _log_dry_run(_build_sweep_configs(env.context))
    summary_bits: List[str] = []
    if env.context.run_next_video:
        summary_bits.append(
            (
                "next-video sweeps="
                f"{len(prepared.planned_tasks)} "
                f"(cached={len(prepared.cached_planned)})"
            )
        )
    if env.context.run_opinion:
        summary_bits.append(
            (
                f"opinion sweeps={len(prepared.planned_opinion_tasks)} "
                f"(cached={len(prepared.cached_planned_opinion)})"
            )
        )
    LOGGER.info(
        "Planned sweep tasks: %s.",
        "; ".join(summary_bits) if summary_bits else "no tasks selected",
    )
    _emit_combined_sweep_plan(
        slate_tasks=prepared.planned_tasks,
        opinion_tasks=prepared.planned_opinion_tasks,
    )


def _emit_dry_run(env: _RunEnv, prepared: _PreparedSweeps) -> None:
    """Emit a stage-focused dry-run summary and exit."""

    _log_dry_run(_build_sweep_configs(env.context))
    emit_stage_dry_run_summary(
        LOGGER,
        include_next=env.context.run_next_video,
        next_label="next-video",
        next_pending=len(prepared.planned_tasks),
        next_cached=len(prepared.cached_planned),
        include_opinion=env.context.run_opinion,
        opinion_pending=len(prepared.planned_opinion_tasks),
        opinion_cached=len(prepared.cached_planned_opinion),
    )


def _run_sweeps(env: _RunEnv, prepared: _PreparedSweeps) -> None:
    """Execute pending sweeps while skipping cached metrics."""

    reuse_cached_metrics = True
    partitions = build_standard_sweeps_partitions(
        include_next=env.context.run_next_video,
        next_label="next-video",
        next_pending=prepared.planned_tasks,
        next_cached=prepared.cached_planned,
        next_executors=SweepPartitionExecutors(
            execute_task=_execute_sweep_task,
            describe_pending=_format_sweep_task_descriptor,
            describe_cached=_describe_sweep_outcome,
        ),
        include_opinion=env.context.run_opinion,
        opinion_pending=prepared.planned_opinion_tasks,
        opinion_cached=prepared.cached_planned_opinion,
        opinion_executors=SweepPartitionExecutors(
            execute_task=_execute_opinion_sweep_task,
            describe_pending=_format_opinion_sweep_task_descriptor,
            describe_cached=_describe_opinion_sweep_outcome,
        ),
        reuse_existing=reuse_cached_metrics,
        opinion_prefix="[OPINION]",
    )
    dispatch_cli_partitions(
        partitions,
        args=env.args,
        logger=LOGGER,
        prepare=prepare_sweep_execution,
    )


def _prepare_for_stage(env: _RunEnv, *, reuse_existing: bool) -> _StageTasks:
    """Prepare stage-specific pending tasks and cached outcomes."""

    pending_tasks: List[SweepTask] = []
    cached_outcomes: List[SweepOutcome] = []
    if env.context.run_next_video:
        pending_tasks, cached_outcomes = _prepare_sweep_tasks(
            studies=env.studies,
            configs=_build_sweep_configs(env.context),
            context=env.sweep_context,
            reuse_existing=reuse_existing,
        )

    pending_opinion_tasks: List[OpinionSweepTask] = []
    cached_opinion_outcomes: List[OpinionSweepOutcome] = []
    if env.context.run_opinion:
        pending_opinion_tasks, cached_opinion_outcomes = _prepare_opinion_sweep_tasks(
            studies=env.studies,
            configs=_build_sweep_configs(env.context),
            context=env.opinion_sweep_context,
            reuse_existing=reuse_existing,
        )

    return _StageTasks(
        pending_tasks=pending_tasks,
        cached_outcomes=cached_outcomes,
        pending_opinion_tasks=pending_opinion_tasks,
        cached_opinion_outcomes=cached_opinion_outcomes,
    )


def _ensure_required_metrics(env: _RunEnv, stage_tasks: _StageTasks) -> None:
    """Validate metrics availability for finalize/report stages, raising or warning."""

    if (
        env.context.run_next_video
        and env.stage in {"finalize", "reports"}
        and stage_tasks.pending_tasks
    ):
        missing = ", ".join(
            _format_sweep_task_descriptor(task) for task in stage_tasks.pending_tasks[:5]
        )
        count_pending = len(stage_tasks.pending_tasks)
        more = "" if count_pending <= 5 else f", … ({count_pending} total)"
        base_message = (
            "Slate sweep metrics missing for the following tasks: "
            f"{missing}{more}."
        )
        if env.context.allow_incomplete:
            LOGGER.warning(
                "%s Repair: run --stage=sweeps to populate them. "
                "Continuing with available metrics because allow-incomplete mode is enabled.",
                base_message,
            )
        else:
            raise RuntimeError(f"{base_message} Run --stage=sweeps to populate them.")

    if (
        env.context.run_opinion
        and env.stage in {"finalize", "reports"}
        and stage_tasks.pending_opinion_tasks
    ):
        missing = ", ".join(
            _format_opinion_sweep_task_descriptor(task)
            for task in stage_tasks.pending_opinion_tasks[:5]
        )
        more = (
            ""
            if len(stage_tasks.pending_opinion_tasks) <= 5
            else f", … ({len(stage_tasks.pending_opinion_tasks)} total)"
        )
        base_message = (
            "Opinion sweep metrics missing for the following tasks: " f"{missing}{more}."
        )
        if env.context.allow_incomplete:
            LOGGER.warning(
                "%s Repair: run --stage=sweeps to populate them. "
                "Continuing with available metrics because allow-incomplete mode is enabled.",
                base_message,
            )
        else:
            raise RuntimeError(f"{base_message} Run --stage=sweeps to populate them.")


def _maybe_execute_full(
    env: _RunEnv, stage_tasks: _StageTasks
) -> tuple[List["SweepOutcome"], List["OpinionSweepOutcome"]]:
    """Execute pending tasks in 'full' stage and return outcomes."""

    executed_outcomes: List[SweepOutcome] = []
    executed_opinion_outcomes: List[OpinionSweepOutcome] = []
    if env.stage == "full":
        if env.context.run_next_video:
            executed_outcomes = _execute_sweep_tasks(
                stage_tasks.pending_tasks,
                jobs=env.context.jobs,
            )
        if env.context.run_opinion:
            executed_opinion_outcomes = _execute_opinion_sweep_tasks(
                stage_tasks.pending_opinion_tasks,
                jobs=env.context.jobs,
            )
    return executed_outcomes, executed_opinion_outcomes


def _merge_and_select(
    env: _RunEnv,
    *,
    cached_outcomes: List["SweepOutcome"],
    executed_outcomes: List["SweepOutcome"],
    cached_opinion_outcomes: List["OpinionSweepOutcome"],
    executed_opinion_outcomes: List["OpinionSweepOutcome"],
) -> tuple[
    List["SweepOutcome"],
    Dict[str, Dict[str, "StudySelection"]],
    List["OpinionSweepOutcome"],
    Dict[str, Dict[str, "OpinionStudySelection"]],
]:
    """Merge outcomes and compute study selections for both pipelines."""

    sweep_outcomes: List[SweepOutcome] = []
    selections: Dict[str, Dict[str, StudySelection]] = {}
    if env.context.run_next_video:
        sweep_outcomes = _merge_sweep_outcomes(cached_outcomes, executed_outcomes)
        if not sweep_outcomes:
            if env.context.allow_incomplete:
                LOGGER.warning(
                    (
                        "No sweep outcomes available for next-video; continuing because "
                        "allow-incomplete mode is enabled."
                    )
                )
            else:
                raise RuntimeError(
                    "No sweep outcomes available for next-video; ensure sweeps have completed."
                )
        else:
            selections = _select_best_configs(
                outcomes=sweep_outcomes,
                studies=env.studies,
                allow_incomplete=env.context.allow_incomplete,
            )

    opinion_sweep_outcomes: List[OpinionSweepOutcome] = []
    opinion_selections: Dict[str, Dict[str, OpinionStudySelection]] = {}
    if env.context.run_opinion:
        opinion_sweep_outcomes = _merge_opinion_sweep_outcomes(
            cached_opinion_outcomes,
            executed_opinion_outcomes,
        )
        if not opinion_sweep_outcomes:
            if env.context.allow_incomplete:
                LOGGER.warning(
                    (
                        "No sweep outcomes available for opinion regression; continuing "
                        "because allow-incomplete mode is enabled."
                    )
                )
            else:
                raise RuntimeError(
                    "No opinion sweep outcomes available; ensure opinion sweeps have completed."
                )
        else:
            opinion_selections = _select_best_opinion_configs(
                outcomes=opinion_sweep_outcomes,
                studies=env.studies,
                allow_incomplete=env.context.allow_incomplete,
            )

    return sweep_outcomes, selections, opinion_sweep_outcomes, opinion_selections


def _reports_from_disk(
    env: _RunEnv,
    *,
    sweep_outcomes: List["SweepOutcome"],
    selections: Dict[str, Dict[str, "StudySelection"]],
    opinion_sweep_outcomes: List["OpinionSweepOutcome"],
    opinion_selections: Dict[str, Dict[str, "OpinionStudySelection"]],
) -> None:
    """Load metrics snapshots from disk and generate reports."""

    slate_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    loso_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    if env.context.run_next_video:
        slate_metrics = _load_final_metrics_from_disk(
            out_dir=env.context.next_video_dir,
            feature_spaces=env.context.feature_spaces,
            studies=env.studies,
        )
        if not slate_metrics:
            message = (
                f"No slate metrics found under {env.context.next_video_dir}. "
                "Run --stage=finalize before generating reports."
            )
            if env.context.allow_incomplete:
                LOGGER.warning("%s Continuing because allow-incomplete mode is enabled.", message)
            else:
                raise RuntimeError(message)
        loso_metrics = _load_loso_metrics_from_disk(
            out_dir=env.context.next_video_dir,
            feature_spaces=env.context.feature_spaces,
            studies=env.studies,
        )
    opinion_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    if env.context.run_opinion:
        for feature_space in env.context.feature_spaces:
            metrics = _load_opinion_metrics(env.context.opinion_dir, feature_space)
            if metrics:
                opinion_metrics[feature_space] = metrics
    opinion_from_next_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    if env.context.run_next_video:
        base_dir = env.context.opinion_dir / "from_next"
        for feature_space in env.context.feature_spaces:
            metrics = _load_opinion_metrics(base_dir, feature_space)
            if metrics:
                opinion_from_next_metrics[feature_space] = metrics
    report_bundle = ReportBundle(
        selections=ReportSelections(
            selections=selections,
            opinion_selections=opinion_selections,
        ),
        outcomes=ReportOutcomes(
            sweep_outcomes=sweep_outcomes,
            opinion_sweep_outcomes=opinion_sweep_outcomes,
        ),
        metrics=ReportMetrics(
            metrics_by_feature=slate_metrics,
            opinion_metrics=opinion_metrics,
            opinion_from_next_metrics=opinion_from_next_metrics,
            loso_metrics=loso_metrics,
        ),
        presentation=ReportPresentation(
            feature_spaces=env.context.feature_spaces,
            sentence_model=(
                env.context.sentence_model
                if "sentence_transformer" in env.context.feature_spaces
                else None
            ),
            k_sweep=env.context.k_sweep,
            studies=env.studies,
            flags=PresentationFlags(
                allow_incomplete=env.context.allow_incomplete,
                include_next_video=env.context.run_next_video,
                include_opinion=env.context.run_opinion,
                include_opinion_from_next=(
                    env.context.run_next_video and bool(opinion_from_next_metrics)
                ),
            ),
            predictions=PredictionRoots(
                opinion_predictions_root=env.context.opinion_dir,
                opinion_from_next_predictions_root=env.context.opinion_dir / "from_next",
            ),
        ),
    )
    _generate_reports(_repo_root(), report_bundle)


def _finalize_and_report(
    env: _RunEnv,
    *,
    sweep_outcomes: List["SweepOutcome"],
    selections: Dict[str, Dict[str, "StudySelection"]],
    opinion_sweep_outcomes: List["OpinionSweepOutcome"],
    opinion_selections: Dict[str, Dict[str, "OpinionStudySelection"]],
) -> None:
    """Run final evaluations, collect metrics, and generate reports."""

    eval_context = EvaluationContext.from_args(
        base_cli=_build_base_cli(env.context, env.extra_cli),
        extra_cli=env.extra_cli,
        next_video_out_dir=env.context.next_video_dir,
        opinion_out_dir=env.context.opinion_dir,
        next_video_word2vec_dir=env.context.word2vec_model_dir,
        opinion_word2vec_dir=env.context.opinion_word2vec_dir,
        reuse_existing=env.context.reuse_final,
    )

    slate_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    if env.context.run_next_video:
        slate_metrics = _run_final_evaluations(
            selections=selections,
            studies=env.studies,
            context=eval_context,
        )

    opinion_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    if env.context.run_opinion:
        opinion_metrics = _run_opinion_evaluations(
            selections=opinion_selections,
            studies=env.studies,
            context=eval_context,
        )

    opinion_from_next_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    if env.context.run_next_video:
        try:
            opinion_from_next_metrics = _run_opinion_from_next_evaluations(
                selections=selections,
                studies=env.studies,
                context=eval_context,
            )
        except FileNotFoundError as exc:
            dataset_hint = getattr(exc, "filename", None) or (exc.args[0] if exc.args else None)
            if env.context.allow_incomplete:
                LOGGER.warning(
                    "Opinion-from-next evaluation skipped; dataset not found at %s. "
                    "Continuing because allow-incomplete mode is enabled.",
                    dataset_hint or env.context.dataset,
                )
            else:
                raise

    loso_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    if env.context.run_next_video:
        loso_metrics = _run_cross_study_evaluations(
            selections=selections,
            studies=env.studies,
            context=eval_context,
        )

    report_bundle = ReportBundle(
        selections=ReportSelections(
            selections=selections,
            opinion_selections=opinion_selections,
        ),
        outcomes=ReportOutcomes(
            sweep_outcomes=sweep_outcomes,
            opinion_sweep_outcomes=opinion_sweep_outcomes,
        ),
        metrics=ReportMetrics(
            metrics_by_feature=slate_metrics,
            opinion_metrics=opinion_metrics,
            opinion_from_next_metrics=opinion_from_next_metrics,
            loso_metrics=loso_metrics,
        ),
        presentation=ReportPresentation(
            feature_spaces=env.context.feature_spaces,
            sentence_model=(
                env.context.sentence_model
                if "sentence_transformer" in env.context.feature_spaces
                else None
            ),
            k_sweep=env.context.k_sweep,
            studies=env.studies,
            flags=PresentationFlags(
                allow_incomplete=env.context.allow_incomplete,
                include_next_video=env.context.run_next_video,
                include_opinion=env.context.run_opinion,
                include_opinion_from_next=(
                    env.context.run_next_video and bool(opinion_from_next_metrics)
                ),
            ),
            predictions=PredictionRoots(
                opinion_predictions_root=env.context.opinion_dir,
                opinion_from_next_predictions_root=env.context.opinion_dir / "from_next",
            ),
        ),
    )
    _generate_reports(_repo_root(), report_bundle)

def _iter_sentence_transformer_config_dirs(base_dir):
    """Yield sentence-transformer configuration directories rooted at ``base_dir``."""

    if base_dir is None:
        return
    try:
        iterator = base_dir.rglob("sentence_transformer")
    except (OSError, RuntimeError):
        LOGGER.debug("Skipping sentence-transformer discovery under %s.", base_dir)
        return
    for st_root in iterator:
        if not st_root.is_dir():
            continue
        try:
            study_dirs = list(st_root.iterdir())
        except OSError:
            continue
        for study_dir in study_dirs:
            if not study_dir.is_dir():
                continue
            try:
                for config_dir in study_dir.iterdir():
                    if config_dir.is_dir():
                        yield config_dir
            except OSError:
                continue


def _parse_sentence_transformer_label(label: str) -> tuple[str | None, int | None, bool | None]:
    """Extract device, batch size, and normalization flags encoded in a config label."""

    device: str | None = None
    batch_size: int | None = None
    normalize: bool | None = None
    for token in label.split("_"):
        if token.startswith("device-"):
            device = token[len("device-") :]
        elif token.startswith("bs"):
            raw = token[2:]
            if raw.isdigit():
                batch_size = int(raw)
        elif token == "norm":
            normalize = True
        elif token == "nonorm":
            normalize = False
    return device, batch_size, normalize


def _discover_sentence_transformer_overrides(context: PipelineContext) -> dict[str, object]:
    """Infer cached sentence-transformer settings from existing sweep artefacts."""

    overrides: dict[str, object] = {}
    roots = [context.sweep_dir, context.opinion_sweep_dir]
    for root in roots:
        if root is None:
            continue
        for config_dir in _iter_sentence_transformer_config_dirs(root):
            device, batch_size, normalize = _parse_sentence_transformer_label(
                config_dir.name.lower()
            )
            if device and "device" not in overrides:
                overrides["device"] = device
            if batch_size is not None and "batch_size" not in overrides:
                overrides["batch_size"] = batch_size
            if normalize is not None and "normalize" not in overrides:
                overrides["normalize"] = normalize
            if len(overrides) == 3:
                return overrides
    return overrides


def _align_sentence_transformer_context(
    context: PipelineContext,
    *,
    stage: str,
) -> None:
    """Adjust the sentence-transformer configuration to reuse cached artefacts."""

    if "sentence_transformer" not in context.feature_spaces:
        return
    consider_reuse = stage in {"finalize", "reports"} or context.reuse_sweeps or context.reuse_final
    if not consider_reuse:
        return
    overrides = _discover_sentence_transformer_overrides(context)

    # Build a new model-defaults bundle and swap it atomically to avoid
    # mutating a frozen dataclass via attribute assignment.
    updates: dict[str, object] = {}

    device = overrides.get("device")
    if device and context.sentence_device != device:
        LOGGER.info(
            "Detected cached sentence-transformer device '%s'; overriding configuration.",
            device,
        )
        updates["sentence_device"] = device

    batch_size = overrides.get("batch_size")
    if batch_size is not None and context.sentence_batch_size != batch_size:
        LOGGER.info(
            "Detected cached sentence-transformer batch size %d; overriding configuration.",
            batch_size,
        )
        updates["sentence_batch_size"] = int(batch_size)

    if "normalize" in overrides and context.sentence_normalize != overrides["normalize"]:
        normalize = bool(overrides["normalize"])
        LOGGER.info(
            "Detected cached sentence-transformer normalization=%s; overriding configuration.",
            "enabled" if normalize else "disabled",
        )
        updates["sentence_normalize"] = normalize

    if updates:
        # context._models is a frozen dataclass; replace returns a new instance.
        new_models = _dc_replace(context._models, **updates)
        # Bypass frozen guard intentionally; this mirrors the property setters.
        object.__setattr__(context, "_models", new_models)


def _describe_sweep_outcome(outcome: "SweepOutcome") -> str:
    """Compose a human-readable descriptor for cached KNN sweep outcomes."""
    return f"{outcome.feature_space}:{outcome.study.key}:{outcome.config.label()}"


def _describe_opinion_sweep_outcome(outcome: "OpinionSweepOutcome") -> str:
    """Compose a human-readable descriptor for cached opinion sweep outcomes."""
    return f"{outcome.feature_space}:{outcome.study.key}:{outcome.config.label()}"


def _execute_opinion_sweep_task(task: "OpinionSweepTask") -> "OpinionSweepOutcome":
    """Execute a single opinion sweep task via the batch helper."""
    return _execute_opinion_sweep_tasks([task], jobs=1)[0]


def main(argv: Sequence[str] | None = None) -> None:
    """
    Coordinate sweeps, evaluations, and report generation for the KNN pipeline.

    :param argv: Optional argument vector override used when invoking the CLI programmatically.
    :type argv: Sequence[str] | None
    :returns: None.
    :rtype: None
    """

    env = _init_run(argv)
    prepared = _prepare_initial(env)

    if env.stage == "plan":
        _emit_plan(env, prepared)
        return

    if getattr(env.args, 'dry_run', False):
        _emit_dry_run(env, prepared)
        return

    if env.stage == "sweeps":
        _run_sweeps(env, prepared)
        return

    reuse_for_stage = env.context.reuse_sweeps or env.stage in {"finalize", "reports"}
    stage_tasks = _prepare_for_stage(env, reuse_existing=reuse_for_stage)
    _ensure_required_metrics(env, stage_tasks)

    executed_outcomes, executed_opinion_outcomes = _maybe_execute_full(env, stage_tasks)
    sweep_outcomes, selections, opinion_sweep_outcomes, opinion_selections = _merge_and_select(
        env,
        cached_outcomes=stage_tasks.cached_outcomes,
        executed_outcomes=executed_outcomes,
        cached_opinion_outcomes=stage_tasks.cached_opinion_outcomes,
        executed_opinion_outcomes=executed_opinion_outcomes,
    )

    if env.stage == "reports":
        _reports_from_disk(
            env,
            sweep_outcomes=sweep_outcomes,
            selections=selections,
            opinion_sweep_outcomes=opinion_sweep_outcomes,
            opinion_selections=opinion_selections,
        )
        return

    _finalize_and_report(
        env,
        sweep_outcomes=sweep_outcomes,
        selections=selections,
        opinion_sweep_outcomes=opinion_sweep_outcomes,
        opinion_selections=opinion_selections,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
