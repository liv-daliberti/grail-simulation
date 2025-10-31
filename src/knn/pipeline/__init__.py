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
    device = overrides.get("device")
    if device and context.sentence_device != device:
        LOGGER.info(
            "Detected cached sentence-transformer device '%s'; overriding configuration.",
            device,
        )
        object.__setattr__(context, "sentence_device", device)
    batch_size = overrides.get("batch_size")
    if batch_size is not None and context.sentence_batch_size != batch_size:
        LOGGER.info(
            "Detected cached sentence-transformer batch size %d; overriding configuration.",
            batch_size,
        )
        object.__setattr__(context, "sentence_batch_size", batch_size)
    if "normalize" in overrides and context.sentence_normalize != overrides["normalize"]:
        normalize = bool(overrides["normalize"])
        LOGGER.info(
            "Detected cached sentence-transformer normalization=%s; overriding configuration.",
            "enabled" if normalize else "disabled",
        )
        object.__setattr__(context, "sentence_normalize", normalize)


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
    # pylint: disable=too-many-branches,too-many-locals,too-many-return-statements,too-many-statements
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

    base_cli = _build_base_cli(context, extra_cli)
    configs = _build_sweep_configs(context)

    sweep_context = SweepTaskContext(
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=context.sweep_dir,
        word2vec_model_base=context.word2vec_model_dir,
    )
    opinion_sweep_context = SweepTaskContext(
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=context.opinion_sweep_dir,
        word2vec_model_base=context.opinion_word2vec_dir,
    )

    # Enumerate the full sweep plan once so it can be reused across stages.
    planned_tasks: List[SweepTask] = []
    cached_planned: List[SweepOutcome] = []
    if context.run_next_video:
        planned_tasks, cached_planned = _prepare_sweep_tasks(
            studies=studies,
            configs=configs,
            context=sweep_context,
            reuse_existing=context.reuse_sweeps,
        )
    planned_opinion_tasks: List[OpinionSweepTask] = []
    cached_planned_opinion: List[OpinionSweepOutcome] = []
    if context.run_opinion:
        planned_opinion_tasks, cached_planned_opinion = _prepare_opinion_sweep_tasks(
            studies=studies,
            configs=configs,
            context=opinion_sweep_context,
            reuse_existing=context.reuse_sweeps,
        )

    if stage == "plan":
        _log_dry_run(configs)
        summary_bits: List[str] = []
        if context.run_next_video:
            summary_bits.append(
                f"next-video sweeps={len(planned_tasks)} (cached={len(cached_planned)})"
            )
        if context.run_opinion:
            summary_bits.append(
                (
                    f"opinion sweeps={len(planned_opinion_tasks)} "
                    f"(cached={len(cached_planned_opinion)})"
                )
            )
        LOGGER.info(
            "Planned sweep tasks: %s.",
            "; ".join(summary_bits) if summary_bits else "no tasks selected",
        )
        _emit_combined_sweep_plan(
            slate_tasks=planned_tasks,
            opinion_tasks=planned_opinion_tasks,
        )
        return

    if args.dry_run:
        _log_dry_run(configs)
        emit_stage_dry_run_summary(
            LOGGER,
            include_next=context.run_next_video,
            next_label="next-video",
            next_pending=len(planned_tasks),
            next_cached=len(cached_planned),
            include_opinion=context.run_opinion,
            opinion_pending=len(planned_opinion_tasks),
            opinion_cached=len(cached_planned_opinion),
        )
        return

    if stage == "sweeps":
        # Always skip execution for tasks that already produced metrics. The sweeps stage
        # is intended to fill gaps rather than rerun the entire grid, so cached artefacts
        # short-circuit regardless of the CLI reuse flag. To force recomputation, clear
        # the cached metrics before invoking the stage.
        reuse_cached_metrics = True
        partitions = build_standard_sweeps_partitions(
            include_next=context.run_next_video,
            next_label="next-video",
            next_pending=planned_tasks,
            next_cached=cached_planned,
            next_executors=SweepPartitionExecutors(
                execute_task=_execute_sweep_task,
                describe_pending=_format_sweep_task_descriptor,
                describe_cached=_describe_sweep_outcome,
            ),
            include_opinion=context.run_opinion,
            opinion_pending=planned_opinion_tasks,
            opinion_cached=cached_planned_opinion,
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
            args=args,
            logger=LOGGER,
            prepare=prepare_sweep_execution,
        )
        return

    reuse_for_stage = context.reuse_sweeps
    if stage in {"finalize", "reports"}:
        reuse_for_stage = True

    pending_tasks: List[SweepTask] = []
    cached_outcomes: List[SweepOutcome] = []
    if context.run_next_video:
        pending_tasks, cached_outcomes = _prepare_sweep_tasks(
            studies=studies,
            configs=configs,
            context=sweep_context,
            reuse_existing=reuse_for_stage,
        )
    pending_opinion_tasks = []
    cached_opinion_outcomes: List[OpinionSweepOutcome] = []
    if context.run_opinion:
        pending_opinion_tasks, cached_opinion_outcomes = _prepare_opinion_sweep_tasks(
            studies=studies,
            configs=configs,
            context=opinion_sweep_context,
            reuse_existing=reuse_for_stage,
        )

    if context.run_next_video and stage in {"finalize", "reports"} and pending_tasks:
        missing = ", ".join(_format_sweep_task_descriptor(task) for task in pending_tasks[:5])
        more = "" if len(pending_tasks) <= 5 else f", … ({len(pending_tasks)} total)"
        base_message = (
            "Sweep metrics missing for the following tasks: "
            f"{missing}{more}."
        )
        if context.allow_incomplete:
            LOGGER.warning(
                "%s Continuing with available metrics because allow-incomplete mode is enabled.",
                base_message,
            )
        else:
            raise RuntimeError(f"{base_message} Run --stage=sweeps to populate them.")
    if context.run_opinion and stage in {"finalize", "reports"} and pending_opinion_tasks:
        missing = ", ".join(
            _format_opinion_sweep_task_descriptor(task) for task in pending_opinion_tasks[:5]
        )
        more = (
            "" if len(pending_opinion_tasks) <= 5 else f", … ({len(pending_opinion_tasks)} total)"
        )
        base_message = (
            "Opinion sweep metrics missing for the following tasks: "
            f"{missing}{more}."
        )
        if context.allow_incomplete:
            LOGGER.warning(
                "%s Continuing with available metrics because allow-incomplete mode is enabled.",
                base_message,
            )
        else:
            raise RuntimeError(f"{base_message} Run --stage=sweeps to populate them.")

    executed_outcomes: List[SweepOutcome] = []
    executed_opinion_outcomes: List[OpinionSweepOutcome] = []
    if stage == "full":
        if context.run_next_video:
            executed_outcomes = _execute_sweep_tasks(pending_tasks, jobs=context.jobs)
        if context.run_opinion:
            executed_opinion_outcomes = _execute_opinion_sweep_tasks(
                pending_opinion_tasks,
                jobs=context.jobs,
            )

    sweep_outcomes: List[SweepOutcome] = []
    selections: Dict[str, Dict[str, StudySelection]] = {}
    if context.run_next_video:
        sweep_outcomes = _merge_sweep_outcomes(cached_outcomes, executed_outcomes)
        if not sweep_outcomes:
            if context.allow_incomplete:
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
                studies=studies,
                allow_incomplete=context.allow_incomplete,
            )

    opinion_sweep_outcomes: List[OpinionSweepOutcome] = []
    opinion_selections: Dict[str, Dict[str, OpinionStudySelection]] = {}
    if context.run_opinion:
        opinion_sweep_outcomes = _merge_opinion_sweep_outcomes(
            cached_opinion_outcomes,
            executed_opinion_outcomes,
        )
        if not opinion_sweep_outcomes:
            if context.allow_incomplete:
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
                studies=studies,
                allow_incomplete=context.allow_incomplete,
            )

    if stage == "reports":
        slate_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
        loso_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
        if context.run_next_video:
            slate_metrics = _load_final_metrics_from_disk(
                out_dir=context.next_video_dir,
                feature_spaces=context.feature_spaces,
                studies=studies,
            )
            if not slate_metrics:
                message = (
                    f"No slate metrics found under {context.next_video_dir}. "
                    "Run --stage=finalize before generating reports."
                )
                if context.allow_incomplete:
                    LOGGER.warning(
                        "%s Continuing because allow-incomplete mode is enabled.",
                        message,
                    )
                else:
                    raise RuntimeError(message)
            loso_metrics = _load_loso_metrics_from_disk(
                out_dir=context.next_video_dir,
                feature_spaces=context.feature_spaces,
                studies=studies,
            )
        opinion_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
        if context.run_opinion:
            for feature_space in context.feature_spaces:
                metrics = _load_opinion_metrics(context.opinion_dir, feature_space)
                if metrics:
                    opinion_metrics[feature_space] = metrics
        opinion_from_next_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
        if context.run_next_video:
            base_dir = context.opinion_dir / "from_next"
            for feature_space in context.feature_spaces:
                metrics = _load_opinion_metrics(base_dir, feature_space)
                if metrics:
                    opinion_from_next_metrics[feature_space] = metrics
        report_bundle = ReportBundle(
            selections=selections,
            sweep_outcomes=sweep_outcomes,
            opinion_selections=opinion_selections,
            opinion_sweep_outcomes=opinion_sweep_outcomes,
            studies=studies,
            metrics_by_feature=slate_metrics,
            opinion_metrics=opinion_metrics,
            opinion_from_next_metrics=opinion_from_next_metrics,
            k_sweep=context.k_sweep,
            loso_metrics=loso_metrics,
            feature_spaces=context.feature_spaces,
            sentence_model=(
                context.sentence_model
                if "sentence_transformer" in context.feature_spaces
                else None
            ),
            allow_incomplete=context.allow_incomplete,
            include_next_video=context.run_next_video,
            include_opinion=context.run_opinion,
            include_opinion_from_next=context.run_next_video and bool(opinion_from_next_metrics),
            opinion_predictions_root=context.opinion_dir,
            opinion_from_next_predictions_root=context.opinion_dir / "from_next",
        )
        _generate_reports(root, report_bundle)
        return

    slate_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    eval_context = EvaluationContext.from_args(
        base_cli=base_cli,
        extra_cli=extra_cli,
        next_video_out_dir=context.next_video_dir,
        opinion_out_dir=context.opinion_dir,
        next_video_word2vec_dir=context.word2vec_model_dir,
        opinion_word2vec_dir=context.opinion_word2vec_dir,
        reuse_existing=context.reuse_final,
    )
    if context.run_next_video:
        slate_metrics = _run_final_evaluations(
            selections=selections,
            studies=studies,
            context=eval_context,
        )

    opinion_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    if context.run_opinion:
        opinion_metrics = _run_opinion_evaluations(
            selections=opinion_selections,
            studies=studies,
            context=eval_context,
        )

    opinion_from_next_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    if context.run_next_video:
        try:
            opinion_from_next_metrics = _run_opinion_from_next_evaluations(
                selections=selections,
                studies=studies,
                context=eval_context,
            )
        except FileNotFoundError as exc:
            dataset_hint = getattr(exc, "filename", None)
            if not dataset_hint and exc.args:
                dataset_hint = exc.args[0]
            if context.allow_incomplete:
                LOGGER.warning(
                    "Opinion-from-next evaluation skipped; dataset not found at %s. "
                    "Continuing because allow-incomplete mode is enabled.",
                    dataset_hint or context.dataset,
                )
            else:
                raise

    loso_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    if context.run_next_video:
        loso_metrics = _run_cross_study_evaluations(
            selections=selections,
            studies=studies,
            context=eval_context,
        )

    report_bundle = ReportBundle(
        selections=selections,
        sweep_outcomes=sweep_outcomes,
        opinion_selections=opinion_selections,
        opinion_sweep_outcomes=opinion_sweep_outcomes,
        studies=studies,
        metrics_by_feature=slate_metrics,
        opinion_metrics=opinion_metrics,
        opinion_from_next_metrics=opinion_from_next_metrics,
        k_sweep=context.k_sweep,
        loso_metrics=loso_metrics,
        feature_spaces=context.feature_spaces,
        sentence_model=(
            context.sentence_model
            if "sentence_transformer" in context.feature_spaces
            else None
        ),
        allow_incomplete=context.allow_incomplete,
        include_next_video=context.run_next_video,
        include_opinion=context.run_opinion,
        include_opinion_from_next=context.run_next_video and bool(opinion_from_next_metrics),
        opinion_predictions_root=context.opinion_dir,
        opinion_from_next_predictions_root=context.opinion_dir / "from_next",
    )
    _generate_reports(root, report_bundle)

if __name__ == "__main__":  # pragma: no cover
    main()
