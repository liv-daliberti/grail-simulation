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

"""Helpers for next-video sweep orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from common.pipeline.executor import execute_indexed_tasks
from common.pipeline.utils import make_placeholder_metrics, base_sweep_outcome_kwargs

from ..context import (
    StudySelection,
    StudySpec,
    SweepConfig,
    SweepOutcome,
    SweepRunContext,
    SweepTask,
)
from .common import (
    LOGGER,
    MissingMetricsLogConfig,
    build_merge_sweep_outcomes,
    get_sweeps_attr,
    load_metrics_with_placeholder,
)


def _sweep_outcome_from_metrics(
    task: SweepTask,
    metrics: Mapping[str, object],
    metrics_path: Path,
) -> SweepOutcome:
    """
    Convert cached sweep metrics into an outcome instance.

    :param task: Sweep task metadata describing the study/config pair.
    :type task: SweepTask
    :param metrics: Metrics payload loaded from ``metrics_path``.
    :type metrics: Mapping[str, object]
    :param metrics_path: Filesystem location of the metrics artefact.
    :type metrics_path: Path
    :returns: Sweep outcome populated with accuracy, coverage, and support.
    :rtype: ~xgb.pipeline.context.SweepOutcome
    """

    # Prefer eligible-only accuracy when available; fall back to overall.
    acc_value = metrics.get("accuracy_eligible")
    if acc_value is None:
        acc_value = metrics.get("accuracy")
    return SweepOutcome(**base_sweep_outcome_kwargs(task, metrics, metrics_path))


def _iter_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: SweepRunContext,
) -> Sequence[SweepTask]:
    """
    Yield sweep tasks with deterministic ordering.

    :param studies: Participant studies slated for evaluation.
    :type studies: Sequence[~common.pipeline.types.StudySpec]
    :param configs: Hyper-parameter configurations to explore.
    :type configs: Sequence[~xgb.pipeline.context.SweepConfig]
    :param context: Shared sweep execution context.
    :type context: SweepRunContext
    :returns: Iterable sequence of sweep tasks sorted by deterministic index.
    :rtype: Sequence[SweepTask]
    """

    base_cli_tuple = tuple(context.base_cli)
    extra_cli_tuple = tuple(context.extra_cli)
    # Restrict training to the same study: do not include alternates.
    tasks: List[SweepTask] = []
    task_index = 0
    for config in configs:
        for study in studies:
            run_root = context.sweep_dir / study.issue_slug / study.study_slug / config.label()
            metrics_path = run_root / study.evaluation_slug / "metrics.json"
            tasks.append(
                SweepTask(
                    index=task_index,
                    study=study,
                    config=config,
                    base_cli=tuple(
                        list(base_cli_tuple)
                        + [
                            "--xgb_tree_method",
                            context.tree_method,
                        ]
                    ),
                    extra_cli=extra_cli_tuple,
                    run_root=run_root,
                    tree_method=context.tree_method,
                    metrics_path=metrics_path,
                )
            )
            task_index += 1
    return tasks


def _prepare_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: SweepRunContext,
    reuse_existing: bool,
) -> Tuple[List[SweepTask], List[SweepOutcome]]:
    """
    Determine pending sweep tasks and return cached outcomes.

    :param studies: Participant studies slated for evaluation.
    :type studies: Sequence[~common.pipeline.types.StudySpec]
    :param configs: Hyper-parameter configurations to explore.
    :type configs: Sequence[~xgb.pipeline.context.SweepConfig]
    :param context: Shared sweep execution context.
    :type context: SweepRunContext
    :param reuse_existing: Flag controlling whether cached artefacts should be reused.
    :type reuse_existing: bool
    :returns: Tuple containing pending tasks and cached outcomes.
    :rtype: Tuple[List[SweepTask], List[~xgb.pipeline.context.SweepOutcome]]
    """

    pending: List[SweepTask] = []
    cached: List[SweepOutcome] = []
    load_metrics = cast(
        Callable[[Path], Mapping[str, object]],
        get_sweeps_attr("_load_metrics"),
    )
    outcome_factory = cast(
        Callable[[SweepTask, Mapping[str, object], Path], SweepOutcome],
        get_sweeps_attr("_sweep_outcome_from_metrics"),
    )
    for task in _iter_sweep_tasks(studies=studies, configs=configs, context=context):
        metrics_path = task.metrics_path
        if reuse_existing and metrics_path.exists():
            LOGGER.info(
                "[SWEEP][SKIP] issue=%s study=%s config=%s (cached).",
                task.study.issue,
                task.study.key,
                task.config.label(),
            )
            cached_metrics = load_metrics(metrics_path)
            cached.append(outcome_factory(task, cached_metrics, metrics_path))
            continue
        pending.append(task)
    return pending, cached


_merge_sweep_outcomes: Callable[
    [Sequence[SweepOutcome], Sequence[SweepOutcome]], List[SweepOutcome]
] = build_merge_sweep_outcomes(
    duplicate_message="Duplicate sweep outcome for index=%d; replacing cached result.",
    docstring=(
        "Combine cached and freshly executed next-video sweep outcomes while preserving "
        "order indices."
    ),
)


def _execute_sweep_tasks(
    tasks: Sequence[SweepTask],
    *,
    jobs: int,
) -> List[SweepOutcome]:
    """
    Execute sweep tasks, optionally in parallel.

    :param tasks: Sweep tasks to execute.
    :type tasks: Sequence[SweepTask]
    :param jobs: Maximum number of parallel workers.
    :type jobs: int
    :returns: Ordered list of sweep outcomes.
    :rtype: List[~xgb.pipeline.context.SweepOutcome]
    """

    return execute_indexed_tasks(
        tasks,
        _execute_sweep_task,
        jobs=jobs,
        logger=LOGGER,
        label="sweep",
    )


def _run_sweeps(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: SweepRunContext,
) -> List[SweepOutcome]:
    """
    Execute hyper-parameter sweeps and collect outcome metadata.

    :param studies: Participant studies slated for evaluation.
    :type studies: Sequence[~common.pipeline.types.StudySpec]
    :param configs: Hyper-parameter configurations to explore.
    :type configs: Sequence[~xgb.pipeline.context.SweepConfig]
    :param context: Shared sweep execution context.
    :type context: SweepRunContext
    :returns: Ordered list of combined cached and executed outcomes.
    :rtype: List[~xgb.pipeline.context.SweepOutcome]
    """

    pending_tasks, cached_outcomes = _prepare_sweep_tasks(
        studies=studies,
        configs=configs,
        context=context,
        reuse_existing=False,
    )
    executed_outcomes = _execute_sweep_tasks(pending_tasks, jobs=context.jobs)
    return _merge_sweep_outcomes(cached_outcomes, executed_outcomes)


def _execute_sweep_task(task: SweepTask) -> SweepOutcome:
    """
    Execute a single XGBoost sweep task and return the resulting metrics.

    :param task: Sweep task to execute.
    :type task: SweepTask
    :returns: Sweep outcome populated with metrics.
    :rtype: ~xgb.pipeline.context.SweepOutcome
    """

    run_root = task.run_root
    run_root.mkdir(parents=True, exist_ok=True)

    cli_args: List[str] = list(task.base_cli)
    cli_args.extend(task.config.cli_args(None))
    cli_args.extend(["--issues", task.study.issue])
    cli_args.extend(["--participant_studies", task.study.key])
    cli_args.extend(["--out_dir", str(run_root)])
    cli_args.extend(task.extra_cli)
    # Within-study training only for sweeps.
    LOGGER.info(
        "[SWEEP] issue=%s study=%s training restricted to within-study only.",
        task.study.issue,
        task.study.key,
    )

    evaluation_dir = task.metrics_path.parent
    has_existing_outputs = evaluation_dir.exists()
    missing_metrics = not task.metrics_path.exists()
    if has_existing_outputs and "--overwrite" not in cli_args:
        if missing_metrics:
            LOGGER.warning(
                "[SWEEP][RECOVER] issue=%s study=%s config=%s detected partial outputs at %s; "
                "automatically enabling overwrite for rerun.",
                task.study.issue,
                task.study.key,
                task.config.label(),
                evaluation_dir,
            )
        else:
            LOGGER.info(
                "[SWEEP][OVERWRITE] issue=%s study=%s config=%s existing outputs at %s; "
                "enabling overwrite to refresh metrics.",
                task.study.issue,
                task.study.key,
                task.config.label(),
                evaluation_dir,
            )
        cli_args.append("--overwrite")

    LOGGER.info(
        "[SWEEP] issue=%s study=%s config=%s",
        task.study.issue,
        task.study.key,
        task.config.label(),
    )
    run_xgb_cli = cast(
        Callable[[Sequence[str]], None],
        get_sweeps_attr("_run_xgb_cli"),
    )
    run_xgb_cli(cli_args)

    # Handle runs that were skipped by the evaluator (e.g., no train/eval rows)
    # by logging and producing a placeholder outcome instead of raising.
    load_metrics_with_log = cast(
        Callable[[Path, StudySpec, int, str], Optional[Dict[str, object]]],
        get_sweeps_attr("_load_metrics_with_log"),
    )
    metrics = load_metrics_with_placeholder(
        metrics_path=task.metrics_path,
        study=task.study,
        loader=load_metrics_with_log,
        placeholder_factory=lambda: make_placeholder_metrics(
            task.study.evaluation_slug,
            [task.study.key],
            extra_fields=[],
        ),
        log_config=MissingMetricsLogConfig(
            message=(
                "[SWEEP][MISS] issue=%s study=%s missing metrics at %s; "
                "recording placeholder outcome."
            ),
            debug_message="[SWEEP][MISS] Unable to write placeholder metrics at %s",
            logger=LOGGER,
        ),
    )
    return SweepOutcome(
        order_index=task.index,
        study=task.study,
        config=task.config,
        metrics_path=task.metrics_path,
        metrics=metrics,
    )


def _load_final_metrics_from_disk(
    *,
    next_video_dir: Path,
    studies: Sequence[StudySpec],
) -> Dict[str, Mapping[str, object]]:
    """
    Load persisted final evaluation metrics per study.

    :param next_video_dir: Directory containing final next-video artefacts.
    :type next_video_dir: Path
    :param studies: Participant studies to load.
    :type studies: Sequence[~common.pipeline.types.StudySpec]
    :returns: Mapping from study key to metrics payloads.
    :rtype: Dict[str, Mapping[str, object]]
    """

    metrics_by_study: Dict[str, Mapping[str, object]] = {}
    load_metrics = cast(
        Callable[[Path], Mapping[str, object]],
        get_sweeps_attr("_load_metrics"),
    )
    inject_metadata = cast(
        Callable[[Dict[str, object], StudySpec], None],
        get_sweeps_attr("_inject_study_metadata"),
    )
    for spec in studies:
        metrics_path = next_video_dir / spec.evaluation_slug / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = dict(load_metrics(metrics_path))
        inject_metadata(metrics, spec)
        metrics_by_study[spec.key] = metrics
    return metrics_by_study


def _load_loso_metrics_from_disk(
    *,
    next_video_dir: Path,
    studies: Sequence[StudySpec],
) -> Dict[str, Mapping[str, object]]:
    """
    Load leave-one-study-out evaluation metrics for each study.

    :param next_video_dir: Directory containing next-video evaluation artefacts.
    :type next_video_dir: Path
    :param studies: Participant studies to load.
    :type studies: Sequence[~common.pipeline.types.StudySpec]
    :returns: Mapping from study key to LOSO metrics payloads.
    :rtype: Dict[str, Mapping[str, object]]
    """

    metrics_by_study: Dict[str, Mapping[str, object]] = {}
    load_metrics_with_log = cast(
        Callable[..., Optional[Dict[str, object]]],
        get_sweeps_attr("_load_metrics_with_log"),
    )
    inject_metadata = cast(
        Callable[[Dict[str, object], StudySpec], None],
        get_sweeps_attr("_inject_study_metadata"),
    )
    loso_root = next_video_dir / "loso"
    if not loso_root.exists():
        return metrics_by_study
    for spec in studies:
        metrics_path = loso_root / spec.evaluation_slug / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_metrics_with_log(
            metrics_path,
            spec,
            log_level=logging.DEBUG,
            message="[LOSO][MISS] issue=%s study=%s expected metrics at %s but none found.",
        )
        if metrics is None:
            continue
        inject_metadata(metrics, spec)
        metrics_by_study[spec.key] = metrics
    return metrics_by_study


def _select_best_configs(outcomes: Sequence[SweepOutcome]) -> Dict[str, StudySelection]:
    """
    Pick the best configuration per study using accuracy, coverage, and support.

    :param outcomes: Sweep outcomes covering all studies and configurations.
    :type outcomes: Sequence[~xgb.pipeline.context.SweepOutcome]
    :returns: Mapping from study key to the chosen configuration.
    :rtype: Dict[str, StudySelection]
    """

    selections: Dict[str, StudySelection] = {}

    for outcome in outcomes:
        current = selections.get(outcome.study.key)
        if current is None:
            selections[outcome.study.key] = StudySelection(study=outcome.study, outcome=outcome)
            continue
        incumbent = current.outcome
        if outcome.accuracy > incumbent.accuracy + 1e-9:
            selections[outcome.study.key] = StudySelection(study=outcome.study, outcome=outcome)
            continue
        if incumbent.accuracy - outcome.accuracy <= 1e-9:
            if outcome.coverage > incumbent.coverage + 1e-9:
                selections[outcome.study.key] = StudySelection(
                    study=outcome.study,
                    outcome=outcome,
                )
            elif (
                abs(outcome.coverage - incumbent.coverage) <= 1e-9
                and outcome.evaluated > incumbent.evaluated
            ):
                selections[outcome.study.key] = StudySelection(
                    study=outcome.study,
                    outcome=outcome,
                )
    return selections


__all__ = [
    "_execute_sweep_task",
    "_execute_sweep_tasks",
    "_iter_sweep_tasks",
    "_load_final_metrics_from_disk",
    "_load_loso_metrics_from_disk",
    "_merge_sweep_outcomes",
    "_prepare_sweep_tasks",
    "_run_sweeps",
    "_select_best_configs",
    "_sweep_outcome_from_metrics",
]
