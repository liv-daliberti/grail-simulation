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

"""Opinion sweep orchestration helpers extracted from ``pipeline_sweeps``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from common.pipeline_executor import execute_indexed_tasks
from common.pipeline_utils import merge_indexed_outcomes
from common.opinion_sweep_types import AccuracySummary, MetricsArtifact

from .pipeline_context import (
    OpinionStudySelection,
    OpinionSweepOutcome,
    OpinionSweepTask,
    StudySpec,
    SweepConfig,
    SweepTaskContext,
)
from .pipeline_utils import (
    ensure_dir,
    ensure_opinion_selection_coverage,
    extract_opinion_summary,
    prepare_task_grid,
    TaskCacheStrategy,
)

LOGGER = logging.getLogger("knn.pipeline.sweeps")

CliRunner = Callable[[Sequence[str]], None]


def _compare_metric(lhs: Optional[float], rhs: Optional[float]) -> int:
    """Return comparison result {-1,0,1} for two optional floats."""

    if lhs is None or rhs is None:
        return 0
    delta = lhs - rhs
    if abs(delta) <= 1e-9:
        return 0
    return -1 if delta < 0 else 1


def _opinion_is_better(
    candidate: OpinionSweepOutcome, incumbent: OpinionSweepOutcome
) -> bool:
    """Determine whether ``candidate`` should replace ``incumbent``."""

    for lhs, rhs in ((candidate.mae, incumbent.mae), (candidate.rmse, incumbent.rmse)):
        result = _compare_metric(lhs, rhs)
        if result != 0:
            return result < 0

    candidate_participants = candidate.participants or 0
    incumbent_participants = incumbent.participants or 0
    if candidate_participants != incumbent_participants:
        return candidate_participants > incumbent_participants
    return candidate.best_k < incumbent.best_k


def _build_opinion_task(
    *,
    index: int,
    config: SweepConfig,
    study: StudySpec,
    context: SweepTaskContext,
) -> OpinionSweepTask:
    run_root = (
        context.sweep_dir
        / "opinion"
        / config.feature_space
        / study.study_slug
        / config.label()
    )
    metrics_path = (
        run_root
        / "opinion"
        / config.feature_space
        / study.key
        / f"opinion_knn_{study.key}_validation_metrics.json"
    )
    word2vec_model_dir = None
    if config.feature_space == "word2vec":
        word2vec_model_dir = (
            context.word2vec_model_base / "sweeps_opinion" / study.study_slug / config.label()
        )
    task = OpinionSweepTask(
        index=index,
        study=study,
        config=config,
        base_cli=tuple(context.base_cli),
        extra_cli=tuple(context.extra_cli),
        run_root=run_root,
        word2vec_model_dir=word2vec_model_dir,
        metrics_path=metrics_path,
    )
    return task


def build_opinion_task(
    *,
    index: int,
    config: SweepConfig,
    study: StudySpec,
    context: SweepTaskContext,
) -> OpinionSweepTask:
    """Public wrapper that exposes the opinion task builder for testing."""
    return _build_opinion_task(
        index=index,
        config=config,
        study=study,
        context=context,
    )


def _load_cached_opinion_outcome(task: OpinionSweepTask) -> Optional[OpinionSweepOutcome]:
    """Load cached opinion metrics if available."""

    try:
        with open(task.metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except FileNotFoundError:
        LOGGER.debug("Expected cached opinion metrics at %s but none found.", task.metrics_path)
        return None
    LOGGER.info(
        "[OPINION][SKIP] feature=%s study=%s label=%s (metrics cached).",
        task.config.feature_space,
        task.study.key,
        task.config.label(),
    )
    return opinion_sweep_outcome_from_metrics(task, metrics, task.metrics_path)


def prepare_opinion_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: SweepTaskContext,
    reuse_existing: bool,
) -> Tuple[List[OpinionSweepTask], List[OpinionSweepOutcome]]:
    """
    Return opinion sweep tasks requiring execution and cached outcomes.

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :param configs: Iterable of sweep configurations scheduled for execution.

    :type configs: Sequence[SweepConfig]

    :param context: Shared CLI/runtime parameters reused across sweep invocations.
    :type context: SweepTaskContext

    :param reuse_existing: Whether to reuse cached results instead of recomputing them.

    :type reuse_existing: bool

    :returns: opinion sweep tasks requiring execution and cached outcomes

    :rtype: Tuple[List[OpinionSweepTask], List[OpinionSweepOutcome]]

    """
    return prepare_task_grid(
        configs,
        studies,
        reuse_existing=reuse_existing,
        build_task=lambda task_index, config, study: _build_opinion_task(
            index=task_index,
            config=config,
            study=study,
            context=context,
        ),
        cache=TaskCacheStrategy(load_cached=_load_cached_opinion_outcome),
    )


def _coerce_optional_float(value: object, default: float | None = None) -> Optional[float]:
    """Return ``value`` converted to ``float`` when possible."""

    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def opinion_sweep_outcome_from_metrics(
    task: OpinionSweepTask,
    metrics: Mapping[str, object],
    metrics_path: Path,
) -> OpinionSweepOutcome:
    """
    Translate cached opinion metrics into an :class:`OpinionSweepOutcome`.

    :param task: Individual sweep task describing an execution unit.

    :type task: OpinionSweepTask

    :param metrics: Metrics dictionary captured from a previous pipeline stage.

    :type metrics: Mapping[str, object]

    :param metrics_path: Filesystem path where the metrics JSON artefact resides.

    :type metrics_path: Path

    :returns: Opinion sweep outcome reconstructed from cached metrics.

    :rtype: OpinionSweepOutcome

    """
    summary = extract_opinion_summary(metrics)

    mae = summary.mae if summary.mae is not None else float(metrics.get("best_mae", float("inf")))

    participants = summary.participants if summary.participants is not None else int(
        metrics.get("n_participants", 0)
    )
    best_k = summary.best_k if summary.best_k is not None else int(metrics.get("best_k", 0))

    baseline_mae = summary.baseline_mae
    if baseline_mae is None:
        baseline_mae = _coerce_optional_float(
            metrics.get("baseline", {}).get("mae_using_before")
        )

    mae_delta = summary.mae_delta
    if mae_delta is None and baseline_mae is not None:
        mae_delta = float(mae) - float(baseline_mae)

    accuracy = summary.accuracy
    if accuracy is None:
        accuracy = _coerce_optional_float(
            metrics.get("best_metrics", {}).get("direction_accuracy")
        )

    baseline_accuracy = summary.baseline_accuracy
    if baseline_accuracy is None:
        baseline_accuracy = _coerce_optional_float(
            metrics.get("baseline", {}).get("direction_accuracy")
        )

    accuracy_delta = summary.accuracy_delta
    if accuracy_delta is None and accuracy is not None and baseline_accuracy is not None:
        accuracy_delta = accuracy - baseline_accuracy

    eligible = summary.eligible
    if eligible is None:
        candidate = metrics.get("eligible")
        eligible = int(candidate) if isinstance(candidate, (int, float)) else None
    if eligible is None:
        eligible = participants

    return OpinionSweepOutcome(
        order_index=task.index,
        study=task.study,
        config=task.config,
        feature_space=task.config.feature_space,
        mae=float(mae),
        rmse=float(
            summary.rmse
            if summary.rmse is not None
            else _coerce_optional_float(
                metrics.get("best_metrics", {}).get("rmse_after"),
                default=0.0,
            )
        ),
        r2_score=float(
            summary.r2_score
            if summary.r2_score is not None
            else _coerce_optional_float(
                metrics.get("best_metrics", {}).get("r2_after"),
                default=0.0,
            )
        ),
        baseline_mae=baseline_mae,
        mae_delta=mae_delta,
        best_k=best_k,
        participants=participants,
        artifact=MetricsArtifact(path=metrics_path, payload=metrics),
        accuracy_summary=AccuracySummary(
            value=accuracy,
            baseline=baseline_accuracy,
            delta=accuracy_delta,
            eligible=eligible,
        ),
    )


def merge_opinion_sweep_outcomes(
    cached: Sequence[OpinionSweepOutcome],
    executed: Sequence[OpinionSweepOutcome],
) -> List[OpinionSweepOutcome]:
    """
    Combine cached and freshly executed opinion outcomes preserving order.

    :param cached: Previously computed artefacts available for reuse.

    :type cached: Sequence[OpinionSweepOutcome]

    :param executed: Iterable of tasks that were actually executed during the run.

    :type executed: Sequence[OpinionSweepOutcome]

    :returns: Mapping of feature spaces to merged opinion sweep outcomes.

    :rtype: List[OpinionSweepOutcome]

    """
    return merge_indexed_outcomes(
        cached,
        executed,
        logger=LOGGER,
        message="Duplicate opinion sweep outcome detected for index=%d (study=%s). Overwriting.",
        args_factory=lambda _existing, incoming: (
            incoming.order_index,
            incoming.study.key,
        ),
    )


def execute_opinion_sweep_tasks(
    tasks: Sequence[OpinionSweepTask],
    *,
    jobs: int = 1,
    cli_runner: CliRunner,
) -> List[OpinionSweepOutcome]:
    """
    Run the supplied opinion sweep tasks, optionally in parallel.

    :param tasks: Collection of sweep tasks scheduled for execution.
    :type tasks: Sequence[OpinionSweepTask]
    :param jobs: Maximum number of parallel workers allowed.
    :type jobs: int
    :param cli_runner: Callable used to invoke the CLI for each task.
    :type cli_runner: Callable[[Sequence[str]], None]
    :returns: List of opinion sweep outcomes generated from the provided tasks.
    :rtype: List[OpinionSweepOutcome]
    """
    if not tasks:
        return []

    return execute_indexed_tasks(
        tasks,
        lambda task: execute_opinion_sweep_task(task, cli_runner=cli_runner),
        jobs=jobs,
        logger=LOGGER,
        label="opinion sweep",
    )


def execute_opinion_sweep_task(
    task: OpinionSweepTask,
    *,
    cli_runner: CliRunner,
) -> OpinionSweepOutcome:
    """
    Execute a single opinion sweep task and return the captured metrics.

    :param task: Individual sweep task describing an execution unit.

    :type task: OpinionSweepTask

    :param cli_runner: Callable used to invoke the CLI for the opinion sweep.
    :type cli_runner: Callable[[Sequence[str]], None]

    :returns: Opinion sweep outcome produced by executing the given task.

    :rtype: OpinionSweepOutcome

    """
    run_root = ensure_dir(task.run_root)
    outputs_root = run_root / "opinion" / task.config.feature_space
    model_dir = None
    if task.config.feature_space == "word2vec":
        if task.word2vec_model_dir is None:
            raise RuntimeError("Word2Vec opinion sweep task missing model directory.")
        model_dir = ensure_dir(task.word2vec_model_dir)

    cli_args: List[str] = list(task.base_cli)
    cli_args.extend(task.config.cli_args(word2vec_model_dir=model_dir))
    cli_args.extend(["--task", "opinion"])
    cli_args.extend(["--out-dir", str(run_root)])
    cli_args.extend(["--opinion-studies", task.study.key])
    cli_args.extend(task.extra_cli)

    LOGGER.info(
        "[OPINION][SWEEP] feature=%s study=%s label=%s",
        task.config.feature_space,
        task.study.key,
        task.config.label(),
    )
    cli_runner(cli_args)
    metrics_path = (
        outputs_root / task.study.key / f"opinion_knn_{task.study.key}_validation_metrics.json"
    )
    if not metrics_path.exists():
        raise FileNotFoundError(f"Opinion sweep metrics missing at {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return opinion_sweep_outcome_from_metrics(task, metrics, metrics_path)


def format_opinion_sweep_task_descriptor(task: OpinionSweepTask) -> str:
    """
    Return a short descriptor for an opinion sweep task.

    :param task: Individual sweep task describing an execution unit.

    :type task: OpinionSweepTask

    :returns: a short descriptor for an opinion sweep task

    :rtype: str

    """
    return f"{task.config.feature_space}:{task.study.key}:{task.config.label()}"


def select_best_opinion_configs(
    *,
    outcomes: Sequence[OpinionSweepOutcome],
    studies: Sequence[StudySpec],
    allow_incomplete: bool = False,
) -> Dict[str, Dict[str, OpinionStudySelection]]:
    """
    Select the best configuration per feature space and study for opinion.

    :param outcomes: Iterable of sweep outcomes available for aggregation.

    :type outcomes: Sequence[OpinionSweepOutcome]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :param allow_incomplete: Whether processing may continue when some sweep data is missing.

    :type allow_incomplete: bool

    :returns: Mapping of feature spaces to their selected opinion configurations.

    :rtype: Dict[str, Dict[str, OpinionStudySelection]]

    """
    selections: Dict[str, Dict[str, OpinionStudySelection]] = {}
    for outcome in outcomes:
        per_feature = selections.setdefault(outcome.feature_space, {})
        current = per_feature.get(outcome.study.key)
        if current is None or _opinion_is_better(outcome, current.outcome):
            per_feature[outcome.study.key] = OpinionStudySelection(
                study=outcome.study,
                outcome=outcome,
            )

    ensure_opinion_selection_coverage(
        selections,
        studies,
        allow_incomplete=allow_incomplete,
        logger=LOGGER,
        require_selected=bool(outcomes),
    )
    return selections


__all__ = [
    "build_opinion_task",
    "execute_opinion_sweep_task",
    "execute_opinion_sweep_tasks",
    "format_opinion_sweep_task_descriptor",
    "merge_opinion_sweep_outcomes",
    "opinion_sweep_outcome_from_metrics",
    "prepare_opinion_sweep_tasks",
    "select_best_opinion_configs",
]
