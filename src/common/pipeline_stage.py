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

"""Shared helpers that coordinate distributed sweep execution."""

from __future__ import annotations

import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generic, Optional, Sequence, TypeVar


TaskT = TypeVar("TaskT")
OutcomeT = TypeVar("OutcomeT")


@dataclass(frozen=True)
class DryRunSummary:
    """Summary information for a pipeline dry-run."""

    label: str
    pending: int
    cached: int


def log_dry_run_summary(logger, entries: Sequence[DryRunSummary]) -> None:
    """Emit a consistent dry-run summary across pipeline implementations."""

    summary_bits = [
        f"{entry.label} pending={entry.pending} cached={entry.cached}"
        for entry in entries
    ]
    logger.info(
        "Dry-run mode. %s.",
        "; ".join(summary_bits) if summary_bits else "No tasks selected.",
    )


def emit_stage_dry_run_summary(
    logger,
    *,
    include_next: bool,
    next_label: str,
    next_pending: int,
    next_cached: int,
    include_opinion: bool,
    opinion_pending: int,
    opinion_cached: int,
) -> None:
    """Convenience wrapper that logs dry-run summaries for next-video and opinion stages."""

    summaries: list[DryRunSummary] = []
    if include_next:
        summaries.append(
            DryRunSummary(label=next_label, pending=next_pending, cached=next_cached)
        )
    if include_opinion:
        summaries.append(
            DryRunSummary(
                label="opinion",
                pending=opinion_pending,
                cached=opinion_cached,
            )
        )
    log_dry_run_summary(logger, summaries)


@dataclass(frozen=True)
class SweepPartitionExecutors(Generic[TaskT, OutcomeT]):
    """Callable bundle operating on sweep partition tasks."""

    execute_task: Callable[[TaskT], OutcomeT]
    describe_pending: Callable[[TaskT], str]
    describe_cached: Callable[[OutcomeT], str]


@dataclass(frozen=True)
class SweepPartitionPaths(Generic[TaskT, OutcomeT]):
    """Path accessors for sweep partition artifacts."""

    pending: Callable[[TaskT], Path]
    cached: Callable[[OutcomeT], Path]


@dataclass(frozen=True)
class SweepPartitionLookups(Generic[TaskT, OutcomeT]):
    """Optional lookup providers customised by callers."""

    indexers: SweepPartitionIndexers[TaskT, OutcomeT] | None = None
    paths: SweepPartitionPaths[TaskT, OutcomeT] | None = None


@dataclass(frozen=True)
class SweepPartitionSpec(Generic[TaskT, OutcomeT]):
    """Configuration bundle describing a sweep partition."""

    label: str
    pending: Sequence[TaskT]
    cached: Sequence[OutcomeT]
    reuse_existing: bool
    executors: SweepPartitionExecutors[TaskT, OutcomeT]
    prefix: str = ""
    lookups: SweepPartitionLookups[TaskT, OutcomeT] | None = None


@dataclass(frozen=True)
class SweepPartitionIndexers(Generic[TaskT, OutcomeT]):
    """Index accessors used when normalising sweep partitions."""

    pending: Callable[[TaskT], int]
    cached: Callable[[OutcomeT], int]


@dataclass(frozen=True)
class SweepPartitionState(Generic[TaskT, OutcomeT]):
    """Lookup tables representing tasks and cached outcomes."""

    pending_by_index: Dict[int, TaskT]
    cached_by_index: Dict[int, OutcomeT]
    total_slots: int


@dataclass
class SweepPartition(Generic[TaskT, OutcomeT]):
    """Partition of sweep tasks that share execution semantics."""

    label: str
    state: SweepPartitionState[TaskT, OutcomeT]
    reuse_existing: bool
    prefix: str
    executors: SweepPartitionExecutors[TaskT, OutcomeT]
    paths: SweepPartitionPaths[TaskT, OutcomeT]

def _default_pending_index(task: TaskT) -> int:
    return getattr(task, "index")


def _default_cached_index(outcome: OutcomeT) -> int:
    return getattr(outcome, "order_index")


def _default_pending_metrics_path(task: TaskT) -> Path:
    return getattr(task, "metrics_path")


def _default_cached_metrics_path(outcome: OutcomeT) -> Path:
    return getattr(outcome, "metrics_path")


_DEFAULT_INDEXERS: SweepPartitionIndexers[TaskT, OutcomeT] = SweepPartitionIndexers(
    pending=_default_pending_index,
    cached=_default_cached_index,
)
_DEFAULT_PATHS: SweepPartitionPaths[TaskT, OutcomeT] = SweepPartitionPaths(
    pending=_default_pending_metrics_path,
    cached=_default_cached_metrics_path,
)


def make_sweep_partition(
    spec: SweepPartitionSpec[TaskT, OutcomeT],
) -> SweepPartition[TaskT, OutcomeT]:
    """Normalise sweep task partitions for distributed execution."""

    lookups = spec.lookups or SweepPartitionLookups()
    indexers = lookups.indexers or _DEFAULT_INDEXERS
    paths = lookups.paths or _DEFAULT_PATHS
    pending_by_index = {indexers.pending(task): task for task in spec.pending}
    cached_by_index: Dict[int, OutcomeT] = {}
    for outcome in spec.cached:
        index = indexers.cached(outcome)
        if index in pending_by_index:
            continue
        cached_by_index[index] = outcome
    indices = set(pending_by_index).union(cached_by_index)
    total_slots = (max(indices) + 1) if indices else 0
    state = SweepPartitionState(
        pending_by_index=pending_by_index,
        cached_by_index=cached_by_index,
        total_slots=total_slots,
    )
    return SweepPartition(
        label=spec.label,
        reuse_existing=spec.reuse_existing,
        prefix=spec.prefix,
        state=state,
        executors=spec.executors,
        paths=paths,
    )


def build_sweep_partition(  # pylint: disable=too-many-arguments
    *,
    label: str,
    pending: Sequence[TaskT],
    cached: Sequence[OutcomeT],
    reuse_existing: bool,
    executors: SweepPartitionExecutors[TaskT, OutcomeT],
    prefix: str = "",
    lookups: SweepPartitionLookups[TaskT, OutcomeT] | None = None,
) -> SweepPartition[TaskT, OutcomeT]:
    """
    Convenience wrapper around :func:`make_sweep_partition` for common pipelines.

    Pipelines frequently construct ``SweepPartitionSpec`` instances with only the
    standard arguments. Wrapping the boilerplate helps avoid duplicate code in
    downstream packages while keeping the explicit spec-based API available for
    advanced use-cases.
    """

    spec = SweepPartitionSpec(
        label=label,
        pending=pending,
        cached=cached,
        reuse_existing=reuse_existing,
        executors=executors,
        prefix=prefix,
        lookups=lookups,
    )
    return make_sweep_partition(spec)


def _run_partition_task(
    partition: SweepPartition[TaskT, OutcomeT],
    task_index: int,
    *,
    logger,
) -> None:
    """Execute or skip a single sweep task within ``partition``."""

    prefix = f"{partition.prefix} " if partition.prefix else ""
    state = partition.state
    executors = partition.executors
    task = state.pending_by_index.get(task_index)
    if task is None:
        cached = state.cached_by_index.get(task_index)
        if cached is not None:
            logger.info(
                (
                    "%sSkipping sweep task %d (%s %s); metrics already present "
                    "at %s."
                ),
                prefix,
                task_index,
                partition.label,
                executors.describe_cached(cached),
                partition.paths.cached(cached),
            )
            return
        logger.warning(
            "%sNo %s sweep task registered for index %d; skipping.",
            prefix,
            partition.label,
            task_index,
        )
        return

    metrics_path = partition.paths.pending(task)
    if partition.reuse_existing and metrics_path.exists():
        logger.info(
            "%sSkipping sweep task %d (%s %s); metrics already present at %s.",
            prefix,
            task_index,
            partition.label,
            executors.describe_pending(task),
            metrics_path,
        )
        return

    outcome = executors.execute_task(task)
    order_index = getattr(outcome, "order_index", task_index)
    logger.info(
        "%sCompleted sweep task %d (%s %s; order=%d). Metrics stored at %s.",
        prefix,
        task_index,
        partition.label,
        executors.describe_pending(task),
        order_index,
        getattr(outcome, "metrics_path", metrics_path),
    )


def dispatch_sweep_task(
    partitions: Sequence[SweepPartition[TaskT, OutcomeT]],
    *,
    task_id: int,
    logger,
) -> None:
    """Dispatch ``task_id`` to the appropriate sweep partition."""

    offset = 0
    for partition in partitions:
        total_slots = partition.state.total_slots
        if task_id < offset + total_slots:
            _run_partition_task(partition, task_id - offset, logger=logger)
            return
        offset += total_slots
    raise RuntimeError(
        f"Sweep task index {task_id} beyond partitioned workload (total={offset})."
    )


def prepare_sweep_execution(
    *,
    total_tasks: int,
    cli_task_id: Optional[int],
    cli_task_count: Optional[int],
    logger,
    env_var: str = "SLURM_ARRAY_TASK_ID",
) -> Optional[int]:
    """Resolve the sweep task index to execute for the current worker."""
    if total_tasks == 0:
        logger.info("No sweep tasks pending; existing metrics cover the grid.")
        return None

    task_id = cli_task_id
    if task_id is None:
        env_value = os.environ.get(env_var)
        if env_value is None:
            raise RuntimeError(
                "Sweep stage requires --sweep-task-id or the "
                "SLURM_ARRAY_TASK_ID environment variable."
            )
        try:
            task_id = int(env_value)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid {env_var} '{env_value}'; expected an integer."
            ) from exc

    if cli_task_count is not None and cli_task_count != total_tasks:
        logger.warning(
            "Sweep task count mismatch: expected=%d provided=%d.",
            total_tasks,
            cli_task_count,
        )

    if not 0 <= task_id < total_tasks:
        if (
            cli_task_count is not None
            and cli_task_count > total_tasks
            and 0 <= task_id < cli_task_count
        ):
            logger.info(
                "Sweep task index %d beyond pending range (0..%d); skipping.",
                task_id,
                total_tasks - 1,
            )
            return None
        raise RuntimeError(
            f"Sweep task index {task_id} out of range (0..{total_tasks - 1})."
        )
    return task_id


@dataclass(frozen=True)
class SweepExecutionOptions:
    """Parameters controlling how sweep partitions are dispatched."""

    cli_task_id: Optional[int] = None
    cli_task_count: Optional[int] = None
    env_var: str = "SLURM_ARRAY_TASK_ID"
    prepare: Callable[..., Optional[int]] = prepare_sweep_execution


def execute_sweep_partitions(
    partitions: Sequence[SweepPartition[TaskT, OutcomeT]],
    *,
    logger,
    options: SweepExecutionOptions | None = None,
) -> Optional[int]:
    """
    Resolve the sweep task to run and dispatch it to ``partitions``.

    Replaces the repeated pattern of computing the aggregate task count,
    invoking :func:`prepare_sweep_execution`, and calling
    :func:`dispatch_sweep_task`. Returns the concrete task index when a
    task is executed and ``None`` when no work is required.
    """

    options = options or SweepExecutionOptions()
    total_tasks = sum(partition.state.total_slots for partition in partitions)
    task_id = options.prepare(
        total_tasks=total_tasks,
        cli_task_id=options.cli_task_id,
        cli_task_count=options.cli_task_count,
        logger=logger,
        env_var=options.env_var,
    )
    if task_id is None:
        return None
    dispatch_sweep_task(partitions, task_id=task_id, logger=logger)
    return task_id


def execute_partitions_for_cli(  # pragma: no cover - thin convenience wrapper
    partitions: Sequence[SweepPartition[TaskT, OutcomeT]],
    *,
    args: Namespace,
    logger,
    prepare: Callable[..., Optional[int]] = prepare_sweep_execution,
) -> Optional[int]:
    """
    Convenience wrapper that extracts CLI sweep task arguments from ``args``.

    Reduces duplication across pipelines by forwarding the relevant namespace
    attributes to :func:`execute_sweep_partitions`.
    """

    return execute_sweep_partitions(
        partitions,
        logger=logger,
        options=SweepExecutionOptions(
            cli_task_id=getattr(args, "sweep_task_id", None),
            cli_task_count=getattr(args, "sweep_task_count", None),
            prepare=prepare,
        ),
    )


def dispatch_cli_partitions(
    partitions: Sequence[SweepPartition[TaskT, OutcomeT]],
    *,
    args: Namespace,
    logger,
    prepare: Callable[..., Optional[int]] = prepare_sweep_execution,
) -> None:
    """
    Execute sweep partitions using CLI arguments without returning a task id.

    Dedicated helper used by pipelines that only need side effects from
    :func:`execute_partitions_for_cli`, reducing duplicate call scaffolding.
    """

    execute_partitions_for_cli(
        partitions,
        args=args,
        logger=logger,
        prepare=prepare,
    )
