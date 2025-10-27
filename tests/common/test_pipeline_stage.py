"""Unit tests for :mod:`common.pipeline_stage`."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pytest

from common.pipeline_stage import (
    DryRunSummary,
    SweepPartitionExecutors,
    SweepPartitionSpec,
    dispatch_sweep_task,
    log_dry_run_summary,
    make_sweep_partition,
    prepare_sweep_execution,
)


@dataclass
class DummyTask:
    index: int
    metrics_path: Path
    label: str


@dataclass
class DummyOutcome:
    order_index: int
    metrics_path: Path
    label: str


def _executors(label: str, recorded: list[tuple[str, int]]) -> SweepPartitionExecutors[DummyTask, DummyOutcome]:
    def execute(task: DummyTask) -> DummyOutcome:
        recorded.append((label, task.index))
        return DummyOutcome(
            order_index=task.index,
            metrics_path=task.metrics_path,
            label=f"{label}-outcome-{task.index}",
        )

    return SweepPartitionExecutors(
        execute_task=execute,
        describe_pending=lambda task: task.label,
        describe_cached=lambda outcome: outcome.label,
    )


def test_make_sweep_partition_excludes_cached_overlap(tmp_path: Path) -> None:
    recorded: list[tuple[str, int]] = []
    pending = [
        DummyTask(index=0, metrics_path=tmp_path / "pending-0.json", label="pending-0"),
        DummyTask(index=2, metrics_path=tmp_path / "pending-2.json", label="pending-2"),
    ]
    cached = [
        DummyOutcome(index, metrics_path=tmp_path / f"cached-{index}.json", label=f"cached-{index}")
        for index in (2, 3)
    ]

    partition = make_sweep_partition(
        SweepPartitionSpec(
            label="Opinion",
            pending=pending,
            cached=cached,
            reuse_existing=False,
            executors=_executors("Opinion", recorded),
        )
    )

    assert partition.state.total_slots == 4
    assert set(partition.state.pending_by_index) == {0, 2}
    assert partition.state.cached_by_index == {3: cached[1]}
    assert partition.paths.pending(pending[0]) == pending[0].metrics_path
    assert partition.paths.cached(cached[1]) == cached[1].metrics_path
    assert recorded == []


def test_dispatch_sweep_task_invokes_expected_partition(tmp_path: Path, caplog) -> None:
    recorded: list[tuple[str, int]] = []
    partition_a = make_sweep_partition(
        SweepPartitionSpec(
            label="First",
            pending=[
                DummyTask(index=0, metrics_path=tmp_path / "first-0.json", label="First-0"),
                DummyTask(index=1, metrics_path=tmp_path / "first-1.json", label="First-1"),
            ],
            cached=[],
            reuse_existing=False,
            executors=_executors("First", recorded),
            prefix="first",
        )
    )
    partition_b = make_sweep_partition(
        SweepPartitionSpec(
            label="Second",
            pending=[
                DummyTask(index=0, metrics_path=tmp_path / "second-0.json", label="Second-0"),
            ],
            cached=[],
            reuse_existing=False,
            executors=_executors("Second", recorded),
            prefix="second",
        )
    )
    logger = logging.getLogger("tests.common.pipeline_stage.dispatch")

    with caplog.at_level(logging.INFO, logger=logger.name):
        dispatch_sweep_task(
            [partition_a, partition_b],
            task_id=2,
            logger=logger,
        )

    assert recorded == [("Second", 0)]
    assert any("Completed sweep task 0 (Second Second-0" in message for message in caplog.messages)


def test_dispatch_sweep_task_raises_for_out_of_range(tmp_path: Path) -> None:
    partition = make_sweep_partition(
        SweepPartitionSpec(
            label="Only",
            pending=[
                DummyTask(index=0, metrics_path=tmp_path / "only-0.json", label="Only-0"),
                DummyTask(index=1, metrics_path=tmp_path / "only-1.json", label="Only-1"),
            ],
            cached=[],
            reuse_existing=False,
            executors=_executors("Only", []),
        )
    )
    logger = logging.getLogger("tests.common.pipeline_stage.range")

    with pytest.raises(RuntimeError, match="beyond partitioned workload"):
        dispatch_sweep_task([partition], task_id=3, logger=logger)


def test_prepare_sweep_execution_reads_env_and_warns(monkeypatch, caplog) -> None:
    logger = logging.getLogger("tests.common.pipeline_stage.prepare.env")
    monkeypatch.setenv("CUSTOM_TASK_ID", "2")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        task_id = prepare_sweep_execution(
            total_tasks=3,
            cli_task_id=None,
            cli_task_count=4,
            logger=logger,
            env_var="CUSTOM_TASK_ID",
        )

    assert task_id == 2
    assert caplog.messages == ["Sweep task count mismatch: expected=3 provided=4."]


def test_prepare_sweep_execution_skips_when_cli_count_exceeds_total(caplog) -> None:
    logger = logging.getLogger("tests.common.pipeline_stage.prepare.skip")

    with caplog.at_level(logging.INFO, logger=logger.name):
        task_id = prepare_sweep_execution(
            total_tasks=3,
            cli_task_id=4,
            cli_task_count=5,
            logger=logger,
        )

    assert task_id is None
    assert "Sweep task index 4 beyond pending range (0..2); skipping." in caplog.messages
    assert "Sweep task count mismatch: expected=3 provided=5." in caplog.messages


def test_prepare_sweep_execution_requires_env(monkeypatch) -> None:
    logger = logging.getLogger("tests.common.pipeline_stage.prepare.required")
    monkeypatch.delenv("MISSING_TASK_ID", raising=False)

    with pytest.raises(RuntimeError, match="requires --sweep-task-id"):
        prepare_sweep_execution(
            total_tasks=1,
            cli_task_id=None,
            cli_task_count=None,
            logger=logger,
            env_var="MISSING_TASK_ID",
        )


def test_log_dry_run_summary_formats_entries(caplog) -> None:
    logger = logging.getLogger("tests.common.pipeline_stage.dryrun")
    entries = [
        DryRunSummary(label="alpha", pending=2, cached=1),
        DryRunSummary(label="beta", pending=0, cached=3),
    ]

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_dry_run_summary(logger, entries)

    assert caplog.messages == [
        "Dry-run mode. alpha pending=2 cached=1; beta pending=0 cached=3."
    ]
