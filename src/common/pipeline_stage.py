"""Common helpers for pipeline stage orchestration."""

from __future__ import annotations

import os
from typing import Optional


def prepare_sweep_execution(
    *,
    total_tasks: int,
    cli_task_id: Optional[int],
    cli_task_count: Optional[int],
    logger,
    env_var: str = "SLURM_ARRAY_TASK_ID",
) -> Optional[int]:
    """Validate sweep arguments and return the task index to execute."""

    if total_tasks == 0:
        logger.info("No sweep tasks pending; existing metrics cover the grid.")
        return None

    task_id = cli_task_id
    if task_id is None:
        env_value = os.environ.get(env_var)
        if env_value is None:
            raise RuntimeError(
                "Sweep stage requires --sweep-task-id or the SLURM_ARRAY_TASK_ID environment variable."
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

    if not (0 <= task_id < total_tasks):
        raise RuntimeError(
            f"Sweep task index {task_id} out of range (0..{total_tasks - 1})."
        )
    return task_id

