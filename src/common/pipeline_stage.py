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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

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
    """

    Validate sweep arguments and return the task index to execute.



    :param total_tasks: Value provided for ``total_tasks``.

    :type total_tasks: int

    :param cli_task_id: Value provided for ``cli_task_id``.

    :type cli_task_id: Optional[int]

    :param cli_task_count: Value provided for ``cli_task_count``.

    :type cli_task_count: Optional[int]

    :param logger: Value provided for ``logger``.

    :type logger: Any

    :param env_var: Value provided for ``env_var``.

    :type env_var: str

    :returns: Result produced by ``prepare_sweep_execution``.

    :rtype: Optional[int]

    """


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
        raise RuntimeError(
            f"Sweep task index {task_id} out of range (0..{total_tasks - 1})."
        )
    return task_id
