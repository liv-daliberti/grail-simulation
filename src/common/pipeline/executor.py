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

"""Generic task execution helpers used by sweep orchestration code."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Optional, Sequence, TypeVar

TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


def execute_indexed_tasks(
    tasks: Sequence[TaskT],
    worker: Callable[[TaskT], ResultT],
    *,
    jobs: int,
    logger,
    label: str = "task",
) -> List[ResultT]:
    """
    Execute ``tasks`` possibly in parallel, preserving task order.

    :param tasks: Ordered sequence of task payloads.
    :param worker: Callable invoked for each task payload.
    :param jobs: Maximum number of parallel processes to spawn.
    :param logger: Logger receiving progress information.
    :param label: Human-readable label used in log messages and errors.
    :returns: List of results aligned with the original task ordering.
    :raises RuntimeError: If any task completes without returning a result.
    """
    if not tasks:
        return []

    jobs = max(1, jobs)
    if jobs == 1:
        return [worker(task) for task in tasks]

    logger.info("Launching %d parallel %s workers across %d tasks.", jobs, label, len(tasks))
    ordered_results: List[Optional[ResultT]] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_index = {executor.submit(worker, task): index for index, task in enumerate(tasks)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            ordered_results[index] = future.result()

    results: List[ResultT] = []
    for maybe_result in ordered_results:
        if maybe_result is None:
            raise RuntimeError(f"{label.capitalize()} completed without returning a result.")
        results.append(maybe_result)
    return results
