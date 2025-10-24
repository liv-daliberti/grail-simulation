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

    Execute ``tasks`` using ``worker`` with optional parallelism.



    :param tasks: Value provided for ``tasks``.

    :type tasks: Sequence[TaskT]

    :param worker: Value provided for ``worker``.

    :type worker: Callable[[TaskT], ResultT]

    :param jobs: Value provided for ``jobs``.

    :type jobs: int

    :param logger: Value provided for ``logger``.

    :type logger: Any

    :param label: Value provided for ``label``.

    :type label: str

    :returns: Result produced by ``execute_indexed_tasks``.

    :rtype: List[ResultT]

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


def execute_sequential_tasks(
    tasks: Sequence[TaskT],
    worker: Callable[[TaskT], ResultT],
) -> List[ResultT]:
    """

    Execute ``tasks`` sequentially and collect the results.



    :param tasks: Value provided for ``tasks``.

    :type tasks: Sequence[TaskT]

    :param worker: Value provided for ``worker``.

    :type worker: Callable[[TaskT], ResultT]

    :returns: Result produced by ``execute_sequential_tasks``.

    :rtype: List[ResultT]

    """


    results: List[ResultT] = []
    for task in tasks:
        results.append(worker(task))
    return results
