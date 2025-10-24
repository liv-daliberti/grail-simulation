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

"""Shared utilities for merging sweep outcomes and representing selections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Sequence, TypeVar


T = TypeVar("T")
StudyT = TypeVar("StudyT")
OutcomeT = TypeVar("OutcomeT")


def merge_ordered(
    cached: Sequence[T],
    executed: Sequence[T],
    *,
    order_key: Callable[[T], int],
    on_replace: Callable[[T, T], None] | None = None,
) -> List[T]:
    """Merge cached and freshly executed results while preserving order indices.

    :param cached: Previously materialised results (e.g., metrics read from disk).
    :type cached: Sequence[T]
    :param executed: Newly computed results from the current run.
    :type executed: Sequence[T]
    :param order_key: Callable returning a deterministic integer position for each result.
    :type order_key: Callable[[T], int]
    :param on_replace: Optional callback invoked when ``executed`` replaces ``cached`` at an index.
    :type on_replace: Callable[[T, T], None] | None
    :returns: Combined results ordered by ``order_key``.
    :rtype: List[T]
    """

    by_index: Dict[int, T] = {order_key(item): item for item in cached}
    for item in executed:
        index = order_key(item)
        if index in by_index and on_replace is not None:
            on_replace(by_index[index], item)
        by_index[index] = item
    return [by_index[index] for index in sorted(by_index)]


@dataclass
class OpinionStudySelection(Generic[StudyT, OutcomeT]):
    """Pair a study descriptor with the selected outcome for that study."""
    study: StudyT
    outcome: OutcomeT

    @property
    def config(self):
        """Return the configuration object associated with the selection."""
        return self.outcome.config


__all__ = ["merge_ordered", "OpinionStudySelection"]
