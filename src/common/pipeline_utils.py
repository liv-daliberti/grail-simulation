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
    """

    Merge cached and freshly executed results preserving order indices.



        Parameters

        ----------

        cached:

            Previously materialised results (e.g., metrics read from disk).

        executed:

            Newly computed results from the current run.

        order_key:

            Callable returning a deterministic integer position for each result.

        on_replace:

            Optional callback invoked when an executed item replaces a cached one

            at the same index.



    :param cached: Value provided for ``cached``.

    :type cached: Sequence[T]

    :param executed: Value provided for ``executed``.

    :type executed: Sequence[T]

    :param order_key: Value provided for ``order_key``.

    :type order_key: Callable[[T], int]

    :param on_replace: Value provided for ``on_replace``.

    :type on_replace: Callable[[T, T], None] | None

    :returns: Result produced by ``merge_ordered``.

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
    """

    Base selection container shared between opinion pipelines.



    :ivar study: Attribute ``study``.

    :vartype study: StudyT

    :ivar outcome: Attribute ``outcome``.

    :vartype outcome: OutcomeT

    """


    study: StudyT
    outcome: OutcomeT

    @property
    def config(self):
        """

        Return the configuration promoting this study outcome.



        :returns: Result produced by ``config``.

        :rtype: Any

        """


        return self.outcome.config


__all__ = ["merge_ordered", "OpinionStudySelection"]
