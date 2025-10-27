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

import logging
from operator import attrgetter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Sequence, TypeVar


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


def make_duplicate_warning(
    logger: logging.Logger,
    message: str,
    *,
    args_factory: Callable[[T, T], Sequence[Any]] | None = None,
) -> Callable[[T, T], None]:
    """Return an ``on_replace`` callback that logs duplicate outcomes consistently.

    :param logger: Logger used to emit the warning.
    :type logger: logging.Logger
    :param message: Logging format string passed to :py:meth:`logging.Logger.warning`.
    :type message: str
    :param args_factory: Optional callable returning the arguments supplied alongside the
        format string. When omitted, the message is emitted without additional arguments.
    :type args_factory: Callable[[T, T], Sequence[Any]] | None
    :returns: Callback suitable for the ``on_replace`` parameter of :func:`merge_ordered`.
    :rtype: Callable[[T, T], None]
    """

    def _on_replace(existing: T, incoming: T) -> None:
        if args_factory is None:
            logger.warning(message)
        else:
            logger.warning(message, *args_factory(existing, incoming))

    return _on_replace


def merge_ordered_with_warning(
    cached: Sequence[T],
    executed: Sequence[T],
    *,
    order_key: Callable[[T], int],
    logger: logging.Logger,
    message: str,
    args_factory: Callable[[T, T], Sequence[Any]] | None = None,
) -> List[T]:
    """
    Convenience wrapper around :func:`merge_ordered` that logs duplicate replacements.

    :param cached: Previously materialised results (e.g., metrics read from disk).
    :type cached: Sequence[T]
    :param executed: Newly computed results from the current run.
    :type executed: Sequence[T]
    :param order_key: Callable returning a deterministic integer position for each result.
    :type order_key: Callable[[T], int]
    :param logger: Logger used when emitting duplicate warnings.
    :type logger: logging.Logger
    :param message: Logging format string passed to :py:meth:`logging.Logger.warning`.
    :type message: str
    :param args_factory: Optional callable returning arguments supplied with the warning.
    :type args_factory: Callable[[T, T], Sequence[Any]] | None
    :returns: Combined results ordered by ``order_key``.
    :rtype: List[T]
    """

    return merge_ordered(
        cached,
        executed,
        order_key=order_key,
        on_replace=make_duplicate_warning(
            logger,
            message,
            args_factory=args_factory,
        ),
    )


def merge_indexed_outcomes(
    cached: Sequence[T],
    executed: Sequence[T],
    *,
    logger: logging.Logger,
    message: str,
    args_factory: Callable[[T, T], Sequence[Any]] | None = None,
) -> List[T]:
    """Merge outcomes ordered by their ``order_index`` attribute.

    This wraps :func:`merge_ordered_with_warning` with a fixed ``order_key`` based on
    ``order_index`` to cut down on repeated boilerplate across pipeline modules.
    """

    return merge_ordered_with_warning(
        cached,
        executed,
        order_key=attrgetter("order_index"),
        logger=logger,
        message=message,
        args_factory=args_factory,
    )


@dataclass
class OpinionStudySelection(Generic[StudyT, OutcomeT]):
    """Pair a study descriptor with the selected outcome for that study."""
    study: StudyT
    outcome: OutcomeT

    @property
    def config(self):
        """Return the configuration object associated with the selection."""
        return self.outcome.config


__all__ = [
    "merge_ordered",
    "make_duplicate_warning",
    "merge_ordered_with_warning",
    "merge_indexed_outcomes",
    "OpinionStudySelection",
]
