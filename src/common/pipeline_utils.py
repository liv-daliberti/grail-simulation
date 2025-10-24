"""Utility helpers shared across pipeline orchestration modules."""

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
