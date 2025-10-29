"""Unit tests for :mod:`common.pipeline.utils`."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from common.pipeline.utils import (
    merge_indexed_outcomes,
    merge_ordered,
    merge_ordered_with_warning,
)


@dataclass
class DummyItem:
    index: int
    label: str


@dataclass
class DummyOutcome:
    order_index: int
    label: str


def test_merge_ordered_prefers_executed_and_invokes_callback() -> None:
    cached = [DummyItem(index=0, label="cached-0"), DummyItem(index=2, label="cached-2")]
    executed = [DummyItem(index=1, label="executed-1"), DummyItem(index=2, label="executed-2")]
    replacements: list[tuple[str, str]] = []

    def on_replace(old, new) -> None:
        replacements.append((old.label, new.label))

    merged = merge_ordered(
        cached,
        executed,
        order_key=lambda item: item.index,
        on_replace=on_replace,
    )
    assert [item.label for item in merged] == ["cached-0", "executed-1", "executed-2"]
    assert replacements == [("cached-2", "executed-2")]


def test_merge_ordered_with_warning_logs_duplicates(caplog) -> None:
    logger = logging.getLogger("tests.common.pipeline.utils.warning")
    cached = [DummyItem(index=5, label="cached-five")]
    executed = [DummyItem(index=5, label="executed-five")]

    with caplog.at_level(logging.WARNING, logger=logger.name):
        merged = merge_ordered_with_warning(
            cached,
            executed,
            order_key=lambda item: item.index,
            duplicate_warning=lambda old, new: logger.warning(
                "Replacing %s with %s", old.label, new.label
            ),
        )

    assert [item.label for item in merged] == ["executed-five"]
    assert caplog.messages == ["Replacing cached-five with executed-five"]


def test_merge_indexed_outcomes_replaces_cached_entries(caplog) -> None:
    logger = logging.getLogger("tests.common.pipeline.utils.indexed")
    cached = [
        DummyOutcome(order_index=0, label="cached-0"),
        DummyOutcome(order_index=3, label="cached-3"),
    ]
    executed = [
        DummyOutcome(order_index=0, label="executed-0"),
        DummyOutcome(order_index=2, label="executed-2"),
    ]

    with caplog.at_level(logging.WARNING, logger=logger.name):
        merged = merge_indexed_outcomes(
            cached,
            executed,
            logger=logger,
            message="Duplicate at %d replaced: %s -> %s",
            args_factory=lambda old, new: (
                old.order_index,
                old.label,
                new.label,
            ),
        )

    assert [item.label for item in merged] == ["executed-0", "executed-2", "cached-3"]
    assert caplog.messages == ["Duplicate at 0 replaced: cached-0 -> executed-0"]
