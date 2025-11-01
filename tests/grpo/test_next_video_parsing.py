#!/usr/bin/env python
# pylint: disable=missing-function-docstring
from __future__ import annotations

from grpo.next_video import _parse_index


def test_parse_index_prefers_answer_tag() -> None:
    assert _parse_index("some text <answer> 2 </answer> trailing") == 2


def test_parse_index_falls_back_to_trailing_lines() -> None:
    raw = """
    explanation about choices
    final line
    3
    """.strip()
    assert _parse_index(raw) == 3


def test_parse_index_returns_none_for_invalid_output() -> None:
    assert _parse_index("no tag, not a number") is None

