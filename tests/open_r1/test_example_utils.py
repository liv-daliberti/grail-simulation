#!/usr/bin/env python
"""Unit tests for :mod:`open_r1.example_utils` opinion helpers."""

from __future__ import annotations

import json

import pytest

from open_r1.constants import DEFAULT_SYSTEM_PROMPT
from open_r1.example_utils import (  # pylint: disable=protected-access
    _opinion_direction_label,
    row_to_training_example,
)


def _make_example(**overrides) -> dict[str, object]:
    """Return a minimal training row populated with sensible defaults."""

    base = {
        "issue": "gun_control",
        "participant_study": "study1",
        "gun_index": 3.0,
        "gun_index_2": 4.25,
        "slate_items_json": json.dumps(
            [
                {"title": "First Option", "id": "vid1"},
                {"title": "Second Option", "id": "vid2"},
            ]
        ),
        "n_options": 2,
        "next_video_id": "vid2",
        "watched_vids_json": "[]",
        "watched_detailed_json": "[]",
        "viewer_profile_sentence": "Viewer likes civic policy videos.",
        "current_video_id": "vid0",
        "current_video_title": "Policy Intro",
        "session_id": "sess-1",
        "step_index": 0,
        "display_step": 1,
    }
    base.update(overrides)
    return base


def test_opinion_direction_label_increase() -> None:
    row = _make_example()
    assert _opinion_direction_label(row) == "increase"


def test_opinion_direction_label_decrease() -> None:
    row = _make_example(gun_index=5.0, gun_index_2=3.75)
    assert _opinion_direction_label(row) == "decrease"


def test_opinion_direction_label_handles_tiny_deltas() -> None:
    row = _make_example(gun_index=4.0, gun_index_2=4.0000001)
    assert _opinion_direction_label(row) == "no_change"


def test_opinion_direction_label_missing_spec() -> None:
    row = _make_example(issue="other_issue")
    assert _opinion_direction_label(row) is None


def test_row_to_training_example_injects_opinion_direction() -> None:
    row = _make_example()
    result = row_to_training_example(
        row,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        solution_key="next_video_id",
    )
    assert result is not None
    assert result["opinion_direction"] == "increase"


def test_row_to_training_example_handles_missing_direction() -> None:
    row = _make_example(issue="other_issue")
    result = row_to_training_example(
        row,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        solution_key="next_video_id",
    )
    assert result is not None
    assert "opinion_direction" in result
    assert result["opinion_direction"] is None
