"""Unit tests for prompt utility helpers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from clean_data.prompt.utils import (
    canonical_slate_items,
    core_prompt_mask,
    count_prior_history,
)


def test_count_prior_history_parses_serialised_history() -> None:
    """Ensure JSON strings with watch history produce the expected count."""
    row = {
        "current_video_id": "current",
        "watched_vids_json": json.dumps(["earlier", "current", "next"]),
        "watched_detailed_json": json.dumps(
            [
                {"id": "current", "title": "Current"},
                {"id": "earlier", "title": "Earlier"},
            ]
        ),
    }
    assert count_prior_history(row) == 1


def test_count_prior_history_handles_missing_current_fallback() -> None:
    """When the current id is missing from lists, use the last item as current."""
    row = {
        "current_video_id": "missing",
        "watched_vids_json": json.dumps(["earlier", "later"]),
    }
    assert count_prior_history(row) == 1


def test_canonical_slate_items_handles_arrays() -> None:
    """Ensure slate normalization extracts ordered video identifiers."""
    slate = np.array(
        [
            {"id": "abc"},
            {"video_id": "def"},
            {"id": " ghi "},
            {"id": ""},
        ]
    )
    assert canonical_slate_items(slate) == ("abc", "def", "ghi")


def test_core_prompt_mask_filters_non_core_studies() -> None:
    """Rows outside the core studies/issues should be excluded."""
    df = pd.DataFrame(
        {
            "participant_study": ["study1", "study4", "unknown"],
            "issue": ["gun_control", "minimum_wage", "other_issue"],
        }
    )
    mask = core_prompt_mask(df)
    assert mask.tolist() == [True, False, False]
