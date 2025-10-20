"""Smoke tests for prompt building and filtering."""

from __future__ import annotations

import json

import pytest

from datasets import Dataset, DatasetDict

from clean_data.filters import filter_prompt_ready
from clean_data.prompt.constants import PASSTHROUGH_COLUMNS, REQUIRED_PROMPT_COLUMNS
from clean_data.prompting import row_to_example

pytestmark = pytest.mark.prompt_smoke


def _base_example() -> dict:
    slate_items = [
        {"title": "First option", "id": "next_video"},
        {"title": "Backup", "id": "backup"},
    ]
    history = [
        {"id": "current_video", "title": "Current", "watch_seconds": 12, "total_length": 20},
        {"id": "earlier", "title": "Earlier", "watch_seconds": 30, "total_length": 45},
    ]
    return {
        "issue": "gun_control",
        "slate_items_json": json.dumps(slate_items),
        "watched_detailed_json": json.dumps(history),
        "watched_vids_json": json.dumps(["earlier", "current_video", "next_video"]),
        "current_video_id": "current_video",
        "current_video_title": "Current",
        "next_video_id": "next_video",
        "n_options": 2,
        "viewer_profile_sentence": "",
    }


def test_row_to_example_produces_prompt():
    example = _base_example()
    prompt = row_to_example(example, system_prompt=None, sol_key=None, max_hist=3)
    assert prompt is not None
    assert prompt["answer"] == "1"
    assert prompt["slate_items"][0]["id"] == "next_video"


def test_row_to_example_contains_required_columns():
    example = _base_example()
    example.update(
        {
            "session_id": "session-123",
            "step_index": 0,
            "display_step": 1,
            "urlid": "url-abc",
            "topic_id": "gun_control",
        }
    )
    prompt = row_to_example(example, system_prompt=None, sol_key=None, max_hist=3)
    assert prompt is not None

    missing_required = [col for col in REQUIRED_PROMPT_COLUMNS if col not in prompt]
    assert not missing_required, f"Missing required columns: {missing_required}"

    passthrough_present = [col for col in PASSTHROUGH_COLUMNS if col in example]
    for column in passthrough_present:
        assert prompt[column] == example[column]


def test_filter_prompt_ready_filters_missing_slates():
    good = Dataset.from_dict({key: [value] for key, value in _base_example().items()})
    filtered = filter_prompt_ready(DatasetDict({"train": good}))
    assert len(filtered["train"]) == 1

    bad_example = _base_example()
    bad_example["slate_items_json"] = json.dumps([])
    bad = Dataset.from_dict({key: [value] for key, value in bad_example.items()})
    filtered_bad = filter_prompt_ready(DatasetDict({"train": bad}))
    assert len(filtered_bad["train"]) == 0
