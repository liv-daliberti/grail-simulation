"""Smoke tests for clean_data module helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from clean_data.clean_data import (
    BuildOptions,
    build_clean_dataset,
    ensure_shared_schema,
    validate_required_columns,
)
from clean_data.filters import filter_prompt_ready


def _example_row() -> dict:
    slate_items = [
        {"title": "Option A", "id": "next_video"},
        {"title": "Option B", "id": "other"},
    ]
    history = [
        {"id": "current", "title": "Current", "watch_seconds": 12, "total_length": 20},
        {"id": "earlier", "title": "Earlier", "watch_seconds": 30, "total_length": 45},
    ]
    return {
        "issue": "minimum_wage",
        "slate_items_json": json.dumps(slate_items),
        "watched_detailed_json": json.dumps(history),
        "watched_vids_json": json.dumps(["earlier", "current", "next_video"]),
        "current_video_id": "current",
        "current_video_title": "Current",
        "next_video_id": "next_video",
        "n_options": 2,
        "viewer_profile_sentence": "",
    }


def test_validate_required_columns_detects_missing():
    dataset = DatasetDict({"train": Dataset.from_dict({"prompt": ["example"]})})
    with pytest.raises(ValueError):
        validate_required_columns(dataset)


def test_ensure_shared_schema_populates_missing_columns():
    left = Dataset.from_dict({"prompt": ["a"], "answer": ["1"]})
    right = Dataset.from_dict({"prompt": ["b"]})
    aligned = ensure_shared_schema({"train": left, "validation": right})
    assert set(aligned["validation"].column_names) == set(aligned["train"].column_names)


def test_build_clean_dataset_from_saved_directory():
    dataset = DatasetDict({"train": Dataset.from_dict({k: [v] for k, v in _example_row().items()})})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "saved_dataset"
        dataset.save_to_disk(path)
        result = build_clean_dataset(str(path), options=BuildOptions(validation_ratio=0.0))
    assert "train" in result
    row = result["train"][0]
    assert row["prompt"]
    assert row["answer"] == "1"


def test_filter_prompt_ready_removes_invalid_rows():
    good = Dataset.from_dict({key: [value] for key, value in _example_row().items()})
    filtered = filter_prompt_ready(DatasetDict({"train": good}))
    assert len(filtered["train"]) == 1

    bad_example = _example_row()
    bad_example["slate_items_json"] = json.dumps([])
    bad = Dataset.from_dict({key: [value] for key, value in bad_example.items()})
    filtered_bad = filter_prompt_ready(DatasetDict({"train": bad}))
    assert len(filtered_bad["train"]) == 0
