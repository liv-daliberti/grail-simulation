"""Integration tests covering clean_data → prompt builder → baseline pipelines."""

from __future__ import annotations

import json

import pytest
from datasets import Dataset

from clean_data.prompting import row_to_example
from gpt4o.conversation import make_conversation_record
from knn.features import assemble_document, prepare_training_documents
from prompt_builder import build_user_prompt
from xgb.features import prepare_prompt_documents


@pytest.fixture
def raw_clean_row() -> dict:
    slate_items = [
        {
            "id": "dQw4w9WgXcQ",
            "title": "Policy Analysis Deep Dive",
            "channel_title": "Civic Reports",
            "length_seconds": 240,
        },
        {
            "id": "J---aiyznGQ",
            "title": "Opposing View Commentary",
            "channel_title": "Public Debate",
            "length_seconds": 180,
        },
    ]
    watch_history = [
        {
            "id": "prevVideo1B",
            "title": "Previous Topic Overview",
            "watch_seconds": 90,
            "total_length": 200,
            "channel_title": "Civic Reports",
        },
        {
            "id": "curVideo01A",
            "title": "Current Issue Summary",
            "watch_seconds": 120,
            "total_length": 220,
            "channel_title": "Civic Reports",
        },
    ]
    return {
        "session_id": "sess-001",
        "step_index": 2,
        "display_step": 3,
        "display_order_key": "sess-001:3",
        "issue": "gun_control",
        "issue_source": "survey",
        "issue_detail": "Gun control policy preferences",
        "slate_source": "ranking_model",
        "topic_id": "gun-policy",
        "urlid": "video://sample",
        "viewer_profile_sentence": "",
        "age": 42,
        "gender": "woman",
        "race": "White",
        "city": "Chicago",
        "state": "IL",
        "education": "college",
        "college": "yes",
        "income": "$75,000 - $99,999",
        "employment_status": "employed",
        "marital_status": "married",
        "children_in_house": "yes",
        "household_size": 4,
        "religion": "Protestant",
        "ideo1": "liberal",
        "pid1": "Democrat",
        "freq_youtube": "3",
        "current_video_id": "curVideo01A",
        "current_video_title": "Current Issue Summary",
        "current_video_channel": "Civic Reports",
        "next_video_id": "dQw4w9WgXcQ",
        "next_video_title": "Policy Analysis Deep Dive",
        "next_video_channel": "Civic Reports",
        "next_video_channel_id": "civic-reports",
        "slate_items_json": json.dumps(slate_items),
        "trajectory_json": json.dumps({"order": slate_items}),
        "watched_vids_json": json.dumps(
            [entry["id"] for entry in watch_history] + ["dQw4w9WgXcQ"]
        ),
        "watched_detailed_json": json.dumps(watch_history),
        "n_options": len(slate_items),
    }


@pytest.fixture
def clean_example(raw_clean_row: dict) -> dict:
    example = row_to_example(
        raw_clean_row,
        system_prompt=None,
        sol_key=None,
        max_hist=3,
    )
    assert example is not None, "row_to_example returned None for a prompt-ready row"
    example.setdefault("video_id", example.get("gold_id", ""))
    slate_items = example.get("slate_items")
    if isinstance(slate_items, list) and slate_items:
        lines = []
        for idx, item in enumerate(slate_items, 1):
            title = item.get("title") or item.get("id") or "(untitled)"
            vid = item.get("id") or ""
            suffix = f" ({vid})" if vid else ""
            lines.append(f"{idx}. {title}{suffix}")
        example["slate_text"] = "\n".join(lines)
    return example


def test_prompt_builder_handles_clean_example(clean_example: dict) -> None:
    prompt_text = build_user_prompt(clean_example, max_hist=3)
    assert "PROFILE:" in prompt_text
    assert "OPTIONS:" in prompt_text

    document = assemble_document(clean_example, extra_fields=None)
    assert document, "assemble_document should generate non-empty training text"


def test_knn_and_xgb_feature_builders_accept_clean_example(clean_example: dict) -> None:
    train_ds = Dataset.from_list([clean_example])

    docs, labels_id, labels_title = prepare_training_documents(
        train_ds,
        max_train=0,
        seed=0,
        extra_fields=None,
    )
    assert docs and labels_id
    assert labels_id[0], "Expected a canonicalised gold id label"

    xgb_docs, xgb_ids, xgb_titles = prepare_prompt_documents(
        train_ds,
        max_train=0,
        seed=0,
        extra_fields=None,
    )
    assert xgb_docs == docs
    assert xgb_ids == labels_id
    assert xgb_titles == labels_title


def test_gpt4o_conversation_builder_accepts_clean_example(clean_example: dict) -> None:
    record = make_conversation_record(clean_example)
    assert record["prompt"][0]["role"] == "system"
    assert record["prompt"][1]["role"] == "user"
    assert record["n_options"] == clean_example["n_options"]
    assert record["gold_index"] == clean_example["gold_index"]


def test_clean_example_includes_required_open_r1_columns(clean_example: dict) -> None:
    pytest.importorskip("torch", reason="open_r1 modules require torch")
    pytest.importorskip("transformers", reason="open_r1 modules require transformers")
    try:
        from open_r1.grail import PASSTHROUGH_FIELDS as GRAIL_PASSTHROUGH_FIELDS  # type: ignore
        from open_r1.grail import TRAIN_KEEP_COLUMNS  # type: ignore
        from open_r1.grpo import KEEP_COLUMNS as GRPO_KEEP_COLUMNS  # type: ignore
        from open_r1.grpo import PASSTHROUGH_KEYS as GRPO_PASSTHROUGH_KEYS  # type: ignore
    except ImportError as exc:
        pytest.skip(f"open_r1 dependencies unavailable: {exc}")

    missing_grail = TRAIN_KEEP_COLUMNS - clean_example.keys()
    assert not missing_grail, f"Grail training columns missing: {sorted(missing_grail)}"

    missing_grpo = GRPO_KEEP_COLUMNS - clean_example.keys()
    assert not missing_grpo, f"GRPO columns missing: {sorted(missing_grpo)}"

    for passthrough_key in sorted(GRAIL_PASSTHROUGH_FIELDS.union(GRPO_PASSTHROUGH_KEYS)):
        assert passthrough_key in clean_example, f"Missing passthrough field {passthrough_key}"
