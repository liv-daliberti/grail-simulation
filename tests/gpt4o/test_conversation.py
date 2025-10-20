"""Focused unit tests for high-value helpers in :mod:`gpt4o.conversation`."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gpt4o import conversation

pytestmark = pytest.mark.gpt4o


class _StubTitleResolver:
    def __init__(self) -> None:
        self.mapping: dict[str, str] = {
            "dQw4w9WgXcQ": "Rick Astley - Never Gonna Give You Up",
            "oldoldold01": "Earlier Clip",
            "nowplaying01": "Now Playing Title",
        }

    def resolve(self, video_id: str | None) -> str | None:
        if not video_id:
            return None
        return self.mapping.get(video_id)


@pytest.fixture(autouse=True)
def stub_title_resolver(monkeypatch: pytest.MonkeyPatch) -> _StubTitleResolver:
    resolver = _StubTitleResolver()
    monkeypatch.setattr(conversation, "TITLE_RESOLVER", resolver)
    return resolver


def test_pick_case_insensitive_returns_matching_value() -> None:
    record = {"Name": "Ada", "AGE": "36"}
    assert conversation.pick_case_insensitive(record, "name", "age") == "Ada"
    assert conversation.pick_case_insensitive(record, "missing") is None
    assert conversation.pick_case_insensitive("not-a-dict", "name") is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("example", "expected"),
    [
        ({"income": "  $50,000 - $74,999 "}, "$50,000 - $74,999"),
        ({"other": "household income around $120,000"}, "household income around $120,000"),
        ({"income_gt50k": 1}, ">$50k household income"),
        ({"income_gt50k": 0}, "≤$50k household income"),
    ],
)
def test_extract_income_covers_primary_and_fallback_paths(example: dict, expected: str) -> None:
    assert conversation.extract_income(example) == expected


def test_extract_party_prefers_text_then_numeric_scale() -> None:
    assert conversation.extract_party({"pid": "Strong Democrat"}) == "Democratic"
    numeric_party = {"pid1": 2, "pid2": 2, "pid3": 2}
    assert conversation.extract_party(numeric_party) == "Democratic-leaning"
    assert conversation.extract_party({}) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", "Extremely liberal"),
        ("4", "Moderate"),
        ("6.7", "Extremely conservative"),
        ("Libertarian", "Liberal"),
        (None, None),
        ("", None),
    ],
)
def test_format_ideology_normalises_numeric_and_string_inputs(raw: Any, expected: str | None) -> None:
    assert conversation.format_ideology(raw) == expected


def test_extract_marital_status_detects_keywords() -> None:
    assert conversation.extract_marital_status({"q18": "Married"}) == "Married"
    assert conversation.extract_marital_status({"marital_status": "Living with partner"}) == "Living with partner"
    assert conversation.extract_marital_status({}) is None


def test_extract_race_prefers_boolean_hints_then_text() -> None:
    example = {"white": 1, "black": 0}
    assert conversation.extract_race(example) == "White"
    example = {"q26": "African American"}
    assert conversation.extract_race(example) == "Black"
    assert conversation.extract_race({}) is None


def test_humanise_profile_composes_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    example = {
        "age": 42,
        "female": 1,
        "male": 0,
        "white": 1,
        "income": "$75,000 - $99,999",
        "ideo": "2",
        "freq_youtube": "4",
        "college": 1,
        "pid": "Democrat",
        "marital_status": "Married",
    }
    sentence = conversation.humanise_profile(example)
    assert "42-year-old" in sentence
    assert "white woman" in sentence.lower()
    assert "democratic-leaning" in sentence or "democratic" in sentence.lower()
    assert "watches YouTube several times a week" in sentence


def test_build_profile_block_respects_env_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAIL_PROFILE_COLS", "age,city")
    example = {"age": 30, "city": "Chicago", "ideo": "3", "pid": "Independent"}
    block = conversation.build_profile_block(example)
    assert "race:" not in block
    assert "age: 30" in block
    assert "city: Chicago" in block


def test_extract_now_watching_prefers_explicit_fields(stub_title_resolver: _StubTitleResolver) -> None:
    example = {
        "video_id": "dQw4w9WgXcQ",
        "current_video_title": "Supplied Title",
    }
    title, video_id = conversation._extract_now_watching(example)  # pylint: disable=protected-access
    assert title == "Supplied Title"
    assert video_id == "dQw4w9WgXcQ"


def test_extract_now_watching_falls_back_to_trajectory(stub_title_resolver: _StubTitleResolver) -> None:
    stub_title_resolver.mapping["trajvideo01"] = "Trajectory Title"
    example = {
        "trajectory_json": json.dumps(
            {
                "current": {"video_id": "trajvideo01"},
            }
        )
    }
    title, video_id = conversation._extract_now_watching(example)  # pylint: disable=protected-access
    assert title == "Trajectory Title"
    assert video_id == "trajvideo01"


def test_get_history_pointer_parses_variants() -> None:
    example = {"trajectory_json": json.dumps({"current_index": "2", "current_end_ms": "1234"})}
    pointer, end_ms = conversation._get_history_pointer(example)  # pylint: disable=protected-access
    assert pointer == 2
    assert end_ms == 1234.0


def test_extract_history_filters_by_pointer() -> None:
    trajectory = json.dumps(
        {
            "order": [
                {"idx": 0, "title": "First", "video_id": "aaa11111111", "watch_seconds": 30, "total_length": 100},
                {"idx": 1, "title": "Second", "video_id": "bbb22222222", "watch_seconds": 20, "total_length": 120},
            ]
        }
    )
    example = {"trajectory_json": trajectory}
    extracted = conversation._extract_history(example, up_to_idx=1)  # pylint: disable=protected-access
    assert len(extracted) == 1
    assert extracted[0]["title"] == "First"


def test_format_history_lines_renders_expected_structure() -> None:
    sequence = [
        {"idx": 3, "watch_seconds": 15, "total_length": 120, "title": "Sample"},
        {"title": "No Metrics"},
    ]
    lines = conversation._format_history_lines(sequence)  # pylint: disable=protected-access
    assert lines[0].startswith("- [3 • 15s/120s] Sample")
    assert lines[1] == "- No Metrics"


def test_make_conversation_record_integrates_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAIL_PROFILE_COLS", "age,gender")
    monkeypatch.setenv("GRAIL_MAX_HISTORY", "1")
    example = {
        "age": 35,
        "female": 1,
        "male": 0,
        "white": 1,
        "income": "$50,000 - $74,999",
        "ideo": "3",
        "pid": "Democrat",
        "freq_youtube": "2",
        "college": 1,
        "current_video_title": "Now Playing Title",
        "current_video_id": "nowplaying01",
        "trajectory_json": json.dumps(
            {
                "current_index": 1,
                "order": [
                    {"idx": 0, "title": "Previous Video", "video_id": "oldoldold01", "watch_seconds": 40, "total": 120},
                    {"idx": 1, "title": "Now Playing Title", "video_id": "nowplaying01", "watch_seconds": 10, "total": 200},
                ],
            }
        ),
        "slate_text": "1. dQw4w9WgXcQ",
        "video_id": "dQw4w9WgXcQ",
        "state_text": "Prior context about the viewer.",
        "video_index": "3",
    }

    record = conversation.make_conversation_record(example)
    assert record["gold_index"] == 1
    assert record["n_options"] == 1
    assert record["metadata"]["now_playing"].startswith("Now Playing Title")
    user_prompt = record["prompt"][1]["content"]
    assert "Viewer: " in user_prompt
    assert "CONTEXT: Prior context about the viewer." in user_prompt
    assert "OPTIONS:" in user_prompt
    assert "1. Rick Astley - Never Gonna Give You Up" in user_prompt
