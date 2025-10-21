"""Unit tests for the ``prompt_builder`` package."""

from __future__ import annotations

import json

import pytest

from prompt_builder import build_user_prompt, render_profile


pytestmark = pytest.mark.prompt_builder


@pytest.fixture
def sample_example() -> dict:
    return {
        "viewer_profile_sentence": "",
        "age": 45,
        "gender": "woman",
        "race": "Black or African American",
        "city": "Chicago",
        "state": "Illinois",
        "education": "Bachelor's degree",
        "college": "yes",
        "income": "$75,000 - $99,999",
        "employment_status": "employed full time",
        "occupation": "Engineer",
        "marital_status": "married",
        "children_in_house": "yes",
        "household_size": 4,
        "religion": "Protestant",
        "relig_attend": "weekly",
        "veteran": "no",
        "user_language": "en",
        "pid1": "Democrat",
        "ideo1": "liberal",
        "vote_2020": "Joe Biden",
        "trump_approve": "strongly disapprove",
        "biden_approve": "somewhat approve",
        "freq_youtube": "3",
        "binge_youtube": "no",
        "q8": "Vox",
        "watched_vids_json": json.dumps(["abc", "def", "ghi"]),
        "watched_detailed_json": json.dumps(
            [
                {"id": "abc", "title": "Old Clip", "watch_seconds": 30, "total_length": 120},
                {"id": "def", "title": "Another Clip", "watch_seconds": 40, "total_length": 200},
                {"id": "ghi", "title": "Current Clip", "watch_seconds": 50, "total_length": 210},
            ]
        ),
        "current_video_id": "ghi",
        "current_video_title": "Current Clip",
        "current_video_channel": "News Channel",
        "slate_items_json": json.dumps(
            [
                {
                    "id": "opt1",
                    "title": "Recommended Video",
                    "channel_title": "Channel One",
                    "length_seconds": 180,
                    "view_count": 123456,
                },
                {
                    "id": "opt2",
                    "title": "Another Recommendation",
                    "channel_title": "Channel Two",
                    "duration_seconds": 95,
                },
            ]
        ),
    }


pytestmark = pytest.mark.prompt_builder

def test_render_profile_produces_sentences(sample_example: dict) -> None:
    profile = render_profile(sample_example)
    assert profile.sentences, "Expected at least one profile sentence"
    first_sentence = profile.sentences[0]
    assert "45-year-old" in first_sentence
    assert any("They live in" in sentence for sentence in profile.sentences)
    assert not profile.viewer_placeholder


def test_build_user_prompt_structure(sample_example: dict) -> None:
    prompt_text = build_user_prompt(sample_example, max_hist=2)
    assert "PROFILE:" in prompt_text
    assert "CURRENT VIDEO:" in prompt_text
    assert "RECENTLY WATCHED (NEWEST LAST):" in prompt_text
    assert "OPTIONS:" in prompt_text
    assert "Recommended Video" in prompt_text


def test_recently_watched_duration_has_seconds(sample_example: dict) -> None:
    prompt_text = build_user_prompt(sample_example, max_hist=2)
    lines = prompt_text.splitlines()
    try:
        start = lines.index("RECENTLY WATCHED (NEWEST LAST):")
    except ValueError as exc:  # pragma: no cover - defensive
        raise AssertionError("Recently watched section missing") from exc
    history_lines = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "(no recently watched videos available)":
            continue
        if stripped == "OPTIONS:":
            break
        history_lines.append(stripped)
    assert history_lines, "Expected watch history entries to be present"
    assert all("?/" not in line for line in history_lines)
    assert all("/?" not in line for line in history_lines)


def test_option_engagement_summary_lists_metrics(sample_example: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from prompt_builder import prompt as prompt_module

    monkeypatch.setattr(
        prompt_module,
        "lookup_video_stats",
        lambda video_id: {"like_count": 42, "comment_count": 7, "share_count": 3},
    )
    prompt_text = build_user_prompt(sample_example, max_hist=1)
    option_lines = [line for line in prompt_text.splitlines() if line.strip().startswith("likes:")]
    assert option_lines, "Expected engagement summary line under options"
    assert all("likes: 42" in line for line in option_lines)
    assert all("comments: 7" in line for line in option_lines)
    assert all("shares: 3" in line for line in option_lines)
