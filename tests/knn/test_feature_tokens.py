"""Tests for selection-aware token augmentation in knn.features."""

from __future__ import annotations

from knn.core.data import SOLUTION_COLUMN
from knn.core.features import (
    CandidateMetadata,
    candidate_feature_tokens,
    selection_feature_tokens,
)


def _has_token(tokens: list[str], prefix: str) -> bool:
    return any(token.startswith(prefix) for token in tokens)


def test_selection_feature_tokens_include_slot_and_video() -> None:
    """Selection tokens should encode slot, issue, and identifiers."""
    example = {
        "gold_index": 2,
        "issue": "Minimum Wage",
        "participant_study": "study2",
        SOLUTION_COLUMN: "abc123XYZ00",
    }
    candidates = [
        CandidateMetadata(slot=1, title="Option A", video_id="foo11111111"),
        CandidateMetadata(
            slot=2,
            title="Option B",
            video_id="abc123XYZ00",
            channel_title="Channel Name",
            channel_id="channel-42",
        ),
    ]
    tokens = selection_feature_tokens(example, candidates)
    assert "slot_token_2" in tokens
    assert _has_token(tokens, "video_token_")
    assert _has_token(tokens, "title_token_")
    assert "issue_token_minimumwage" in tokens
    assert "study_token_study2" in tokens


def test_candidate_feature_tokens_align_with_selection_tokens() -> None:
    """Candidate tokens should mirror the selection token scheme."""
    candidate = CandidateMetadata(
        slot=3,
        title="Candidate Title",
        video_id="vid12345678",
        channel_title="Some Channel",
    )
    tokens = candidate_feature_tokens(candidate, option_count=4)
    assert "options_token_4" in tokens
    assert "slot_token_3" in tokens
    assert _has_token(tokens, "video_token_")
    assert _has_token(tokens, "title_token_")
