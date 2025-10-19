"""Shared constants for prompt building and stats."""

from __future__ import annotations

from typing import Set

from clean_data.helpers import REQUIRED_FOR_GRPO as REQUIRED_PROMPT_COLUMNS

PASSTHROUGH_COLUMNS: Set[str] = {
    "issue",
    "rating_copy",
    "rating_index",
    "rating_video_id",
    "urlid",
    "topic_id",
    "session_id",
    "step_index",
    "display_step",
    "display_order_key",
    "current_video_raw_id",
    "current_video_channel",
    "current_video_channel_id",
    "next_video_id",
    "next_video_raw_id",
    "next_video_title",
    "next_video_channel",
    "next_video_channel_id",
    "percent_visible",
    "session_finished",
    "start_time_ms",
    "end_time_ms",
    "trajectory_json",
    "issue_source",
    "issue_detail",
    "slate_source",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are choosing EXACTLY ONE item from a short slate for a specific viewer.\n"
    "Think briefly in <think>…</think>, then output ONLY the option NUMBER (1..N) "
    "inside <answer>…</answer>.\n"
    "Format (STRICT): <think>…</think><answer>3</answer>"
)

YOUTUBE_FREQ_MAP = {
    "0": "rarely",
    "1": "occasionally",
    "2": "a few times a month",
    "3": "weekly",
    "4": "several times a week",
    "5": "daily",
}

__all__ = [
    "PASSTHROUGH_COLUMNS",
    "DEFAULT_SYSTEM_PROMPT",
    "YOUTUBE_FREQ_MAP",
    "REQUIRED_PROMPT_COLUMNS",
]
