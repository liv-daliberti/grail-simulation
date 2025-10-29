#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for rendering recommendation option lines within prompts."""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Sequence

from .formatters import clean_text
from .parsers import as_list_json, format_count, is_nanlike
from .video_stats import lookup_video_stats as _default_lookup_video_stats


def options_lines(ex: Dict[str, Any], show_ids: bool) -> List[str]:
    """
    Render recommendation options as numbered lines.

    :param ex: Dataset example containing ``slate_items_json`` data.
    :type ex: Dict[str, Any]
    :param show_ids: Whether to display YouTube ids alongside titles.
    :type show_ids: bool
    :returns: List of formatted option strings ready for inclusion in the prompt.
    :rtype: List[str]
    """

    items = as_list_json(ex.get("slate_items_json"))
    if not items:
        return []
    lines: List[str] = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        line = _option_line(index, item, show_ids)
        if line:
            lines.append(line)
    return lines


def _resolve_lookup_video_stats() -> Callable[[str], Dict[str, Any]]:
    """
    Return the active ``lookup_video_stats`` implementation.

    The prompt module re-exports this helper so tests can monkeypatch
    ``prompt_builder.prompt.lookup_video_stats``. Resolving it dynamically keeps
    those patches in sync without introducing an import cycle.
    """

    prompt_module = sys.modules.get("prompt_builder.prompt")
    if prompt_module is not None:
        candidate = getattr(prompt_module, "lookup_video_stats", None)
        if callable(candidate):
            return candidate  # type: ignore[return-value]
    return _default_lookup_video_stats


def _option_line(position: int, item: Dict[str, Any], show_ids: bool) -> str:
    """
    Render a single entry in the recommendation slate.

    :param position: One-based index of the recommendation.
    :type position: int
    :param item: Slate item metadata including title, id, and channel.
    :type item: Dict[str, Any]
    :param show_ids: Whether to append the YouTube id when present.
    :type show_ids: bool
    :returns: Fully formatted option line ready for inclusion in the prompt.
    :rtype: str
    """

    title = clean_text(item.get("title"), limit=160)
    option_id = clean_text(item.get("id"))
    display_title = title or option_id or "(untitled)"
    channel = clean_text(
        item.get("channel_title")
        or item.get("channel")
        or item.get("channel_name")
    )
    duration_text = _format_duration(item)
    descriptors: List[str] = []
    if channel:
        descriptors.append(channel)
    if duration_text:
        descriptors.append(f"{duration_text} long")
    if option_id and (show_ids or not title):
        descriptors.append(f"id {option_id}")
    stats_lookup = _resolve_lookup_video_stats()
    stats = stats_lookup(option_id) if option_id else {}
    if stats is None:
        stats = {}
    engagement = _option_engagement_summary(item, stats)
    line = f"{position}. {display_title}"
    if descriptors:
        line += f" ({', '.join(descriptors)})"
    if engagement:
        line += f" — Engagement: {engagement}"
    return line


def _format_duration(item: Dict[str, Any]) -> str:
    """
    Convert raw duration metadata into a seconds string.

    :param item: Dictionary containing duration metadata.
    :type item: Dict[str, Any]
    :returns: Seconds string such as ``"120s"`` or an empty string when unavailable.
    :rtype: str
    """

    duration_raw = (
        item.get("length_seconds")
        or item.get("duration_seconds")
        or item.get("duration")
    )
    if duration_raw is None or not str(duration_raw).strip():
        return ""
    try:
        duration_val = float(duration_raw)
    except (TypeError, ValueError):
        return ""
    if duration_val <= 0:
        return ""
    return f"{int(round(duration_val))}s"


def _option_engagement_summary(item: Dict[str, Any], stats: Dict[str, Any]) -> str:
    """
    Summarise engagement metrics for a recommendation option.

    :param item: Slate item metadata containing inline engagement counts.
    :type item: Dict[str, Any]
    :param stats: Lookup statistics sourced from
        :func:`prompt_builder.video_stats.lookup_video_stats`.
    :type stats: Dict[str, Any]
    :returns: Comma-separated phrase describing views, likes, comments, and share
        counts when available.
    :rtype: str
    """

    def first_present(source: Dict[str, Any], keys: Sequence[str]) -> Any:
        """
        Return the first non-missing value from ``source`` matching ``keys``.

        :param source: Mapping containing potential metric values.
        :type source: Dict[str, Any]
        :param keys: Field names to probe in order.
        :type keys: Sequence[str]
        :returns: Raw value when found, otherwise ``None``.
        :rtype: Any
        """

        for key in keys:
            if key not in source:
                continue
            value = source.get(key)
            if is_nanlike(value):
                continue
            text = str(value).strip()
            if not text:
                continue
            return value
        return None

    metrics = [
        (
            "views",
            (
                "view_count",
                "views",
                "viewCount",
                "ViewCount",
                "viewer_count",
                "viewerCount",
                "Viewers",
            ),
            ("view_count", "views", "viewCount", "ViewCount", "viewer_count", "viewerCount"),
        ),
        (
            "likes",
            ("like_count", "likes", "likeCount", "LikeCount", "Likes"),
            ("like_count", "likes", "likeCount", "LikeCount"),
        ),
        (
            "comments",
            ("comment_count", "comments", "commentCount", "CommentCount", "Comments"),
            ("comment_count", "comments", "commentCount", "CommentCount"),
        ),
        (
            "shares",
            ("share_count", "shares", "shareCount", "ShareCount", "Shares"),
            ("share_count", "shares", "shareCount", "ShareCount"),
        ),
    ]
    parts: List[str] = []
    for label, item_keys, stat_keys in metrics:
        value = first_present(item, item_keys)
        if value is None and stats:
            value = first_present(stats, stat_keys)
            if value is None and stat_keys:
                value = stats.get(stat_keys[0])
        formatted = format_count(value)
        if formatted is None:
            continue
        parts.append(f"{label} {formatted}")
    return ", ".join(parts)


__all__ = [
    "options_lines",
]
