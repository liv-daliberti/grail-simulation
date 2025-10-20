"""Top-level prompt construction entry points."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

from .formatters import clean_text
from .parsers import as_list_json, format_count, secs
from .profiles import ProfileRender, render_profile, synthesize_viewer_sentence


def _last_index(xs: Any, val: Any) -> Optional[int]:
    if not isinstance(xs, list) or val is None:
        return None
    idx = None
    for i, v in enumerate(xs):
        if v == val:
            idx = i
    return idx


def _render_profile_text(profile: ProfileRender) -> str:
    profile_text = " ".join(sentence for sentence in profile.sentences if sentence)
    if not profile_text:
        return "Profile information is unavailable."
    return profile_text


def build_user_prompt(ex: Dict[str, Any], max_hist: int = 12) -> str:
    """Build the user prompt for a cleaned interaction row."""

    show_ids = os.getenv("GRAIL_SHOW_IDS", "0") == "1"
    lines: List[str] = []

    profile = render_profile(ex)
    lines.append("PROFILE:")
    lines.append(_render_profile_text(profile))

    history_section = _history_section(ex, show_ids, max_hist)
    if history_section:
        lines.extend(history_section)

    current_section = _current_video_section(ex, show_ids)
    if current_section:
        lines.extend(current_section)

    lines.extend(_options_section(ex, show_ids))

    return "\n".join(lines)


def _current_video_section(ex: Dict[str, Any], show_ids: bool) -> List[str]:
    """Return the current video lines, prefixed with a blank line."""

    title = clean_text(ex.get("current_video_title"), limit=160)
    current_id = clean_text(ex.get("current_video_id"))
    channel = clean_text(ex.get("current_video_channel") or ex.get("current_video_channel_title"))
    parts: List[str] = []
    if title:
        parts.append(title)
    if channel:
        parts.append(f"channel: {channel}")
    if current_id:
        if show_ids or not title:
            parts.append(f"id: {current_id}")
    if not parts:
        return []
    return ["", "CURRENT VIDEO:", " — ".join(parts)]


def _history_section(ex: Dict[str, Any], show_ids: bool, max_hist: int) -> List[str]:
    """Return the viewing history section (most recent first)."""

    vids = as_list_json(ex.get("watched_vids_json"))
    detailed = as_list_json(ex.get("watched_detailed_json"))
    current_id = clean_text(ex.get("current_video_id"))
    cur_idx = None
    if current_id:
        cur_idx = _last_index(vids, current_id)
        if cur_idx is None and isinstance(detailed, list):
            for index in range(len(detailed) - 1, -1, -1):
                entry = detailed[index]
                if isinstance(entry, dict) and clean_text(entry.get("id")) == current_id:
                    cur_idx = index
                    break
    if cur_idx is None and isinstance(vids, list) and vids:
        cur_idx = len(vids) - 1
    prior: List[dict] = []
    if isinstance(detailed, list) and cur_idx is not None and cur_idx > 0:
        prior = detailed[:cur_idx]
    if not prior:
        return []

    section: List[str] = ["", "HISTORY (most recent first):"]
    limit = max_hist if max_hist and max_hist > 0 else len(prior)
    recent = list(reversed(prior))[:limit]
    for idx, record in enumerate(recent, 1):
        if not isinstance(record, dict):
            continue
        descriptor = _history_descriptor(record, show_ids)
        section.append(f"{idx}. {descriptor}")
    return section


def _history_descriptor(record: Dict[str, Any], show_ids: bool) -> str:
    title = clean_text(
        record.get("title")
        or record.get("name")
        or record.get("video_title"),
        limit=160,
    )
    rid = clean_text(record.get("id"))
    channel = clean_text(record.get("channel_title") or record.get("channel"))
    watch_seconds = secs(record.get("watch_seconds"))
    total_length = secs(record.get("total_length"))
    descriptor = f"[{watch_seconds}/{total_length}] {title or '(untitled)'}"
    extras: List[str] = []
    if channel:
        extras.append(f"channel: {channel}")
    if show_ids and rid:
        extras.append(f"id: {rid}")
    if extras:
        descriptor = f"{descriptor} — {', '.join(extras)}"
    return descriptor


def _options_section(ex: Dict[str, Any], show_ids: bool) -> List[str]:
    """Return the options section, always including the header."""

    section: List[str] = ["", "OPTIONS:"]
    items = as_list_json(ex.get("slate_items_json"))
    if not items:
        section.append("(no options provided)")
        return section
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        section.append(_option_line(index, item, show_ids))
    if len(section) == 2:  # only header produced lines
        section.append("(no options provided)")
    return section


def _option_line(position: int, item: Dict[str, Any], show_ids: bool) -> str:
    title = clean_text(item.get("title"), limit=160)
    option_id = clean_text(item.get("id"))
    if not title and option_id:
        title = option_id
    channel = clean_text(
        item.get("channel_title")
        or item.get("channel")
        or item.get("channel_name")
    )
    duration_text = _format_duration(item)
    parts: List[str] = [title or "(untitled)"]
    if channel:
        parts.append(f"channel: {channel}")
    if duration_text:
        parts.append(f"duration: {duration_text}")
    parts.extend(_option_stats(item))
    if option_id and (show_ids or not title):
        parts.append(f"id: {option_id}")
    return f"{position}. {' — '.join(parts)}"


def _format_duration(item: Dict[str, Any]) -> str:
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


def _option_stats(item: Dict[str, Any]) -> Iterable[str]:
    stat_labels = [
        ("view_count", "views"),
        ("like_count", "likes"),
        ("dislike_count", "dislikes"),
        ("favorite_count", "favorites"),
        ("comment_count", "comments"),
    ]
    for key, label in stat_labels:
        formatted = format_count(item.get(key))
        if formatted is not None:
            yield f"{label}: {formatted}"


__all__ = [
    "build_user_prompt",
    "synthesize_viewer_sentence",
]
