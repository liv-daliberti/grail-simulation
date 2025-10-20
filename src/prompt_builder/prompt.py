"""Top-level prompt construction entry points."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

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

    current_title = clean_text(ex.get("current_video_title"), limit=160)
    current_id = clean_text(ex.get("current_video_id"))
    current_channel = clean_text(ex.get("current_video_channel") or ex.get("current_video_channel_title"))
    current_line_parts: List[str] = []
    if current_title:
        current_line_parts.append(current_title)
    if current_channel:
        current_line_parts.append(f"channel: {current_channel}")
    if show_ids and current_id:
        current_line_parts.append(f"id: {current_id}")
    elif not current_title and current_id:
        current_line_parts.append(f"id: {current_id}")
    current_section: List[str] = []
    if current_line_parts:
        current_section = ["", "CURRENT VIDEO:", " — ".join(current_line_parts)]

    vids = as_list_json(ex.get("watched_vids_json"))
    det = as_list_json(ex.get("watched_detailed_json"))

    current_id_clean = current_id
    cur_idx = None
    if current_id_clean:
        cur_idx = _last_index(vids, current_id_clean)
        if cur_idx is None and isinstance(det, list):
            for j in range(len(det) - 1, -1, -1):
                try:
                    if isinstance(det[j], dict) and clean_text(det[j].get("id")) == current_id_clean:
                        cur_idx = j
                        break
                except Exception:
                    continue
    if cur_idx is None and isinstance(vids, list) and vids:
        cur_idx = len(vids) - 1

    prior: List[dict] = []
    if isinstance(det, list) and cur_idx is not None and cur_idx > 0:
        prior = det[:cur_idx]

    if prior:
        lines.append("")
        lines.append("HISTORY (most recent first):")
        limit = max_hist if max_hist and max_hist > 0 else len(prior)
        recent = list(reversed(prior))[:limit]
        for idx, record in enumerate(recent, 1):
            if not isinstance(record, dict):
                continue
            title = clean_text(
                record.get("title")
                or record.get("name")
                or record.get("video_title"),
                limit=160,
            )
            rid = clean_text(record.get("id"))
            channel = clean_text(record.get("channel_title") or record.get("channel"))
            ws = secs(record.get("watch_seconds"))
            tl = secs(record.get("total_length"))
            descriptor = f"[{ws}/{tl}] {title or '(untitled)'}"
            extras: List[str] = []
            if channel:
                extras.append(f"channel: {channel}")
            if show_ids and rid:
                extras.append(f"id: {rid}")
            if extras:
                descriptor = f"{descriptor} — {', '.join(extras)}"
            lines.append(f"{idx}. {descriptor}")

    if current_section:
        lines.extend(current_section)

    items = as_list_json(ex.get("slate_items_json"))
    lines.append("")
    lines.append("OPTIONS:")
    if items:
        for i, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            title = clean_text(item.get("title"), limit=160)
            option_id = clean_text(item.get("id"))
            if not title and option_id:
                title = option_id
            channel = clean_text(item.get("channel_title") or item.get("channel") or item.get("channel_name"))
            duration_raw = item.get("length_seconds") or item.get("duration_seconds") or item.get("duration")
            duration_text = ""
            try:
                if duration_raw is not None and str(duration_raw).strip():
                    duration_val = float(duration_raw)
                    if duration_val > 0:
                        duration_text = f"{int(round(duration_val))}s"
            except Exception:
                duration_text = ""
            parts = [title or "(untitled)"]
            if channel:
                parts.append(f"channel: {channel}")
            if duration_text:
                parts.append(f"duration: {duration_text}")
            stat_labels = [
                ("view_count", "views"),
                ("like_count", "likes"),
                ("dislike_count", "dislikes"),
                ("favorite_count", "favorites"),
                ("comment_count", "comments"),
            ]
            stat_parts: List[str] = []
            for key, label in stat_labels:
                formatted = format_count(item.get(key))
                if formatted is not None:
                    stat_parts.append(f"{label}: {formatted}")
            if stat_parts:
                parts.extend(stat_parts)
            if show_ids and option_id:
                parts.append(f"id: {option_id}")
            elif not title and option_id:
                parts.append(f"id: {option_id}")
            lines.append(f"{i}. {' — '.join(parts)}")
    else:
        lines.append("(no options provided)")

    return "\n".join(lines)


__all__ = [
    "build_user_prompt",
    "synthesize_viewer_sentence",
]
