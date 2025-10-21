"""Top-level prompt construction entry points."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .formatters import clean_text
from .parsers import as_list_json, format_count, secs
from .profiles import ProfileRender, render_profile, synthesize_viewer_sentence


def _last_index(xs: Any, val: Any) -> Optional[int]:
    """Return the last index of ``val`` within ``xs`` when ``xs`` is a list.

    :param xs: Sequence potentially containing ``val``.
    :param val: Value to locate.
    :returns: Zero-based index or ``None`` when not found.
    """
    if not isinstance(xs, list) or val is None:
        return None
    idx = None
    for i, v in enumerate(xs):
        if v == val:
            idx = i
    return idx


def _render_profile_text(profile: ProfileRender) -> str:
    """Join profile sentences into a single descriptive paragraph.

    :param profile: Rendered profile object.
    :returns: Combined profile text or a fallback string.
    """
    profile_text = " ".join(sentence for sentence in profile.sentences if sentence)
    if not profile_text:
        return "Profile information is unavailable."
    return profile_text


def build_user_prompt(ex: Dict[str, Any], max_hist: int = 12) -> str:
    """
    Assemble the user prompt used for recommendation tasks.

    :param ex: Cleaned interaction row with profile, history, and slate fields.
    :type ex: Dict[str, Any]
    :param max_hist: Maximum number of prior videos to include in the history section.
    :type max_hist: int
    :returns: Multi-line prompt describing the viewer, viewing context, and options.
    :rtype: str
    """

    show_ids = os.getenv("GRAIL_SHOW_IDS", "0") == "1"
    lines: List[str] = []

    profile = render_profile(ex)
    lines.append("PROFILE:")
    lines.append(_render_profile_text(profile))

    current_section = _current_video_section(ex, show_ids)
    if current_section:
        lines.extend(current_section)

    lines.extend(_recently_watched_section(ex, show_ids, max_hist))

    lines.extend(_options_section(ex, show_ids))

    return "\n".join(lines)


def _current_video_section(ex: Dict[str, Any], show_ids: bool) -> List[str]:
    """
    Build the current video subsection for the prompt.

    :param ex: Interaction row containing information about the active video.
    :type ex: Dict[str, Any]
    :param show_ids: Flag controlling whether identifiers should always be emitted.
    :type show_ids: bool
    :returns: Lines describing the active video, or an empty list when unavailable.
    :rtype: List[str]
    """

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


def _recently_watched_section(ex: Dict[str, Any], show_ids: bool, max_hist: int) -> List[str]:
    """
    Build the recently watched subsection for the prompt.

    :param ex: Interaction row containing history-related fields.
    :type ex: Dict[str, Any]
    :param show_ids: Flag controlling whether identifiers should always be emitted.
    :type show_ids: bool
    :param max_hist: Maximum number of historical items to include.
    :type max_hist: int
    :returns: Lines summarising previously watched videos, newest entry last.
    :rtype: List[str]
    """

    descriptors = _recently_watched_descriptors(ex, show_ids, max_hist)
    section: List[str] = ["", "RECENTLY WATCHED (NEWEST LAST):"]
    if descriptors:
        section.extend(f"{idx}. {entry}" for idx, entry in enumerate(descriptors, 1))
    else:
        section.append("(no recently watched videos available)")
    return section


def _recently_watched_descriptors(ex: Dict[str, Any], show_ids: bool, max_hist: int) -> List[str]:
    """Return formatted descriptors for recently watched videos.

    :param ex: Interaction row containing history arrays.
    :param show_ids: Whether to include video identifiers.
    :param max_hist: Maximum number of entries to inspect.
    :returns: List of descriptor strings ordered oldest → newest.
    """
    prior = _prior_entries(ex)
    if not prior:
        return []
    limit = max_hist if max_hist and max_hist > 0 else len(prior)
    recent = prior[-limit:]
    descriptors: List[str] = []
    for record in recent:
        if isinstance(record, dict):
            descriptors.append(_watched_descriptor(record, show_ids))
    return descriptors


def _prior_entries(ex: Dict[str, Any]) -> List[dict]:
    """Return watch-history entries preceding the current video.

    :param ex: Interaction row containing history payloads.
    :returns: List of entry dictionaries ordered chronologically.
    """
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
    if isinstance(detailed, list) and cur_idx is not None and cur_idx > 0:
        return detailed[:cur_idx]
    return []


def _watched_descriptor(record: Dict[str, Any], show_ids: bool) -> str:
    """Return a formatted descriptor summarising a watched video.

    :param record: Dictionary containing watch metadata.
    :param show_ids: Whether to include identifiers in the descriptor.
    :returns: Descriptor string describing watch progress and metadata.
    """
    title = clean_text(
        record.get("title") or record.get("name") or record.get("video_title"),
        limit=160,
    )
    rid = clean_text(record.get("id"))
    channel = clean_text(record.get("channel_title") or record.get("channel"))
    watch_seconds_value = _extract_duration_seconds(
        record,
        (
            "watch_seconds",
            "watch_duration",
            "watch_time",
            "watch_ms",
        ),
    )
    total_length_value = _extract_duration_seconds(
        record,
        (
            "total_length",
            "total_duration",
            "duration_seconds",
            "length_seconds",
            "total_length_ms",
            "duration",
        ),
    )
    watch_seconds = secs(watch_seconds_value) if watch_seconds_value is not None else "?"
    total_length = secs(total_length_value) if total_length_value is not None else "?"
    descriptor = f"[{watch_seconds}/{total_length}] {title or '(untitled)'}"
    extras: List[str] = []
    if channel:
        extras.append(f"channel: {channel}")
    if show_ids and rid:
        extras.append(f"id: {rid}")
    if extras:
        descriptor = f"{descriptor} — {', '.join(extras)}"
    return descriptor


def _extract_duration_seconds(
    record: Dict[str, Any],
    keys: Sequence[str],
) -> Optional[float]:
    """Return the first positive duration found in ``record`` for ``keys``."""

    for key in keys:
        if key not in record:
            continue
        raw_value = record.get(key)
        if raw_value is None:
            continue
        try:
            duration = float(raw_value)
        except (TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        if key.endswith("_ms") and duration > 1000:
            duration /= 1000.0
        return duration
    return None


def _options_section(ex: Dict[str, Any], show_ids: bool) -> List[str]:
    """
    Build the options subsection for the prompt.

    :param ex: Interaction row containing candidate slate information.
    :type ex: Dict[str, Any]
    :param show_ids: Flag controlling whether identifiers should always be emitted.
    :type show_ids: bool
    :returns: Lines describing available recommendation candidates.
    :rtype: List[str]
    """

    section: List[str] = ["", "OPTIONS:"]
    items = as_list_json(ex.get("slate_items_json"))
    if not items:
        section.append("(no options provided)")
        return section
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        section.extend(_option_lines(index, item, show_ids))
    if len(section) == 2:  # only header produced lines
        section.append("(no options provided)")
    return section


def _option_lines(position: int, item: Dict[str, Any], show_ids: bool) -> List[str]:
    """Render a single recommendation option for inclusion in the prompt.

    :param position: 1-based index of the option.
    :param item: Dictionary containing option metadata.
    :param show_ids: Whether to include identifiers in the output.
    :returns: Formatted option lines (primary line plus engagement summary).
    """
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
    lines = [f"{position}. {' — '.join(parts)}"]
    engagement_summary = _option_engagement_summary(item)
    if engagement_summary:
        lines.append(f"   {engagement_summary}")
    return lines


def _format_duration(item: Dict[str, Any]) -> str:
    """Return a formatted duration string for a slate item.

    :param item: Dictionary containing duration metadata.
    :returns: Seconds string (e.g. ``\"120s\"``) or empty string when unavailable.
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


def _option_stats(item: Dict[str, Any]) -> Iterable[str]:
    """Yield formatted statistics (views, dislikes, etc.) for the option.

    :param item: Dictionary containing count fields.
    :returns: Iterator over formatted ``label: value`` strings.
    """
    stat_labels = [
        ("view_count", "views"),
        ("dislike_count", "dislikes"),
        ("favorite_count", "favorites"),
    ]
    for key, label in stat_labels:
        formatted = format_count(item.get(key))
        if formatted is not None:
            yield f"{label}: {formatted}"


def _option_engagement_summary(item: Dict[str, Any]) -> str:
    """Return a summary line covering likes, comments, and shares."""

    def first_present(keys: Sequence[str]) -> Any:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            return value
        return None

    metrics = [
        ("likes", ("like_count", "likes", "likeCount")),
        ("comments", ("comment_count", "comments", "commentCount")),
        ("shares", ("share_count", "shares", "shareCount")),
    ]
    parts: List[str] = []
    for label, keys in metrics:
        formatted = format_count(first_present(keys))
        parts.append(f"{label}: {formatted if formatted is not None else 'n/a'}")
    return " — ".join(parts)


__all__ = [
    "build_user_prompt",
    "synthesize_viewer_sentence",
]
