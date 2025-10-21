"""Top-level prompt construction entry points."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from .formatters import clean_text, human_join
from .parsers import as_list_json, format_count, is_nanlike
from .profiles import ProfileRender, render_profile, synthesize_viewer_sentence
from .value_maps import format_field_value
from .video_stats import lookup_video_stats


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
    :returns: Natural-language prompt describing the viewer, viewing context, and options.
    :rtype: str
    """

    show_ids = os.getenv("GRAIL_SHOW_IDS", "0") == "1"
    profile = render_profile(ex)
    sections: List[str] = []

    profile_paragraph = _profile_paragraph(profile)
    if profile_paragraph:
        sections.append(f"PROFILE:\n{profile_paragraph}")

    current_sentence = _current_video_sentence(ex, show_ids)
    if current_sentence:
        sections.append(f"CURRENT VIDEO:\n{current_sentence}")

    history_block = _history_block(ex, show_ids, max_hist)
    if history_block:
        sections.append(history_block)

    survey_sentence = _survey_highlights(ex)
    if survey_sentence:
        sections.append(survey_sentence)

    options_block = _options_block(ex, show_ids)
    if options_block:
        sections.append(options_block)

    return "\n\n".join(sections).strip()


def _profile_paragraph(profile: ProfileRender) -> str:
    """Return a paragraph describing the viewer profile."""

    text = _render_profile_text(profile)
    return text.strip()


def _current_video_sentence(ex: Dict[str, Any], show_ids: bool) -> str:
    """Return a sentence describing the currently playing video."""

    title = clean_text(ex.get("current_video_title"), limit=160)
    current_id = clean_text(ex.get("current_video_id"))
    channel = clean_text(
        ex.get("current_video_channel") or ex.get("current_video_channel_title")
    )
    if not (title or channel or current_id):
        return ""
    sentence = "They are currently watching "
    if title:
        sentence += title
    else:
        sentence += "a video"
    if channel:
        sentence += f" from {channel}"
    if current_id and (show_ids or not title):
        sentence += f" (id {current_id})"
    sentence += "."
    return sentence


def _history_sentence(ex: Dict[str, Any], show_ids: bool, max_hist: int) -> str:
    """Return a sentence summarising prior session history."""

    prior_entries = _prior_entries(ex)
    if not prior_entries:
        return ""
    limit = max_hist if max_hist and max_hist > 0 else len(prior_entries)
    recent = prior_entries[-limit:]
    descriptors: List[str] = []
    for record in recent:
        if not isinstance(record, dict):
            continue
        descriptor = _watched_descriptor(record, show_ids)
        if descriptor:
            descriptors.append(descriptor)
    if not descriptors:
        return ""
    if len(descriptors) > 3:
        displayed = descriptors[-3:]
    else:
        displayed = descriptors
    summary = human_join(displayed)
    remaining = len(descriptors) - len(displayed)
    sentence = f"Earlier in the session they watched {summary}."
    if remaining > 0:
        plural = "videos" if remaining > 1 else "video"
        sentence += f" (+{remaining} more {plural})"
    return sentence


def _history_block(ex: Dict[str, Any], show_ids: bool, max_hist: int) -> str:
    """Return the watch-history section with heading and per-video lines."""

    prior_entries = _prior_entries(ex)
    if not prior_entries:
        return "RECENTLY WATCHED (NEWEST LAST):\n(no recently watched videos available)"
    limit = max_hist if max_hist and max_hist > 0 else len(prior_entries)
    recent = prior_entries[-limit:]
    descriptors: List[str] = []
    for record in recent:
        if not isinstance(record, dict):
            continue
        descriptor = _watched_descriptor(record, show_ids)
        if descriptor:
            descriptors.append(descriptor)
    if not descriptors:
        return "RECENTLY WATCHED (NEWEST LAST):\n(no recently watched videos available)"
    return "RECENTLY WATCHED (NEWEST LAST):\n" + "\n".join(descriptors)


SURVEY_HIGHLIGHT_SPECS: Sequence[tuple[str, str]] = (
    ("pid1", "party identification is {value}"),
    ("pid2", "party lean is {value}"),
    ("ideo1", "ideology is {value}"),
    ("pol_interest", "political interest is {value}"),
    ("religpew", "religious affiliation is {value}"),
    ("freq_youtube", "watches YouTube {value}"),
    ("newsint", "{value}"),
    ("participant_study", "participated in {value}"),
)

MIN_WAGE_HIGHLIGHT_SPECS: Sequence[tuple[Sequence[str], str]] = (
    (("minwage_text_w2", "minwage_text_w1"), "preferred minimum wage target is {value}"),
    (("mw_support_w2", "mw_support_w1"), "minimum wage support score is {value}"),
    (("minwage15_w2", "minwage15_w1"), "$15 minimum wage support is {value}"),
)

GUN_HIGHLIGHT_SPECS: Sequence[tuple[Sequence[str], str]] = (
    (("gun_importance",), "gun policy importance is {value}"),
    (("gun_index",), "gun regulation support score is {value}"),
    (("gun_enthusiasm",), "gun enthusiasm is {value}"),
)


def _survey_highlights(ex: Dict[str, Any]) -> str:
    """Return a sentence summarising key survey features."""

    highlights: List[str] = []
    for field, template in SURVEY_HIGHLIGHT_SPECS:
        value = format_field_value(field, ex.get(field))
        if not value:
            continue
        highlights.append(template.format(value=value))

    issue = str(ex.get("issue") or "").strip().lower()
    issue_specs = MIN_WAGE_HIGHLIGHT_SPECS if issue == "minimum_wage" else ()
    if issue == "gun_control":
        issue_specs = GUN_HIGHLIGHT_SPECS
    for fields, template in issue_specs:
        value: Optional[str] = None
        for field in fields:
            candidate = format_field_value(field, ex.get(field))
            if candidate:
                value = candidate
                break
        if value:
            highlights.append(template.format(value=value))

    if not highlights:
        return ""
    return f"Survey highlights: {human_join(highlights)}."


def _options_block(ex: Dict[str, Any], show_ids: bool) -> str:
    """Return a paragraph describing the recommendation slate options."""

    items = as_list_json(ex.get("slate_items_json"))
    if not items:
        return "No recommendation options are available in this slate."
    sentences = [
        _option_sentence(index, item, show_ids)
        for index, item in enumerate(items, 1)
        if isinstance(item, dict)
    ]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        return ""
    return "Today's slate offers:\n" + "\n".join(sentences)


def _option_sentence(position: int, item: Dict[str, Any], show_ids: bool) -> str:
    """Return a human-readable sentence for a specific slate option."""

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
        descriptors.append(f"from {channel}")
    if duration_text:
        descriptors.append(f"{duration_text} long")
    if option_id and (show_ids or not title):
        descriptors.append(f"id {option_id}")
    stats = lookup_video_stats(option_id) if option_id else {}
    if stats is None:
        stats = {}
    engagement = _option_engagement_summary(item, stats)
    sentence = f"Option {position}: {display_title}"
    if descriptors:
        sentence += f" ({', '.join(descriptors)})"
    sentence += "."
    if engagement:
        sentence += f" Engagement: {engagement}."
    return sentence


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
    name = title or "(untitled)"
    descriptors: List[str] = []
    if watch_seconds_value is not None and watch_seconds_value > 0:
        watch_int = int(round(watch_seconds_value))
        if total_length_value is not None and total_length_value > 0:
            total_int = int(round(total_length_value))
            ratio = min(1.0, watch_seconds_value / max(total_length_value, 1e-6))
            descriptors.append(
                f"watched {watch_int}s of {total_int}s ({int(round(ratio * 100))}% complete)"
            )
        else:
            descriptors.append(f"watched for {watch_int}s")
    if channel:
        descriptors.append(f"from {channel}")
    if show_ids and rid:
        descriptors.append(f"id {rid}")
    if descriptors:
        return f"{name} ({', '.join(descriptors)})"
    return name


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


def _option_engagement_summary(item: Dict[str, Any], stats: Dict[str, Any]) -> str:
    """Return a summary line covering likes, comments, and shares."""

    def first_present(source: Dict[str, Any], keys: Sequence[str]) -> Any:
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
    "build_user_prompt",
    "synthesize_viewer_sentence",
]
