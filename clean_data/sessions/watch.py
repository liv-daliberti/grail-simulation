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

"""Helpers for normalizing per-session watch metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from clean_data.helpers import _coerce_session_value as _helpers_coerce_session_value

from .models import SessionTiming


def coerce_session_value(value: Any) -> Any:
    """Convert session log values to numeric scalars when possible.

    :param value: Raw value from the session logs.
    :returns: Coerced numeric/boolean/string value suitable for storage.
    """

    return _helpers_coerce_session_value(value)


def normalize_session_mapping(
    values: Any,
    raw_vids: List[str],
    base_vids: List[str],
) -> Dict[str, Any]:
    """Convert per-video session arrays/dicts into a standard lookup dict.

    :param values: Raw session payload (dict/list/other) to normalize.
    :param raw_vids: Sequence of raw video ids in session order.
    :param base_vids: Sequence of canonical base video ids.
    :returns: Mapping from raw/base ids (and indices) to coerced values.
    """

    mapping: Dict[str, Any] = {}
    if isinstance(values, dict):
        for key, val in values.items():
            if key is None:
                continue
            mapping[str(key)] = coerce_session_value(val)
        return mapping
    if isinstance(values, list):
        for idx, val in enumerate(values):
            coerced = coerce_session_value(val)
            if idx < len(raw_vids):
                mapping.setdefault(raw_vids[idx], coerced)
            if idx < len(base_vids):
                mapping.setdefault(base_vids[idx], coerced)
            mapping.setdefault(str(idx), coerced)
        return mapping
    return mapping


def lookup_session_value(
    mapping: Dict[str, Any],
    raw_id: str,
    base_id: str,
) -> Any:
    """Retrieve a session metric for either the raw or canonical video id.

    :param mapping: Normalized session mapping produced by ``normalize_session_mapping``.
    :param raw_id: Raw video identifier used as a lookup key.
    :param base_id: Canonical video identifier used as a fallback key.
    :returns: Mapped session value or ``None`` when absent.
    """

    if not mapping:
        return None
    for key in (raw_id, base_id, str(raw_id), str(base_id)):
        if key and key in mapping:
            return mapping[key]
    return None


def build_session_timings(sess: dict, raw_vids: List[str], base_vids: List[str]) -> SessionTiming:
    """Normalize per-video timing dictionaries for a session.

    :param sess: Raw session payload containing timing arrays/dicts.
    :param raw_vids: Sequence of raw video ids in session order.
    :param base_vids: Sequence of canonical base video ids.
    :returns: :class:`SessionTiming` bundle with per-field lookup maps.
    """

    return SessionTiming(
        start=normalize_session_mapping(sess.get("vidStartTimes"), raw_vids, base_vids),
        end=normalize_session_mapping(sess.get("vidEndTimes"), raw_vids, base_vids),
        watch=normalize_session_mapping(sess.get("vidWatchTimes"), raw_vids, base_vids),
        total=normalize_session_mapping(sess.get("vidTotalLengths"), raw_vids, base_vids),
        delay=normalize_session_mapping(sess.get("contentStartDelay"), raw_vids, base_vids),
    )


@dataclass(frozen=True)
class VideoMetadataSources:
    """Bundles recommendation metadata lookups for helper functions."""

    tree_meta: Dict[str, Dict[str, Any]]
    fallback_titles: Dict[str, Any]
    tree_issue_map: Dict[str, str]


def _resolve_title(meta: Dict[str, Any], fallback_id: str) -> tuple[str, bool]:
    """Return a title for the candidate along with a missing flag.

    :param meta: Metadata dictionary containing title information.
    :param fallback_id: Video id used when a title is unavailable.
    :returns: Tuple of ``(title, missing_flag)``.
    """

    title = str(meta.get("title") or "").strip()
    if title:
        return title, False
    return f"(title missing for {fallback_id})", True


def _resolve_channel(meta: Dict[str, Any]) -> tuple[str, bool]:
    """Return the channel title with a missing flag.

    :param meta: Metadata dictionary containing channel information.
    :returns: Tuple of ``(channel_title, missing_flag)``.
    """

    channel = str(meta.get("channel_title") or "").strip()
    if channel:
        return channel, False
    return "(channel missing)", True


def _apply_timing_fields(
    entry: Dict[str, Any],
    raw_vid: str,
    base_vid: str,
    timings: SessionTiming,
) -> None:
    """Populate timing metrics for the watched entry when present.

    :param entry: Watched-entry dictionary to be updated in-place.
    :param raw_vid: Raw video identifier for the row.
    :param base_vid: Canonical video identifier for the row.
    :param timings: Precomputed session timing lookups.
    """

    for field_name, timing_map in (
        ("start_delay_ms", timings.delay),
        ("start_ms", timings.start),
        ("end_ms", timings.end),
        ("watch_ms", timings.watch),
        ("total_length_ms", timings.total),
    ):
        value = lookup_session_value(timing_map, raw_vid, base_vid)
        if value is not None:
            entry[field_name] = value


def _video_meta(
    base_id: str,
    raw_id: str,
    metadata: VideoMetadataSources,
) -> Dict[str, Any]:
    """Return augmented metadata for a watched video.

    :param base_id: Canonical video identifier.
    :param raw_id: Raw video identifier (may match ``base_id``).
    :param metadata: Sources providing tree metadata and fallbacks.
    :returns: Metadata dictionary describing the watched video.
    """

    info = dict(metadata.tree_meta.get(base_id) or {})
    if raw_id and raw_id != base_id:
        raw_ids = list(info.get("raw_ids") or [])
        if raw_id not in raw_ids:
            raw_ids.append(raw_id)
        if raw_ids:
            info["raw_ids"] = raw_ids
    if not info.get("title"):
        title = metadata.fallback_titles.get(base_id) or metadata.fallback_titles.get(raw_id)
        if title:
            info["title"] = title
    if not info.get("issue"):
        issue_guess = metadata.tree_issue_map.get(base_id)
        if issue_guess:
            info["issue"] = issue_guess
    return info


def build_detail_entry(
    idx: int,
    raw_vid: str,
    base_vid: str,
    metadata: VideoMetadataSources,
    timings: SessionTiming,
) -> Dict[str, Any]:
    """Return a single watched-detail dictionary with timing fields.

    :param idx: Zero-based position of the video within the session.
    :param raw_vid: Raw video identifier.
    :param base_vid: Canonical video identifier.
    :param metadata: Metadata sources for resolving titles/channels.
    :param timings: Session timing lookup bundle.
    :returns: Dictionary describing the watched video with metadata.
    """

    meta = _video_meta(base_vid, raw_vid, metadata)
    title_val, title_missing = _resolve_title(meta, base_vid)
    channel_val, channel_missing = _resolve_channel(meta)
    entry: Dict[str, Any] = {
        "id": base_vid,
        "raw_id": raw_vid,
        "idx": idx,
        "title": title_val,
        "title_missing": title_missing,
        "channel_title": channel_val,
        "channel_missing": channel_missing,
    }
    if meta.get("channel_id"):
        entry["channel_id"] = meta["channel_id"]
    _apply_timing_fields(entry, raw_vid, base_vid, timings)

    recs = meta.get("recs")
    if isinstance(recs, list):
        entry["recommendations"] = [dict(rec) for rec in recs if isinstance(rec, dict)]
    return entry


def build_watched_details(
    raw_vids: List[str],
    base_vids: List[str],
    metadata: VideoMetadataSources,
    *,
    timings: SessionTiming,
) -> List[Dict[str, Any]]:
    """Return per-video metadata entries for the watched sequence.

    :param raw_vids: Ordered list of raw video identifiers.
    :param base_vids: Ordered list of canonical video identifiers.
    :param metadata: Metadata sources for resolving per-video information.
    :param timings: Session timing lookup bundle.
    :returns: List of dictionaries describing each watched video.
    """

    details: List[Dict[str, Any]] = []
    for idx, raw_vid in enumerate(raw_vids):
        base = base_vids[idx]
        details.append(
            build_detail_entry(
                idx,
                raw_vid,
                base,
                metadata,
                timings,
            )
        )

    return details


__all__ = [
    "VideoMetadataSources",
    "build_detail_entry",
    "build_session_timings",
    "build_watched_details",
    "coerce_session_value",
    "lookup_session_value",
    "normalize_session_mapping",
]
