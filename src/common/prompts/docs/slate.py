"""Slate extraction helpers used during prompt document assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple

from common import canon_text, canon_video_id

from ..fields import (
    NOW_PLAYING_ID_KEYS,
    NOW_PLAYING_TITLE_KEYS,
    NOW_PLAYING_TITLE_KEYS_WITH_META,
)
from .titles import TitleLookup
from .trajectory import load_trajectory_entries


def _pick_ci(mapping: object, *alternates: str) -> Optional[str]:
    """Return the first non-empty value from ``mapping`` matching ``alternates``."""
    if not isinstance(mapping, Mapping):
        return None
    lower = {key.lower(): key for key in mapping.keys()}
    for candidate in alternates:
        original = lower.get(candidate.lower())
        if not original:
            continue
        value = mapping.get(original)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_nanlike(value: object) -> bool:
    """Return ``True`` when ``value`` should be treated as a missing token."""
    if value is None:
        return True
    string = str(value).strip().lower()
    return string in {"", "nan", "none", "null", "na", "n/a"}


def _normalise_title(value: object) -> str:
    """Return a stripped title when ``value`` is a string."""
    return value.strip() if isinstance(value, str) else ""


def _normalise_video_id(value: object) -> str:
    """Return an 11-character YouTube id when ``value`` resembles one."""
    if not value:
        return ""
    candidate = canon_video_id(str(value))
    return candidate if len(candidate) == 11 else ""


def extract_now_watching(
    example: Mapping[str, object],
    title_lookup: TitleLookup | None,
) -> Optional[Tuple[str, str]]:
    """
    Return ``(title, video_id)`` describing the currently watched video.

    :param example: Source record containing now-playing metadata.
    :type example: Mapping[str, object]
    :param title_lookup: Optional callback for hydrating missing titles.
    :type title_lookup: TitleLookup | None
    :returns: Two-tuple ``(title, video_id)`` or ``None`` if unavailable.
    :rtype: Optional[Tuple[str, str]]
    """
    video_id = _pick_ci(example, "video_id", "videoId")
    if video_id and not _is_nanlike(video_id):
        title = _pick_ci(example, *NOW_PLAYING_TITLE_KEYS_WITH_META)
        if _is_nanlike(title) and title_lookup is not None:
            title = title_lookup(video_id)
        return (title or "(untitled)", str(video_id))
    title = _pick_ci(example, *NOW_PLAYING_TITLE_KEYS)
    video_id = _pick_ci(example, *NOW_PLAYING_ID_KEYS)
    if (title and not _is_nanlike(title)) or (video_id and not _is_nanlike(video_id)):
        if _is_nanlike(title) and title_lookup is not None:
            title = title_lookup(video_id) or ""
        return (title or "(untitled)", video_id or "")
    return None


def _structured_slate_candidates(raw: object) -> List[Tuple[object, object]]:
    """Extract raw ``(title, id)`` pairs from structured slate metadata."""
    if not isinstance(raw, list):
        return []
    pairs: List[Tuple[object, object]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        title = (
            entry.get("title")
            or entry.get("video_title")
            or entry.get("name")
            or entry.get("surface")
            or entry.get("text")
            or ""
        )
        video_id = (
            entry.get("id")
            or entry.get("video_id")
            or entry.get("videoId")
            or entry.get("ytid")
            or entry.get("yt_id")
            or entry.get("candidate_id")
            or entry.get("content_id")
            or ""
        )
        pairs.append((title, video_id))
    return pairs


@dataclass
class _SlateCollector:
    """Accumulate cleaned slate entries while minimising local state."""

    title_lookup: TitleLookup | None
    items: List[Tuple[str, str]] = field(default_factory=list)

    def add(self, title: object, video_id: object) -> None:
        """Append a cleaned ``(title, video_id)`` pair to ``items`` when viable."""
        cleaned_id = _normalise_video_id(video_id)
        cleaned_title = _normalise_title(title)
        if not cleaned_id and isinstance(title, str):
            possible_id = _normalise_video_id(title)
            if possible_id:
                cleaned_id = possible_id
                cleaned_title = ""
        if not cleaned_title and cleaned_id and self.title_lookup is not None:
            cleaned_title = self.title_lookup(cleaned_id) or ""
        if cleaned_title or cleaned_id:
            self.items.append((cleaned_title or "(untitled)", cleaned_id))


def _collect_structured_items(collector: _SlateCollector, example: Mapping[str, object]) -> bool:
    """Attempt to populate ``collector`` using structured slate arrays."""
    for key in ("slate_items", "options", "slate_items_with_meta"):
        candidates = _structured_slate_candidates(example.get(key))
        if not candidates:
            continue
        for title, video_id in candidates:
            collector.add(title, video_id)
        return True
    return False


def _collect_text_items(collector: _SlateCollector, slate_text: object) -> bool:
    """Parse textual slate descriptions and append any recovered pairs."""
    if not isinstance(slate_text, str) or not slate_text.strip():
        return False
    for line in slate_text.splitlines():
        token = line.strip()
        if not token:
            continue
        token = re.sub(r"^\s*(?:-|\d+\s*[\.\)])\s*", "", token)
        parts = token.split("\t") if "\t" in token else token.split("|", maxsplit=1)
        title_raw, vid_raw = (parts[0], parts[1]) if len(parts) == 2 else (token, "")
        collector.add(title_raw, vid_raw)
    return bool(collector.items)


def _collect_trajectory_items(collector: _SlateCollector, trajectory_json: object) -> None:
    """Fallback that scans trajectory entries when slate metadata is absent."""
    for entry in load_trajectory_entries(trajectory_json):
        raw_id = _pick_ci(
            entry,
            "video_id",
            "id",
            "videoId",
            "originId",
            "content_id",
        )
        title = _pick_ci(
            entry,
            "title",
            "video_title",
            "name",
            "surface",
            "text",
            "videoTitle",
        )
        collector.add(title, raw_id)


def _deduplicate_slate_items(items: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Remove duplicate slate entries while preserving order."""
    seen: set[str] = set()
    deduped: List[Tuple[str, str]] = []
    for title, video_id in items:
        key = canon_video_id(video_id) or canon_text(title)
        if key:
            if key in seen:
                continue
            seen.add(key)
        deduped.append((title, video_id))
    return deduped


def extract_slate_items(
    example: Mapping[str, object],
    title_lookup: TitleLookup | None,
) -> List[Tuple[str, str]]:
    """
    Return a list of ``(title, video_id)`` tuples extracted from ``example``.

    :param example: Source record containing slate metadata in multiple formats.
    :type example: Mapping[str, object]
    :param title_lookup: Optional callback used to hydrate missing titles.
    :type title_lookup: TitleLookup | None
    :returns: Ordered and de-duplicated slate entries.
    :rtype: List[Tuple[str, str]]
    """
    collector = _SlateCollector(title_lookup)
    if _collect_structured_items(collector, example):
        return _deduplicate_slate_items(collector.items)
    if _collect_text_items(collector, example.get("slate_text")):
        return _deduplicate_slate_items(collector.items)
    _collect_trajectory_items(collector, example.get("trajectory_json"))
    if not collector.items:
        return []
    return _deduplicate_slate_items(collector.items)


__all__ = ["extract_now_watching", "extract_slate_items"]
