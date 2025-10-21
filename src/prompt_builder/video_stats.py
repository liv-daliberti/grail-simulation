"""Lazy loader for precomputed video engagement statistics."""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional


VideoStats = Dict[str, Optional[int]]


def _candidate_paths() -> Iterable[Path]:
    """Yield potential filesystem locations for the stats bundle."""

    override = os.getenv("GRAIL_VIDEO_STATS_PATH")
    if override:
        yield Path(override)
    package_path = Path(__file__).resolve().parent / "data" / "video_stats.json"
    yield package_path
    yield Path("data/cleaned_grail/video_stats.json")
    yield Path("capsule-5416997/data/cleaned_grail/video_stats.json")


@lru_cache(maxsize=1)
def get_video_stats() -> Dict[str, VideoStats]:
    """Return a mapping of video id to engagement metrics.

    The function searches a small set of candidate locations and caches the first
    successful load for the duration of the process. Missing files yield an empty
    mapping.
    """

    for path in _candidate_paths():
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stats: Dict[str, VideoStats] = {}
        for vid, fields in payload.items():
            if not isinstance(vid, str):
                continue
            if not isinstance(fields, dict):
                continue
            stats[vid] = {
                key: _coerce_int(value)
                for key, value in fields.items()
                if key in {"view_count", "like_count", "dislike_count", "favorite_count", "comment_count", "share_count"}
            }
        if stats:
            return stats
    return {}


def lookup_video_stats(video_id: str) -> VideoStats:
    """Return engagement metrics for ``video_id`` when available."""

    if not video_id:
        return {}
    return get_video_stats().get(video_id, {})


def _coerce_int(value: object) -> Optional[int]:
    """Convert ``value`` into an integer when reasonable."""

    if value is None:
        return None

    result: Optional[int]
    if isinstance(value, bool):
        result = int(value)
    elif isinstance(value, int):
        result = value
    elif isinstance(value, float):
        result = None if math.isnan(value) else int(round(value))
    elif isinstance(value, str):
        text = value.replace(",", "").strip()
        if not text:
            result = None
        else:
            try:
                number = float(text)
            except ValueError:
                result = None
            else:
                result = int(round(number))
    else:
        result = None
    return result
