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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional

_VALID_STAT_KEYS = {
    "view_count",
    "like_count",
    "dislike_count",
    "favorite_count",
    "comment_count",
    "share_count",
}


VideoStats = Dict[str, Optional[int]]


def _candidate_paths() -> Iterable[Path]:
    """
    Yield potential filesystem locations for the stats bundle.

    :returns: Iterator of candidate paths searched in order.
    :rtype: Iterable[Path]
    """

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

    :returns: Mapping from video id to engagement metric dictionary.
    :rtype: Dict[str, VideoStats]
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
                if key in _VALID_STAT_KEYS
            }
        if stats:
            return stats
    return {}


def lookup_video_stats(video_id: str) -> VideoStats:
    """
    Return engagement metrics for ``video_id`` when available.

    :param video_id: YouTube video identifier.
    :type video_id: str
    :returns: Engagement metrics dictionary or an empty mapping.
    :rtype: VideoStats
    """

    if not video_id:
        return {}
    return get_video_stats().get(video_id, {})


def _coerce_int(value: object) -> Optional[int]:
    """
    Convert ``value`` into an integer when reasonable.

    :param value: Raw value that may encode a numeric quantity.
    :type value: object
    :returns: Integer representation or ``None`` when conversion is unsafe.
    :rtype: Optional[int]
    """

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
