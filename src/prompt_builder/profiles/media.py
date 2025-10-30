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

"""Media consumption sentence builders for viewer profiles."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from ..constants import NEWS_TRUST_FIELD_NAMES, YT_FREQ_MAP
from ..formatters import clean_text
from ..parsers import format_yes_no
from ..profile_helpers import first_available_text, sentencize
from ..shared import first_non_nan_value

MEDIA_SOURCES: Sequence[Tuple[Sequence[str], str, bool]] = (
    (("q8", "fav_channels"), "Favorite channels", True),
    (("q78", "popular_channels"), "Popular channels followed", False),
    (("media_diet",), "Media diet", False),
    (("news_consumption",), "News consumption", False),
    (("news_sources",), "News sources", False),
    (("news_sources_top",), "Top news sources", False),
    (("news_frequency", "newsint"), "News frequency", False),
    (("platform_use",), "Platform usage", False),
    (("social_media_use",), "Social media use", False),
    (NEWS_TRUST_FIELD_NAMES, "News trust", False),
)


def _media_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences describing overall media consumption habits."""

    media_section: List[str] = []
    media_section.extend(_youtube_frequency_sentences(ex, selected))
    media_section.extend(_youtube_binge_sentences(ex, selected))
    seen: set[str] = set()
    for sources, label, skip_duplicate in MEDIA_SOURCES:
        text = first_available_text(ex, selected, sources)
        if not text:
            continue
        if skip_duplicate and label in seen:
            continue
        media_section.append(f"{label}: {text}")
        seen.add(label)
    sentence = sentencize("Media habits include", media_section)
    return [sentence] if sentence else []


def _youtube_frequency_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return phrases describing YouTube watch frequency."""

    freq_raw = first_non_nan_value(
        ex,
        selected,
        "freq_youtube",
        "q77",
        "Q77",
        "youtube_freq",
        "youtube_freq_v2",
    )
    if freq_raw is None:
        return []
    code = str(freq_raw).strip()
    mapped = YT_FREQ_MAP.get(code)
    if mapped:
        return [f"YouTube frequency: {mapped}"]
    freq_text = clean_text(freq_raw)
    if not freq_text:
        return []
    return [f"YouTube frequency: {freq_text}"]


def _youtube_binge_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return phrases describing binge behaviour and reported watch time."""

    binge_raw = first_non_nan_value(ex, selected, "binge_youtube", "youtube_time")
    if binge_raw is None:
        return []
    binge_text = format_yes_no(binge_raw, yes="yes", no="no")
    if binge_text:
        return [f"Binge watches YouTube: {binge_text}"]
    binge_clean = clean_text(binge_raw)
    if not binge_clean:
        return []
    return [f"YouTube time reported: {binge_clean}"]
