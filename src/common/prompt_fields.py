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

NOW_PLAYING_TITLE_KEYS_WITH_META = (
    "current_video_title",
    "now_playing_title",
    "watching_title",
    "currentVideoTitle",
    "nowPlayingTitle",
    "watchingTitle",
    "now_title",
    "current_title",
    "meta_originTitle",
)

NOW_PLAYING_TITLE_KEYS = (
    "current_video_title",
    "now_playing_title",
    "watching_title",
    "currentVideoTitle",
    "nowPlayingTitle",
    "watchingTitle",
    "now_title",
    "current_title",
)

NOW_PLAYING_ID_KEYS = (
    "current_video_id",
    "now_playing_id",
    "watching_id",
    "currentVideoId",
    "nowPlayingId",
    "watchingId",
    "now_id",
    "current_id",
    "originId",
    "video_id",
    "videoId",
)
