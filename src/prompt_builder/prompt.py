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

"""Prompt rendering helpers for the Grail recommendation tasks."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from .profile_helpers import load_selected_row
from .prompt_helpers import (
    current_video_line,
    history_lines,
    initial_viewpoint_line,
    options_lines,
    survey_lines,
    viewer_summary_line,
)
from .profiles import render_profile, synthesize_viewer_sentence as _synthesize_viewer_sentence
from .video_stats import lookup_video_stats

synthesize_viewer_sentence = _synthesize_viewer_sentence


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
    selected = load_selected_row(ex)

    lines: List[str] = []

    viewer_line = viewer_summary_line(ex, selected, profile)
    if viewer_line:
        lines.append(f"VIEWER {viewer_line}")
    else:
        lines.append("VIEWER (profile information unavailable)")

    viewpoint_line = initial_viewpoint_line(ex, selected)
    if viewpoint_line:
        lines.append(f"Initial Viewpoint: {viewpoint_line}")

    lines.append("")

    current_line = current_video_line(ex, show_ids)
    if current_line:
        lines.append(f"CURRENTLY WATCHING {current_line}")
    else:
        lines.append("CURRENTLY WATCHING (current video unavailable)")

    lines.append("")
    lines.append("RECENTLY WATCHED (NEWEST LAST)")
    history_section = history_lines(ex, show_ids, max_hist)
    if history_section:
        lines.extend(history_section)
    else:
        lines.append("(no recently watched videos available)")

    survey_section = survey_lines(ex)
    if survey_section:
        lines.append("")
        lines.append("SURVEY HIGHLIGHTS")
        lines.extend(survey_section)

    option_lines = options_lines(ex, show_ids)
    lines.append("")
    lines.append("OPTIONS")
    if option_lines:
        lines.extend(option_lines)
    else:
        lines.append("(no recommendation options available)")

    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


__all__ = [
    "build_user_prompt",
    "synthesize_viewer_sentence",
    "lookup_video_stats",
]
