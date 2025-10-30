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


def _viewer_block(
    ex: Dict[str, Any],
    selected: Dict[str, Any],
    profile: Dict[str, Any],
) -> List[str]:
    """
    Build the viewer summary section of the prompt.

    :param ex: Interaction example containing viewer details.
    :type ex: dict[str, Any]
    :param selected: Selected row produced by :func:`load_selected_row`.
    :type selected: dict[str, Any]
    :param profile: Profile features rendered by :func:`render_profile`.
    :type profile: dict[str, Any]
    :returns: Lines describing the viewer context.
    :rtype: list[str]
    """

    lines: List[str] = []
    viewer_line = viewer_summary_line(ex, selected, profile)
    if viewer_line:
        lines.append(f"VIEWER {viewer_line}")
    else:
        lines.append("VIEWER (profile information unavailable)")

    viewpoint_line = initial_viewpoint_line(ex, selected)
    if viewpoint_line:
        lines.append(f"Initial Viewpoint: {viewpoint_line}")
    return lines


def _current_block(ex: Dict[str, Any], show_ids: bool) -> List[str]:
    """
    Build the ``CURRENTLY WATCHING`` block.

    :param ex: Interaction example containing the current video metadata.
    :type ex: dict[str, Any]
    :param show_ids: Whether identifiers should be displayed alongside titles.
    :type show_ids: bool
    :returns: Lines describing the current video.
    :rtype: list[str]
    """

    current_line = current_video_line(ex, show_ids)
    if current_line:
        return [f"CURRENTLY WATCHING {current_line}"]
    return ["CURRENTLY WATCHING (current video unavailable)"]


def _history_block(ex: Dict[str, Any], show_ids: bool, max_hist: int) -> List[str]:
    """
    Build the recent history section.

    :param ex: Interaction example containing history metadata.
    :type ex: dict[str, Any]
    :param show_ids: Whether identifiers should be displayed alongside titles.
    :type show_ids: bool
    :param max_hist: Maximum number of history items to include.
    :type max_hist: int
    :returns: Lines covering the watch history.
    :rtype: list[str]
    """

    lines: List[str] = ["RECENTLY WATCHED (NEWEST LAST)"]
    history_section = history_lines(ex, show_ids, max_hist)
    if history_section:
        lines.extend(history_section)
    else:
        lines.append("(no recently watched videos available)")
    return lines


def _survey_block(ex: Dict[str, Any]) -> List[str]:
    """
    Build the survey highlights section when available.

    :param ex: Interaction example containing survey answers.
    :type ex: dict[str, Any]
    :returns: Survey highlight lines or an empty list when missing.
    :rtype: list[str]
    """

    section = survey_lines(ex)
    if not section:
        return []
    return ["SURVEY HIGHLIGHTS", *section]


def _options_block(ex: Dict[str, Any], show_ids: bool) -> List[str]:
    """
    Build the options section for the prompt.

    :param ex: Interaction example containing slate options.
    :type ex: dict[str, Any]
    :param show_ids: Whether identifiers should be displayed alongside titles.
    :type show_ids: bool
    :returns: Lines describing the recommendation options.
    :rtype: list[str]
    """

    option_lines = options_lines(ex, show_ids)
    if option_lines:
        return ["OPTIONS", *option_lines]
    return ["OPTIONS", "(no recommendation options available)"]


def _normalise_issue_label(ex: Dict[str, Any]) -> str:
    """
    Return a human-readable opinion issue label.

    :param ex: Interaction example containing an ``issue`` field.
    :type ex: dict[str, Any]
    :returns: Capitalised issue label suitable for the questionnaire.
    :rtype: str
    """

    issue_label = str(ex.get("issue") or "the issue").strip()
    if not issue_label:
        return "The issue"
    pretty_issue = issue_label.replace("_", " ").strip()
    if not pretty_issue:
        return "The issue"
    capitalised = pretty_issue[0].upper() + pretty_issue[1:]
    return capitalised


def _questions_block(ex: Dict[str, Any]) -> List[str]:
    """
    Build the final question section.

    :param ex: Interaction example containing opinion metadata.
    :type ex: dict[str, Any]
    :returns: Lines encoding the downstream questions for the policy.
    :rtype: list[str]
    """

    pretty_issue = _normalise_issue_label(ex)
    return [
        "QUESTIONS",
        "1. Which option number will the viewer watch next?",
        (
            "2. After this recommendation, will the viewer's opinion on "
            f"{pretty_issue} increase, decrease, or stay the same?"
        ),
    ]


def _strip_trailing_blank_lines(lines: List[str]) -> List[str]:
    """
    Remove trailing blank lines introduced during section assembly.

    :param lines: Prompt lines that may contain trailing blanks.
    :type lines: list[str]
    :returns: Lines without trailing blanks.
    :rtype: list[str]
    """

    while lines and not lines[-1].strip():
        lines.pop()
    return lines


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

    assembled: List[str] = []
    for block in (
        _viewer_block(ex, selected, profile),
        _current_block(ex, show_ids),
        _history_block(ex, show_ids, max_hist),
        _survey_block(ex),
        _options_block(ex, show_ids),
        _questions_block(ex),
    ):
        if not block:
            continue
        if assembled:
            assembled.append("")
        assembled.extend(block)

    trimmed = _strip_trailing_blank_lines(assembled)
    return "\n".join(trimmed)


__all__ = [
    "build_user_prompt",
    "synthesize_viewer_sentence",
    "lookup_video_stats",
]
