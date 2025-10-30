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

"""Prompt-building helpers shared across opinion pipelines."""

from __future__ import annotations

from typing import Sequence


def format_opinion_user_prompt(
    *,
    issue_label: str,
    pre_study_index: float,
    viewer_context: str,
    post_watch_instruction: str,
    extra_lines: Sequence[str] | None = None,
) -> str:
    """
    Compose the user message guiding opinion prediction responses.

    :param issue_label: Human-readable issue label (e.g., ``"Gun Control"``).
    :param pre_study_index: Participant's pre-study opinion index.
    :param viewer_context: Context document describing the participant.
    :param post_watch_instruction: Instruction describing the post-viewing estimation.
    :param extra_lines: Optional lines appended after the standard instructions.
    :returns: Markdown-formatted user prompt string.
    """

    context = viewer_context.strip() if viewer_context else ""
    lines = [
        f"Issue: {issue_label}",
        "Opinion scale: 1 = strongly oppose, 7 = strongly support.",
        f"Pre-study opinion index: {pre_study_index:.2f}",
        "",
        "Viewer context:",
        context,
        "",
        post_watch_instruction,
        "Reason briefly inside <think> then output ONLY the numeric index (1-7) inside <answer>.",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    return "\n".join(lines).strip()


__all__ = ["format_opinion_user_prompt"]
