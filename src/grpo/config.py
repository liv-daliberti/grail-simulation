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

"""Static configuration used by the GRPO evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

from common.pipeline.prompts import STRICT_NUMBERED_ANSWER_GUIDE

from . import DEFAULT_DATASET_PATH, DEFAULT_EVAL_SPLIT

REPO_ROOT = Path(__file__).resolve().parents[2]

# Dataset defaults ---------------------------------------------------------------------------

DATASET_NAME = DEFAULT_DATASET_PATH
EVAL_SPLIT = DEFAULT_EVAL_SPLIT

# Prompt defaults ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are choosing EXACTLY ONE item from the list of OPTIONS for a specific VIEWER.\n\n"
    "Input you will see:\n"
    "  • Viewer profile and optional context/history\n"
    "  • An \"OPTIONS:\" list with items numbered 1..N\n"
    "    – Each item is shown as either a title (preferred) or an id\n\n"
    "Your job:\n"
    "  • Think briefly in <think>…</think> using the viewer’s profile, context, and options.\n"
    "  • Compare the top 2–3 candidates, then choose the single best option.\n"
    "  • Never invent new items; choose only from the given OPTIONS list.\n\n"
    "Output format (STRICT):\n"
    "  • Always include BOTH tags in this order: <think>…</think> followed by <answer>…</answer>.\n"
    "  • First output your hidden reasoning in <think>…</think>.\n"
    "    – In your thinking, reference candidates by their numbers and names\n"
    "      (or ids) to justify the choice.\n"
    "  • Then output ONLY the chosen option’s NUMBER inside <answer>…</answer>.\n"
    "    – Do NOT output the name, id, or any extra text—ONLY the number.\n"
    "    – Do NOT include punctuation, quotes, or a period after the number.\n"
    f"{STRICT_NUMBERED_ANSWER_GUIDE}"
)

# Opinion defaults -------------------------------------------------------------------------

OPINION_SYSTEM_PROMPT = (
    "You are estimating opinion change on a 1–7 scale "
    "(1 = strongly oppose, 7 = strongly support).\n"
    "Always reason inside <think>…</think>, then output ONLY the numeric index inside "
    "<answer>…</answer>.\n"
)


def repo_root() -> Path:
    """Return the repository root directory.

    :returns: Absolute :class:`pathlib.Path` pointing to the repo root.
    :rtype: Path
    """

    return REPO_ROOT
