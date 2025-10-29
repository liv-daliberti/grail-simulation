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

"""Configuration constants for the GPT-4o baseline and evaluation pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Azure OpenAI defaults (can be overridden via environment variables)
# ---------------------------------------------------------------------------
SANDBOX_API_KEY: str = os.environ.get(
    "SANDBOX_API_KEY", "1e30d0e4d7564ba984e8adff48053009"
)
SANDBOX_ENDPOINT: str = os.environ.get(
    "SANDBOX_ENDPOINT", "https://api-ai-sandbox.princeton.edu/"
)
SANDBOX_API_VER: str = os.environ.get("SANDBOX_API_VER", "2025-03-01-preview")
DEPLOYMENT_NAME: str = os.environ.get("DEPLOYMENT_NAME", "gpt-4o")


def ensure_azure_env() -> None:
    """Ensure Azure configuration defaults are present for downstream clients.

    The sandbox runner historically relied on these values being set globally,
    so we preserve that behaviour while still allowing explicit shell overrides.

    :returns: ``None``. The module-level constants are used when variables are unset.
    """

    os.environ.setdefault("SANDBOX_API_KEY", SANDBOX_API_KEY)
    os.environ.setdefault("SANDBOX_ENDPOINT", SANDBOX_ENDPOINT)
    os.environ.setdefault("SANDBOX_API_VER", SANDBOX_API_VER)
    os.environ.setdefault("DEPLOYMENT_NAME", DEPLOYMENT_NAME)


# ---------------------------------------------------------------------------
# Dataset paths / column definitions
# ---------------------------------------------------------------------------
_LOCAL_DATASET = Path(__file__).resolve().parents[2] / "data" / "cleaned_grail"
if "GPT4O_DATASET" in os.environ:
    _dataset_name = os.environ["GPT4O_DATASET"]
elif _LOCAL_DATASET.exists():
    _dataset_name = str(_LOCAL_DATASET)
else:
    _dataset_name = "od2961/grail-interactions"
DATASET_NAME: Final[str] = _dataset_name
TRAIN_SPLIT: str = os.environ.get("GPT4O_TRAIN_SPLIT", "train")
EVAL_SPLIT: str = os.environ.get("GPT4O_EVAL_SPLIT", "validation")
PROMPT_COLUMN: str = os.environ.get("GPT4O_PROMPT_COLUMN", "state_text")
SOLUTION_COLUMN: str = os.environ.get("GPT4O_SOLUTION_COLUMN", "video_id")

# Defaults for locating auxiliary title metadata
_DEFAULT_TITLE_ROOT = Path(
    "/n/fs/similarity/trees/data/results/capsule-5416997-data/recommendation trees"
)
DEFAULT_TITLE_DIRS: list[str] = [
    str(_DEFAULT_TITLE_ROOT / "trees_gun"),
    str(_DEFAULT_TITLE_ROOT / "trees_wage"),
]


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = (
    "You are choosing EXACTLY ONE item from the list of OPTIONS for a specific VIEWER.\n"
    "\n"
    "Input you will see:\n"
    "  • Viewer profile and optional context/history\n"
    "  • An \"OPTIONS:\" list with items numbered 1..N\n"
    "    – Each item is shown as either a title (preferred) or an id\n"
    "\n"
    "Your job:\n"
    "  • Think briefly in <think>…</think> using the viewer’s profile, context, and options.\n"
    "  • Compare the top 2–3 candidates, then choose the single best option.\n"
    "  • Never invent new items; choose only from the given OPTIONS list.\n"
    "\n"
    "Output format (STRICT):\n"
    "  • Always include BOTH tags in this order: <think>…</think> followed by <answer>…</answer>.\n"
    "  • First output your hidden reasoning in <think>…</think>.\n"
    "    – In your thinking, reference candidates by their numbers and names (or ids)\n"
    "      to justify the choice.\n"
    "  • Then output ONLY the chosen option’s NUMBER inside <answer>…</answer>.\n"
    "    – Do NOT output the name, id, or any extra text—ONLY the number.\n"
    "    – Do NOT include punctuation, quotes,\n"
    "      or a period after the number.\n"
    "\n"
    "Examples of valid <answer>:\n"
    "  <think>\n"
    "  WHY YOU THINK THIS IS THE RIGHT CHOICE\n"
    "  </think>\n"
    "  <answer>\n"
    "  3\n"
    "  </answer>\n"
    "\n"
    "Examples of INVALID <answer> (never do these):\n"
    "  <think></think><answer>3.</answer>                 ← trailing period\n"
    "  <think></think><answer>\"3\"</answer>                ← quoted\n"
    "  <think></think><answer>Option 3</answer>           ← extra words\n"
    "  <think></think><answer>Parkland …</answer>         ← name instead of number\n"
    "  You only have 100 tokens to think and 50 tokens to answer.\n"
)

OPINION_SYSTEM_PROMPT: str = """You estimate how a viewer’s opinion index changes
after watching a recommended video.

Workflow:
  • Read the viewer profile and context provided.
  • Use <think>…</think> to reason about how the next video might shift the viewer’s opinion.
  • Conclude with ONLY the numeric post-study opinion index (1–7) inside <answer>…</answer>.

Formatting rules:
  • <think> MUST contain the reasoning steps.
  • <answer> MUST contain a single number with no extra words or punctuation.
  • Never omit either tag, and never output additional tags.
"""


# Default cache directory used by the CLI when none is provided
DEFAULT_CACHE_DIR: Path = Path.cwd() / "hf_cache"
