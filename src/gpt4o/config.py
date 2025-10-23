"""Shared configuration for the GPT-4o slate baseline."""

from __future__ import annotations

import os
from pathlib import Path

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
    """
    Ensure the Azure environment variables are present.

    The sandbox runner historically relied on these being set globally.
    We keep the behaviour but allow shell overrides to win.
    """

    os.environ.setdefault("SANDBOX_API_KEY", SANDBOX_API_KEY)
    os.environ.setdefault("SANDBOX_ENDPOINT", SANDBOX_ENDPOINT)
    os.environ.setdefault("SANDBOX_API_VER", SANDBOX_API_VER)
    os.environ.setdefault("DEPLOYMENT_NAME", DEPLOYMENT_NAME)


# ---------------------------------------------------------------------------
# Dataset paths / column definitions
# ---------------------------------------------------------------------------
DATASET_NAME: str = os.environ.get("GPT4O_DATASET", "od2961/grail-interactions")
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
SYSTEM_PROMPT: str = """You are choosing EXACTLY ONE item from a short slate for a specific viewer.

Input you will see:
  • Viewer profile and optional context/history
  • An "OPTIONS:" list with items numbered 1..N
    – Each item is shown as either a title (preferred) or an id

Your job:
  • Think briefly in <think>…</think> using the viewer’s profile, context, and options.
  • Compare the top 2–3 candidates, then choose the single best option.
  • Never invent new items; choose only from the given OPTIONS list.

Output format (STRICT):
  • First output your hidden reasoning in <think>…</think>.
    – In your thinking, reference candidates by their numbers and names (or ids)
      to justify the choice.
  • Then output ONLY the chosen option’s NUMBER inside <answer>…</answer>.
    – Do NOT output the name, id, or any extra text—ONLY the number.
    – Do NOT include punctuation, quotes,
      or a period after the number.

Examples of valid <answer>:
  <answer>
  3
  </answer>

Examples of INVALID <answer> (never do these):
  <answer>3.</answer>                 ← trailing period
  <answer>"3"</answer>                ← quoted
  <answer>Option 3</answer>           ← extra words
  <answer>Parkland …</answer>         ← name instead of number
  You only have 100 tokens to think and 50 tokens to answer.
"""


# Default cache directory used by the CLI when none is provided
DEFAULT_CACHE_DIR: Path = Path.cwd() / "hf_cache"
