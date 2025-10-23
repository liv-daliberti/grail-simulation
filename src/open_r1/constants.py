"""Shared constants reused across Open-R1 scripts."""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "You are choosing EXACTLY ONE item from a short slate for a specific viewer.\n"
    "Think briefly in <think>…</think>, then output ONLY the option NUMBER "
    "(1..N) inside <answer>…</answer>.\n"
    "Format (STRICT): <think>…</think><answer>3</answer>"
)

__all__ = ["DEFAULT_SYSTEM_PROMPT"]
