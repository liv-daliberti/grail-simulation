"""Collect structured samples from JSONL prediction outputs.

This module parses model output files for different tasks (e.g., next_video,
opinion) and converts them into typed ``Sample`` records used by reports.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List, Mapping, Optional, Sequence

from common.pipeline.io import iter_jsonl_rows

from .samples_types import Sample

_THINK_RE = re.compile(r"(?si)<think>\s*(.*?)\s*</think>")
_ANSWER_RE = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")


def _iter_jsonl_rows(path: Path) -> Iterator[Mapping[str, object]]:
    """Thin wrapper for compatibility and easy mocking in tests."""
    yield from iter_jsonl_rows(path)


def _extract_user_question(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if isinstance(item, Mapping) and (item.get("role") == "user") and item.get("content"):
            return str(item["content"]).strip()
    return ""


def _extract_think_answer(text: str) -> tuple[str, str]:
    think = ""
    ans = ""
    think_match = _THINK_RE.search(text or "")
    if think_match:
        think = think_match.group(1).strip()
    answer_match = _ANSWER_RE.search(text or "")
    if answer_match:
        ans = answer_match.group(1).strip()
    return think, ans


def _normalise_issue(value: object) -> str:
    try:
        text = str(value or "").strip().lower()
    except (TypeError, ValueError):
        text = ""
    if text in {"gun", "guns", "gun control", "gun_control"}:
        return "gun_control"
    if text in {"wage", "minimum wage", "minimum_wage"}:
        return "minimum_wage"
    return text or "unspecified"



def collect_next_video_samples(files: Sequence[Path]) -> List[Sample]:
    """Parse next-video predictions JSONL files into :class:`Sample`s."""
    out: List[Sample] = []
    for path in files:
        for row in _iter_jsonl_rows(path):
            question = _extract_user_question(row.get("messages"))
            raw = str(row.get("gpt_output") or row.get("raw_output") or "")
            think, answer = _extract_think_answer(raw)
            issue = _normalise_issue(row.get("issue"))
            if question and (think or answer):
                out.append(
                    Sample(
                        issue=issue,
                        task="next_video",
                        question=question,
                        think=think,
                        answer=answer,
                    )
                )
    return out


def collect_opinion_samples(files: Sequence[Path]) -> List[Sample]:
    """Parse opinion predictions JSONL files into :class:`Sample`s."""
    out: List[Sample] = []
    for path in files:
        for row in _iter_jsonl_rows(path):
            question = _extract_user_question(row.get("messages"))
            raw = str(row.get("raw_output") or row.get("gpt_output") or "")
            think, answer = _extract_think_answer(raw)
            issue = _normalise_issue(row.get("issue"))
            before = row.get("before")
            pred = row.get("predicted_after") or row.get("prediction")
            # Coerce numeric values when possible for clearer notes.
            b_val: Optional[float]
            a_val: Optional[float]
            try:
                b_val = float(before) if before is not None else None
            except (TypeError, ValueError):
                b_val = None
            try:
                a_val = float(pred) if pred is not None else None
            except (TypeError, ValueError):
                a_val = None
            if question and (think or answer):
                out.append(
                    Sample(
                        issue=issue,
                        task="opinion",
                        question=question,
                        think=think,
                        answer=answer,
                        before=b_val,
                        predicted_after=a_val,
                    )
                )
    return out


__all__ = [
    "collect_next_video_samples",
    "collect_opinion_samples",
]
