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

"""Render sample generative responses from existing evaluation artefacts.

This module assembles small, human-readable examples showing the exact
question shown to a model and the model's structured response using
the <think>…</think> and <answer>…</answer> tags (with an optional
<opinion> label for opinion-shift predictions).

Outputs are written under a new subdirectory in the reports tree:

  reports/<family>/sample_generative_responses/README.md

where <family> is one of "grpo", "grail", or "gpt4o".
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from common.pipeline.io import write_markdown_lines


_THINK_RE = re.compile(r"(?si)<think>\s*(.*?)\s*</think>")
_ANSWER_RE = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")


def _extract_user_question(messages: object) -> str:
    """Return the last user message from a chat transcript.

    :param messages: Chat message list as persisted in predictions JSONL.
    :returns: The latest user message text or an empty string.
    """

    if not isinstance(messages, list):
        return ""
    # Iterate backwards to find the most recent user message.
    for item in reversed(messages):
        if isinstance(item, Mapping) and (item.get("role") == "user") and item.get("content"):
            return str(item["content"]).strip()
    return ""


def _extract_think_answer(text: str) -> Tuple[str, str]:
    """Extract the <think> and <answer> contents from ``text``.

    :param text: Model output containing the required tags.
    :returns: Tuple ``(think, answer)``; items may be empty when missing.
    """

    think = ""
    ans = ""
    m = _THINK_RE.search(text or "")
    if m:
        think = m.group(1).strip()
    m = _ANSWER_RE.search(text or "")
    if m:
        ans = m.group(1).strip()
    return think, ans


@dataclass(frozen=True)
class Sample:
    """Container for a single rendered sample."""

    issue: str
    task: str  # "next_video" or "opinion"
    question: str
    think: str
    answer: str
    opinion_label: Optional[str] = None  # increase|decrease|no_change when task=="opinion"


def _iter_jsonl_rows(path: Path) -> Iterator[Mapping[str, object]]:
    """Yield parsed JSON objects from ``path`` when available."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, Mapping):
                    yield row
    except OSError:
        return


def _normalise_issue(value: object) -> str:
    try:
        text = str(value or "").strip().lower()
    except Exception:
        text = ""
    if text in {"gun", "guns", "gun control", "gun_control"}:
        return "gun_control"
    if text in {"wage", "minimum wage", "minimum_wage"}:
        return "minimum_wage"
    return text or "unspecified"


def _opinion_direction_label(before: float, predicted_after: float, *, tol: float = 1e-6) -> str:
    """Return ``increase``, ``decrease``, or ``no_change`` based on ``before``→``predicted_after``.

    :param before: Pre-study opinion index (1–7 scale).
    :param predicted_after: Predicted post-study index (1–7 scale).
    :param tol: Absolute tolerance for treating small deltas as no-change.
    :returns: Canonical direction label.
    """

    try:
        b = float(before)
        a = float(predicted_after)
    except (TypeError, ValueError):
        return "no_change"
    delta = a - b
    if math.isfinite(delta):
        if delta > tol:
            return "increase"
        if delta < -tol:
            return "decrease"
    return "no_change"


def _collect_next_video_samples(files: Sequence[Path]) -> List[Sample]:
    """Return parsed samples from next-video predictions JSONL files.

    :param files: Prediction files produced by next-video evaluation runs.
    :returns: Flat list of :class:`Sample` objects (task=="next_video").
    """

    out: List[Sample] = []
    for path in files:
        for row in _iter_jsonl_rows(path):
            question = _extract_user_question(row.get("messages"))
            raw = str(row.get("gpt_output") or row.get("raw_output") or "")
            think, answer = _extract_think_answer(raw)
            issue = _normalise_issue(row.get("issue"))
            if question and (think or answer):
                out.append(Sample(issue=issue, task="next_video", question=question, think=think, answer=answer))
    return out


def _collect_opinion_samples(files: Sequence[Path]) -> List[Sample]:
    """Return parsed samples from opinion predictions JSONL files.

    :param files: Per-study prediction files produced by opinion evaluations.
    :returns: Flat list of :class:`Sample` objects (task=="opinion").
    """

    out: List[Sample] = []
    for path in files:
        for row in _iter_jsonl_rows(path):
            question = _extract_user_question(row.get("messages"))
            raw = str(row.get("raw_output") or row.get("gpt_output") or "")
            think, answer = _extract_think_answer(raw)
            issue = _normalise_issue(row.get("issue"))
            # Derive direction from available numeric fields.
            before = row.get("before")
            pred = row.get("predicted_after") or row.get("prediction")
            direction = _opinion_direction_label(before if before is not None else float("nan"), pred if pred is not None else float("nan"))
            if question and (think or answer):
                out.append(Sample(issue=issue, task="opinion", question=question, think=think, answer=answer, opinion_label=direction))
    return out


def _format_sample_block(idx: int, sample: Sample) -> List[str]:
    """Return Markdown lines rendering a single sample block."""

    lines: List[str] = []
    lines.append(f"### Example {idx} ({sample.task.replace('_', ' ').title()})")
    lines.append("")
    lines.append("#### Question")
    lines.append("")
    lines.append("```text")
    lines.append(sample.question)
    lines.append("```")
    lines.append("")
    lines.append("#### Model Response")
    lines.append("")
    # Write <think> and <answer> blocks exactly as the model produced them.
    # Preserve empty blocks when only one tag is available.
    if sample.think:
        lines.append("<think>")
        lines.append(sample.think)
        lines.append("</think>")
        lines.append("")
    if sample.answer:
        lines.append("<answer>")
        lines.append(sample.answer)
        lines.append("</answer>")
        lines.append("")
    if sample.task == "opinion" and sample.opinion_label:
        lines.append(f"<opinion>{sample.opinion_label}</opinion>")
        lines.append("")
    return lines


def write_sample_responses_report(
    *,
    reports_root: Path,
    family_label: str,
    next_video_files: Sequence[Path],
    opinion_files: Sequence[Path],
    per_issue: int = 5,
) -> None:
    """
    Assemble a report with sample questions and model responses.

    The report includes up to ``per_issue`` examples for each issue (gun_control
    and minimum_wage), prioritising next-video examples and topping up with
    opinion examples when needed.

    :param reports_root: Target directory under which the report is written.
    :param family_label: Human-readable family name (e.g., "GRPO", "GRAIL", "GPT-4o").
    :param next_video_files: Prediction JSONL files for next-video.
    :param opinion_files: Prediction JSONL files for opinion.
    :param per_issue: Number of examples to include per issue.
    :returns: ``None``.
    """

    samples_dir = reports_root / "sample_generative_responses"
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / "README.md"

    lines: List[str] = [f"# {family_label} Sample Generative Model Responses", ""]

    nv_samples = _collect_next_video_samples(list(next_video_files)) if next_video_files else []
    op_samples = _collect_opinion_samples(list(opinion_files)) if opinion_files else []

    # Bucket by issue and task for selection.
    by_issue: MutableMapping[str, List[Sample]] = {"gun_control": [], "minimum_wage": []}
    for s in nv_samples:
        key = s.issue if s.issue in by_issue else "minimum_wage" if "wage" in s.issue else "gun_control"
        by_issue.setdefault(key, []).append(s)
    for s in op_samples:
        key = s.issue if s.issue in by_issue else "minimum_wage" if "wage" in s.issue else "gun_control"
        by_issue.setdefault(key, []).append(s)

    any_samples = any(by_issue.get(k) for k in by_issue)
    if not any_samples:
        lines.append("No examples found in existing predictions. Populate models/*/predictions.jsonl first.")
        lines.append("")
        write_markdown_lines(path, lines)
        return

    # For each issue, select up to per_issue examples preferring next_video first.
    for issue_key, issue_label in (("gun_control", "Gun Control"), ("minimum_wage", "Minimum Wage")):
        selected: List[Sample] = []
        nv_list = [s for s in nv_samples if s.issue == issue_key]
        op_list = [s for s in op_samples if s.issue == issue_key]
        selected.extend(nv_list[:per_issue])
        if len(selected) < per_issue:
            remaining = per_issue - len(selected)
            selected.extend(op_list[:remaining])
        if not selected:
            # Skip empty sections entirely to keep the report concise.
            continue
        lines.append(f"## {issue_label}")
        lines.append("")
        for idx, sample in enumerate(selected, start=1):
            lines.extend(_format_sample_block(idx, sample))

    write_markdown_lines(path, lines)


__all__ = [
    "write_sample_responses_report",
]

