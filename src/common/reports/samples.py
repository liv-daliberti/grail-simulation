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
    # Optional task-specific enrichments for clearer notes
    chosen_option: Optional[int] = None
    before: Optional[float] = None
    predicted_after: Optional[float] = None


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
            chosen: Optional[int] = None
            try:
                chosen = int(answer) if answer.strip() else None
            except Exception:
                chosen = None
            if question and (think or answer):
                out.append(
                    Sample(
                        issue=issue,
                        task="next_video",
                        question=question,
                        think=think,
                        answer=answer,
                        chosen_option=chosen,
                    )
                )
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
            # Coerce numeric values when possible for clearer notes.
            b_val: Optional[float]
            a_val: Optional[float]
            try:
                b_val = float(before) if before is not None else None
            except Exception:
                b_val = None
            try:
                a_val = float(pred) if pred is not None else None
            except Exception:
                a_val = None
            if question and (think or answer):
                out.append(
                    Sample(
                        issue=issue,
                        task="opinion",
                        question=question,
                        think=think,
                        answer=answer,
                        opinion_label=direction,
                        before=b_val,
                        predicted_after=a_val,
                    )
                )
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
    # Present the model's response inside a code fence as requested.
    # Keep <opinion> (derived) together for quick scanning.
    lines.append("```text")
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
    lines.append("```")
    lines.append("")

    # Add an explicit notes section explaining what the sample shows.
    def _bool_label(flag: bool) -> str:
        return "yes" if flag else "no"

    has_think = bool(sample.think)
    has_answer = bool(sample.answer)

    lines.append("#### Notes")
    lines.append("")
    lines.append(f"- Issue: {sample.issue.replace('_', ' ')}")
    lines.append(f"- Task: {'Next-video selection' if sample.task == 'next_video' else 'Opinion shift prediction'}")
    lines.append(f"- Tags present — think: {_bool_label(has_think)}, answer: {_bool_label(has_answer)}")
    if sample.task == "next_video":
        chosen = sample.chosen_option if sample.chosen_option is not None else sample.answer.strip()
        lines.append(f"- Chosen option: {chosen}")
    else:  # opinion
        if sample.before is not None:
            lines.append(f"- Pre-study opinion index: {sample.before:.2f}")
        if sample.predicted_after is not None:
            lines.append(f"- Predicted post-study index: {sample.predicted_after:.2f}")
        if sample.opinion_label:
            lines.append(f"- Predicted direction: {sample.opinion_label}")

    # Include a short rationale summary from <think> when available.
    if sample.think:
        snippet = sample.think.strip().splitlines()[0]
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        lines.append(f"- Short rationale: {snippet}")

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
    lines.extend([
        "This gallery shows concrete questions given to the model and the",
        "exact structured <think>/<answer> outputs it produced. Each example",
        "adds explicit notes clarifying what the model did (selection or",
        "opinion prediction), whether tags are present, and a short rationale",
        "summarised from the <think> block.",
        "",
        "Sections are grouped by issue and each includes up to 5 examples.",
        "",
    ])

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

    # Select a balanced mix: 10 total (5 gun, 5 wage) with 5 opinion and 5 next-video.
    def _bin(items: Sequence[Sample], *, issue: str | None = None, task: str | None = None) -> List[Sample]:
        out: List[Sample] = []
        for s in items:
            if issue and s.issue != issue:
                continue
            if task and s.task != task:
                continue
            out.append(s)
        return out

    gun_nv = _bin(nv_samples, issue="gun_control")
    wage_nv = _bin(nv_samples, issue="minimum_wage")
    gun_op = _bin(op_samples, issue="gun_control")
    wage_op = _bin(op_samples, issue="minimum_wage")

    # Decide which issue takes 3 opinion vs 2 opinion based on availability.
    # Aim for: gun_op_quota + wage_op_quota = 5 and gun_total = wage_total = 5.
    gun_op_quota, wage_op_quota = 2, 3
    if len(gun_op) >= 3 and (len(gun_op) >= len(wage_op) or len(wage_op) < 3):
        gun_op_quota, wage_op_quota = 3, 2
    elif len(wage_op) >= 3:
        gun_op_quota, wage_op_quota = 2, 3

    # Clamp by availability first.
    gun_op_take = min(gun_op_quota, len(gun_op))
    wage_op_take = min(wage_op_quota, len(wage_op))
    # If we fell short on one side, try to borrow from the other issue to still reach 5 opinions.
    total_op = gun_op_take + wage_op_take
    if total_op < 5:
        deficit = 5 - total_op
        # Prefer borrowing from the issue with more remaining op samples.
        gun_op_rem = len(gun_op) - gun_op_take
        wage_op_rem = len(wage_op) - wage_op_take
        while deficit > 0 and (gun_op_rem > 0 or wage_op_rem > 0):
            if gun_op_rem >= wage_op_rem and gun_op_rem > 0:
                gun_op_take += 1
                gun_op_rem -= 1
            elif wage_op_rem > 0:
                wage_op_take += 1
                wage_op_rem -= 1
            deficit -= 1
        total_op = gun_op_take + wage_op_take

    # Next-video quotas per issue so that per-issue totals are 5 and total NV=5.
    gun_nv_quota = max(0, 5 - gun_op_take)
    wage_nv_quota = max(0, 5 - wage_op_take)
    # Clamp by availability and ensure total NV aims at 5.
    gun_nv_take = min(gun_nv_quota, len(gun_nv))
    wage_nv_take = min(wage_nv_quota, len(wage_nv))
    total_nv = gun_nv_take + wage_nv_take
    if total_nv < 5:
        deficit = 5 - total_nv
        gun_nv_rem = len(gun_nv) - gun_nv_take
        wage_nv_rem = len(wage_nv) - wage_nv_take
        while deficit > 0 and (gun_nv_rem > 0 or wage_nv_rem > 0):
            if gun_nv_rem >= wage_nv_rem and gun_nv_rem > 0 and gun_nv_take < 5:
                gun_nv_take += 1
                gun_nv_rem -= 1
            elif wage_nv_rem > 0 and wage_nv_take < 5:
                wage_nv_take += 1
                wage_nv_rem -= 1
            deficit -= 1
        total_nv = gun_nv_take + wage_nv_take

    # Assemble selections.
    select_gun: List[Sample] = gun_op[:gun_op_take] + gun_nv[:gun_nv_take]
    select_wage: List[Sample] = wage_op[:wage_op_take] + wage_nv[:wage_nv_take]

    # If any issue exceeds 5 due to borrowing logic, trim.
    select_gun = select_gun[:5]
    select_wage = select_wage[:5]

    # If overall fewer than 10 (due to limited artefacts), top up with any remaining samples.
    total_selected = len(select_gun) + len(select_wage)
    if total_selected < 10:
        pool: List[Sample] = []
        # Prefer to satisfy remaining task quotas first.
        selected_op = sum(1 for s in select_gun + select_wage if s.task == "opinion")
        selected_nv = sum(1 for s in select_gun + select_wage if s.task == "next_video")
        if selected_op < 5:
            pool.extend(gun_op[gun_op_take:] + wage_op[wage_op_take:])
        if selected_nv < 5:
            pool.extend(gun_nv[gun_nv_take:] + wage_nv[wage_nv_take:])
        # Fallback: any remaining samples if still short.
        if not pool:
            pool.extend([s for s in (nv_samples + op_samples) if s not in select_gun and s not in select_wage])
        for s in pool:
            if total_selected >= 10:
                break
            if s.issue == "gun_control" and len(select_gun) < 5:
                select_gun.append(s)
                total_selected += 1
            elif s.issue == "minimum_wage" and len(select_wage) < 5:
                select_wage.append(s)
                total_selected += 1

    # Render sections
    sections = [("gun_control", "Gun Control", select_gun), ("minimum_wage", "Minimum Wage", select_wage)]
    for issue_key, issue_label, selected in sections:
        if not selected:
            continue
        lines.append(f"## {issue_label}")
        lines.append("")
        # Ensure a stable ordering: show opinions first, then next-video, interleave otherwise.
        def _order_key(s: Sample) -> Tuple[int, str]:
            # opinion first (0), then next_video (1); keep original question hash as tiebreaker
            return (0 if s.task == "opinion" else 1, s.question[:60])
        for idx, sample in enumerate(sorted(selected, key=_order_key), start=1):
            lines.extend(_format_sample_block(idx, sample))

    write_markdown_lines(path, lines)


__all__ = [
    "write_sample_responses_report",
]
