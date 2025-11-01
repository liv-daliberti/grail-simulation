"""Assemble Markdown lines for the sample responses report.

This module collects examples for both tasks, selects a subset per issue,
and renders readable sections. The result is a list of Markdown lines to be
written by the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from .samples_collect import collect_next_video_samples, collect_opinion_samples
from .samples_select import select_issue_samples
from .samples_render import build_header_lines, render_sections_by_issue


def build_sample_report_lines(
    *,
    family_label: str,
    next_video_files: Sequence[Path],
    opinion_files: Sequence[Path],
    per_issue: int = 5,
) -> List[str]:
    """Return all Markdown lines for the sample responses report."""

    lines: List[str] = build_header_lines(family_label)

    nv_samples = collect_next_video_samples(list(next_video_files)) if next_video_files else []
    op_samples = collect_opinion_samples(list(opinion_files)) if opinion_files else []

    if not (nv_samples or op_samples):
        lines.append(
            (
                "No examples found in existing predictions. Populate "
                "models/*/predictions.jsonl first."
            )
        )
        lines.append("")
        return lines

    select_gun, select_wage = select_issue_samples(nv_samples, op_samples, per_issue=per_issue)
    lines.extend(render_sections_by_issue(select_gun, select_wage))
    return lines


__all__ = ["build_sample_report_lines"]
