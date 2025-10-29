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

"""Markdown rendering utilities for the political sciences replication.

These helpers convert the computed statistics into narrative text, tables,
and figure callouts that mirror the published study's presentation. All
renderers are provided under the repository's Apache 2.0 license; see
LICENSE for the detailed terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import math


@dataclass(frozen=True)
class MarkdownArtifacts:
    """Optional collections required for extended report sections."""

    assignment_rows: Iterable[Mapping[str, object]] = ()
    regression_summary: Optional[Mapping[str, float]] = None
    policy_rows: Iterable[Mapping[str, object]] = ()


def _is_nan(value: object) -> bool:
    """Return ``True`` when a value should be treated as missing/NaN.

    The helper mirrors the original R scripts by coercing to float,
    treating conversion failures as ``NaN`` so downstream formatting can
    emit ``\"n/a\"`` markers in the Markdown tables.

    :param value: Candidate numeric value from the summaries.
    :returns: ``True`` when the input is not a finite number.
    """

    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _format_float(value: float, precision: int = 3) -> str:
    """Format a scalar with fixed precision, returning ``\"n/a\"`` for NaNs.

    :param value: Numeric value to display.
    :param precision: Number of decimal places to include.
    :returns: Formatted scalar string or ``\"n/a\"`` when missing.
    """

    if _is_nan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def _format_percent(value: float, precision: int = 1) -> str:
    """Format a proportion as a percentage string with fallback for NaNs.

    :param value: Proportion in ``[0, 1]`` to convert.
    :param precision: Number of decimal places to display.
    :returns: Percentage string or ``\"n/a\"`` when the input is invalid.
    """

    if _is_nan(value):
        return "n/a"
    return f"{value * 100:.{precision}f}%"


def _format_interval(center: float, lower: float, upper: float, precision: int = 3) -> str:
    """Render a signed estimate with its confidence interval bounds.

    :param center: Point estimate.
    :param lower: Lower confidence bound.
    :param upper: Upper confidence bound.
    :param precision: Decimal places applied to each value.
    :returns: Formatted string ``\"+x.xxx [+l.lll, +u.uuu]\"`` or ``\"n/a\"``.
    """

    if any(_is_nan(item) for item in (center, lower, upper)):
        return "n/a"
    return f"{center:+.{precision}f} [{lower:+.{precision}f}, {upper:+.{precision}f}]"


def _opinion_summary_lines(study_rows: Iterable[Mapping[str, object]]) -> List[str]:
    """Render Markdown rows describing opinion shift summaries.

    :param study_rows: Iterable containing dictionaries with study labels and summary stats.
    :returns: Markdown table row strings summarizing opinion shifts per study.
    """

    lines: List[str] = []
    for row in study_rows:
        summary = row["summary"]
        lines.append(
            "| "
            f"{row['label']} | "
            f"{summary['n']:.0f} | "
            f"{_format_float(summary['mean_before'])} | "
            f"{_format_float(summary['mean_after'])} | "
            f"{_format_float(summary['mean_change'])} | "
            f"{_format_float(summary['median_change'])} | "
            f"{_format_percent(summary['share_increase'])} | "
            f"{_format_percent(summary['share_decrease'])} | "
            f"{_format_percent(summary['share_small_change'])} |"
        )
    return lines


def _heatmap_section(heatmap_paths: Iterable[Path], output_dir: Path) -> List[str]:
    """Build Markdown image blocks for the heatmap paths.

    :param heatmap_paths: Iterable of paths to generated heatmap images.
    :param output_dir: Base directory used to compute relative image paths.
    :returns: Markdown lines embedding each heatmap image.
    """

    blocks: List[str] = []
    for heatmap_path in heatmap_paths:
        rel_path = Path(heatmap_path).relative_to(output_dir)
        title = rel_path.stem.replace("_", " ").title()
        blocks.extend(
            [
                f"![{title}]({rel_path.as_posix()})",
                "",
            ]
        )
    return blocks


def _assignment_section(rows: Iterable[Mapping[str, object]]) -> List[str]:
    """Render the control vs. treatment summary table when data is available.

    :param rows: Iterable of dictionaries describing study-level control/treatment changes.
    :returns: Markdown lines forming the summary table (or empty when no rows provided).
    """

    material = list(rows)
    if not material:
        return []

    lines: List[str] = [
        "",
        "### Control vs. treatment summary",
        "",
        "| Study | Control Δ | Treatment Δ |",
        "| ------ | ---------- | ------------ |",
    ]
    for row in material:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{_format_float(row['control_mean_change'])} | "
            f"{_format_float(row['treatment_mean_change'])} |"
        )
    lines.append("")
    return lines


def _policy_section(rows: Iterable[Mapping[str, object]]) -> List[str]:
    """Render the preregistered stratified contrast table when present.

    :param rows: Iterable of dictionaries representing stratified contrast results.
    :returns: Markdown lines describing the contrast table (or empty when no rows).
    """

    material = list(rows)
    if not material:
        return []

    lines: List[str] = [
        "### Preregistered stratified contrasts",
        "",
        "| Study | Cell | Outcome | Effect (95% CI) | MDE (80% power) | q-value | N |",
        "| ------ | ---- | ------- | ---------------- | ---------------- | ------- | --- |",
    ]
    for row in material:
        lines.append(
            "| "
            f"{row['study_label']} | "
            f"{row['contrast_label']} | "
            f"{row['outcome_label']} | "
            f"{_format_interval(row['estimate'], row['ci_low'], row['ci_high'])} | "
            f"{_format_float(row['mde'])} | "
            f"{_format_float(row['p_adjusted'])} | "
            f"{int(row.get('n', 0))} |"
        )
    lines.append(
        "q-values reflect the paper's hierarchical FDR correction applied "
        "within each outcome family."
    )
    return lines


def build_markdown(
    output_dir: Path,
    study_rows: Iterable[Mapping[str, object]],
    heatmap_paths: Iterable[Path],
    mean_change_path: Path,
    artifacts: Optional[MarkdownArtifacts] = None,
) -> List[str]:
    """Render Markdown lines describing the replication results.

    :param output_dir: Directory where referenced assets are written.
    :param study_rows: Iterable of per-study summary dictionaries.
    :param heatmap_paths: Paths to generated heatmap image files.
    :param mean_change_path: Path to the combined mean-change figure.
    :param artifacts: Optional container with supplemental report inputs.
    :returns: List of Markdown lines ready to be written to disk.
    """

    artifacts = artifacts or MarkdownArtifacts()
    assignment_rows = list(artifacts.assignment_rows)
    policy_rows = list(artifacts.policy_rows)

    output_dir = Path(output_dir)
    header_cells = [
        "Study",
        "Participants",
        "Mean pre",
        "Mean post",
        "Mean change",
        "Median change",
        "Share ↑",
        "Share ↓",
        "Share \\|Δ\\| ≤ 0.05",
    ]
    separator_cells = [
        "------",
        "--------------",
        "----------",
        "-----------",
        "-------------",
        "---------------",
        "---------",
        "---------",
        "-----------",
    ]
    lines: List[str] = [
        "# RESEARCH ARTICLE POLITICAL SCIENCES",
        "",
        "## Short-term exposure to filter-bubble recommendation systems has limited "
        "polarization effects",
        "",
        "This section replicates headline opinion-shift findings from "
        "_Short-term exposure to filter-bubble recommendation systems has limited "
        "polarization effects: Naturalistic experiments on YouTube_ (Liu et al., "
        "PNAS 2025) using the cleaned data in this repository.",
        "",
        "### Opinion shift summary",
        "",
        f"| {' | '.join(header_cells)} |",
        f"| {' | '.join(separator_cells)} |",
    ]

    lines.extend(_opinion_summary_lines(study_rows))

    lines.extend(
        [
            "",
            "The minimal mean shifts and high share of small opinion changes "
            "(|Δ| ≤ 0.05 on a 0–1 scale) mirror the paper's conclusion that "
            "short-term algorithmic perturbations produced limited "
            "polarization in Studies 1–3.",
            "",
            "### Pre/post opinion heatmaps",
            "",
        ]
    )

    lines.extend(_heatmap_section(heatmap_paths, output_dir))

    rel_mean_change_path = Path(mean_change_path).relative_to(output_dir)
    lines.extend(
        [
            "### Control vs. treatment shifts and pooled regression",
            "",
            f"![Mean opinion change]({rel_mean_change_path.as_posix()})",
            "",
            "The first three panels separate mean opinion changes for the control and "
            "treatment arms of Studies 1–3 with 95% confidence intervals. "
            "The fourth panel reports the pooled regression coefficient comparing treatment versus "
            "control after adjusting for baseline opinion and study fixed effects.",
            "",
        ]
    )

    lines.append(
        "Replication notes: opinion indices are scaled to [0, 1] and "
        "computed from the same survey composites used in the published study. "
        "Participants lacking a post-wave response are excluded from the relevant "
        "heatmap and summary."
    )

    lines.extend(_assignment_section(assignment_rows))

    if artifacts.regression_summary:
        lines.extend(
            [
                "Pooled regression (control-adjusted) β̂ ≈ "
                f"{_format_float(artifacts.regression_summary.get('coefficient', float('nan')))} "
                f"with p ≈ {artifacts.regression_summary.get('p_value', float('nan')):.2e}.",
                "",
            ]
        )

    lines.extend(_policy_section(policy_rows))

    return lines
