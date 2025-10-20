"""Markdown rendering utilities for the political sciences replication."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping

import math


def _is_nan(value: object) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _format_float(value: float, precision: int = 3) -> str:
    if _is_nan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def _format_percent(value: float, precision: int = 1) -> str:
    if _is_nan(value):
        return "n/a"
    return f"{value * 100:.{precision}f}%"


def build_markdown(
    output_dir: Path,
    study_rows: Iterable[Mapping[str, object]],
    heatmap_paths: Iterable[Path],
    mean_change_path: Path,
) -> List[str]:
    """Render Markdown lines describing the replication results."""

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
        "## Short-term exposure to filter-bubble recommendation systems has limited polarization effects",
        "",
        "This section replicates headline opinion-shift findings from "
        "_Short-term exposure to filter-bubble recommendation systems has limited polarization effects: "
        "Naturalistic experiments on YouTube_ (Liu et al., PNAS 2025) using the cleaned data in this repository.",
        "",
        "### Opinion shift summary",
        "",
        f"| {' | '.join(header_cells)} |",
        f"| {' | '.join(separator_cells)} |",
    ]

    for row in study_rows:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{row['summary']['n']:.0f} | "
            f"{_format_float(row['summary']['mean_before'])} | "
            f"{_format_float(row['summary']['mean_after'])} | "
            f"{_format_float(row['summary']['mean_change'])} | "
            f"{_format_float(row['summary']['median_change'])} | "
            f"{_format_percent(row['summary']['share_increase'])} | "
            f"{_format_percent(row['summary']['share_decrease'])} | "
            f"{_format_percent(row['summary']['share_small_change'])} |"
        )

    lines.extend(
        [
            "",
            "The minimal mean shifts and high share of small opinion changes (|Δ| ≤ 0.05 on a 0–1 scale) "
            "mirror the paper's conclusion that short-term algorithmic perturbations produced limited "
            "polarization in Studies 1–3.",
            "",
            "### Pre/post opinion heatmaps",
            "",
        ]
    )

    for heatmap_path in heatmap_paths:
        rel_path = Path(heatmap_path).relative_to(output_dir)
        title = rel_path.stem.replace("_", " ").title()
        lines.extend(
            [
                f"![{title}]({rel_path.as_posix()})",
                "",
            ]
        )

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
        "Participants lacking a post-wave response are excluded from the relevant heatmap and summary."
    )

    return lines
