"""Markdown rendering utilities for the political sciences replication."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping


def _format_float(value: float, precision: int = 3) -> str:
    if value != value:  # NaN check
        return "n/a"
    return f"{value:.{precision}f}"


def _format_percent(value: float, precision: int = 1) -> str:
    if value != value:
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
        "| Study | Participants | Mean pre | Mean post | Mean change | Median change | Share ↑ | Share ↓ | |Δ| ≤ 0.05 |",
        "|-------|--------------|----------|-----------|-------------|---------------|---------|---------|-----------|",
    ]

    for row in study_rows:
        lines.append(
            "| {label} | {n:.0f} | {mean_pre} | {mean_post} | {mean_change} | {median_change} | "
            "{share_inc} | {share_dec} | {share_small} |".format(
                label=row["label"],
                n=row["summary"]["n"],
                mean_pre=_format_float(row["summary"]["mean_before"]),
                mean_post=_format_float(row["summary"]["mean_after"]),
                mean_change=_format_float(row["summary"]["mean_change"]),
                median_change=_format_float(row["summary"]["median_change"]),
                share_inc=_format_percent(row["summary"]["share_increase"]),
                share_dec=_format_percent(row["summary"]["share_decrease"]),
                share_small=_format_percent(row["summary"]["share_small_change"]),
            )
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
            "### Mean shifts with 95% confidence intervals",
            "",
            f"![Mean opinion change]({rel_mean_change_path.as_posix()})",
            "",
        ]
    )

    lines.append(
        "Replication notes: opinion indices are scaled to [0, 1] and "
        "computed from the same survey composites used in the published study. "
        "Participants lacking a post-wave response are excluded from the relevant heatmap and summary."
    )

    return lines
