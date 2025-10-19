"""Generate the human-readable Markdown report for prompt analytics results.

The functions here translate the structured summaries computed by the
prompt CLI into narrative sections, tables, and figure references so the
cleaning workflow can emit a ready-to-publish README alongside plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ReportFigures:
    """Paths to rendered summary figures."""

    prior_history: Path
    n_options: Path
    demographic: Path


@dataclass
class ReportCounts:
    """Aggregated count metrics used in the report."""

    prior_history: Dict[str, Dict[int, int]]
    n_options: Dict[str, Dict[int, int]]
    demographic_missing: Dict[str, int]
    unique_content: Dict[str, Dict[str, int]]
    participant: Dict[str, Dict[str, Any]]


@dataclass
class ReportSummaries:
    """Feature-level summaries used to render the Markdown report."""

    feature: Dict[str, Dict[str, Dict[str, float | int]]]
    profile: Dict[str, Dict[str, float]]
    counts: ReportCounts
    skipped_features: List[str]


@dataclass
class ReportContext:
    """Full context bundle required to build the prompt report."""

    output_dir: Path
    figures_dir: Path
    summaries: ReportSummaries
    figures: ReportFigures


def _relative_path(path: Path, base: Path) -> str:
    """Return ``path`` relative to ``base`` when possible for cleaner Markdown links."""

    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _format_table(header: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["-" * (len(h) + 2) for h in header]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _profile_section(context: ReportContext) -> List[str]:
    rows: List[List[str]] = []
    summaries = context.summaries.profile
    for split in ("train", "validation"):
        summary = summaries.get(split, {"rows": 0.0, "missing_profile": 0.0})
        total_rows = summary.get("rows", 0.0)
        missing = summary.get("missing_profile", 0.0)
        ratio = f"{(missing / total_rows):.2%}" if total_rows else "0.00%"
        rows.append([
            split,
            str(int(total_rows)),
            str(int(missing)),
            ratio,
        ])
    return [
        "## Profile availability",
        "",
        *_format_table(["Split", "Rows", "Missing profile", "Share missing"], rows),
        "",
    ]


def _prior_history_section(context: ReportContext) -> List[str]:
    rows: List[List[str]] = []
    counts = context.summaries.counts.prior_history
    all_keys = sorted(set(counts["train"]).union(counts["validation"]))
    for key in all_keys:
        rows.append(
            [str(key), str(counts["train"].get(key, 0)), str(counts["validation"].get(key, 0))]
        )
    return [
        "## Prior video counts",
        "",
    ] + _format_table(["Prior videos", "Train", "Validation"], rows) + [""]


def _n_options_section(context: ReportContext) -> List[str]:
    rows: List[List[str]] = []
    counts = context.summaries.counts.n_options
    all_keys = sorted(set(counts["train"]).union(counts["validation"]))
    for key in all_keys:
        rows.append(
            [str(key), str(counts["train"].get(key, 0)), str(counts["validation"].get(key, 0))]
        )
    return [
        "## Slate size distribution (`n_options`)",
        "",
    ] + _format_table(["Slate size", "Train", "Validation"], rows) + [""]


def _unique_content_section(context: ReportContext) -> List[str]:
    counts = context.summaries.counts.unique_content
    lines = [
        "## Unique content counts",
        "",
        "| Split | Current videos | Gold videos | Unique slates | Unique state texts |",
        "|-------|----------------|-------------|---------------|--------------------|",
    ]
    for split in ("train", "validation"):
        split_counts = counts.get(split, {})
        lines.append(
            f"| {split} | {split_counts.get('current_video_ids', 0)} | "
            f"{split_counts.get('gold_video_ids', 0)} | {split_counts.get('slate_texts', 0)} | "
            f"{split_counts.get('state_texts', 0)} |"
        )
    lines.append("")
    return lines


def _participant_section(context: ReportContext) -> List[str]:
    counts = context.summaries.counts.participant
    table_rows: List[List[str]] = []
    for split in ("train", "validation"):
        split_counts = counts.get(split, {})
        by_issue = split_counts.get("by_issue_study", {})
        for issue_name, study_map in sorted(by_issue.items(), key=lambda item: item[0].lower()):
            for study_name, total in sorted(study_map.items(), key=lambda item: item[0].lower()):
                table_rows.append([split, issue_name, study_name, str(total)])
        table_rows.append([split, "all", "all", str(split_counts.get("overall", 0))])

    lines = [
        "## Unique participants per study and issue",
        "",
        *_format_table(["Split", "Issue", "Study", "Participants"], table_rows),
        "",
    ]

    overall_counts = counts.get("overall", {})
    issue_items = sorted(
        overall_counts.get("by_issue", {}).items(),
        key=lambda item: item[0].lower(),
    )
    study_items = sorted(
        overall_counts.get("by_study", {}).items(),
        key=lambda item: item[0].lower(),
    )

    lines.append(f"- Overall participants (all issues): {overall_counts.get('overall', 0)}")
    for issue_name, value in issue_items:
        lines.append(f"- Overall participants for {issue_name}: {value}")
    for study_name, value in study_items:
        lines.append(f"- Overall participants in {study_name}: {value}")
    lines.append("")
    return lines


def _skipped_features_section(skipped_features: List[str]) -> List[str]:
    if not skipped_features:
        return []
    lines = ["## Features skipped due to missing data", ""]
    lines.extend(f"- {feature}" for feature in sorted(skipped_features))
    lines.append("")
    return lines


def _shortfall_lines(overall_counts: Dict[str, Any]) -> List[str]:
    """Summaries comparing expected and cleaned participant totals."""

    overall_by_issue = overall_counts.get("by_issue", {})
    gun_total = overall_by_issue.get("gun_control", 0)
    wage_total = overall_by_issue.get("minimum_wage", 0)

    lines = [
        "- Original study participants: 1,650 (Study 1 — gun rights)",
        "  1,679 (Study 2 — minimum wage MTurk), and 2,715 (Study 3 — minimum wage YouGov).",
        f"- Cleaned dataset participants captured here: {gun_total} (gun control)",
        f"  and {wage_total} (minimum wage).",
        "  Study 4 (Shorts) is excluded because the released interaction logs",
        "  do not contain recommendation slates.",
        "- Shortfall summary (Studies 1–3 only):",
    ]

    expected_by_study = {
        "study1": 1650,
        "study2": 1679,
        "study3": 2715,
    }
    actual_by_study = {
        key: overall_counts.get("by_study", {}).get(key, 0) for key in expected_by_study
    }
    shortfall_notes = {
        "study1": (
            "98 sessions log only the starter clip (`vids` length = 1) and 15 log multiple clips",
            "but no recommendation slate (`displayOrders` empty).",
        ),
        "study2": (
            "14 sessions log only the starter clip; 17 have multiple clips but no slate metadata",
            "(`displayOrders` empty).",
        ),
        "study3": ("No gap — interaction logs are complete.",),
    }
    study_labels = {
        "study1": "Study 1 (gun control MTurk)",
        "study2": "Study 2 (minimum wage MTurk)",
        "study3": "Study 3 (minimum wage YouGov)",
    }
    for key in ("study1", "study2", "study3"):
        expected = expected_by_study[key]
        actual = actual_by_study.get(key, 0)
        delta = expected - actual
        gap_text = "no gap" if delta == 0 else f"{'-' if delta > 0 else '+'}{abs(delta)}"
        lines.append(
            "  - "
            f"{study_labels[key]}: {expected} expected vs. {actual} usable ({gap_text})."
        )
        note = " ".join(shortfall_notes.get(key, ())).strip()
        if note:
            lines.append(f"    {note}")

    lines.extend(
        [
            "- Only gun-control and minimum-wage sessions (Studies 1–3) are retained;",
            "  other topic IDs from the capsule are excluded.",
            "",
        ]
    )
    return lines


def _coverage_section(context: ReportContext) -> List[str]:
    lines = [
        "## Dataset coverage notes",
        "",
        "Builder note: rows missing all survey demographics (age, gender, race, income, etc.)",
        "are dropped during cleaning so every retained interaction has viewer context.",
        "This removes roughly 22% of the ~33k raw interactions.",
        "",
        '> "The short answer is that sessions.json contains EVERYTHING.',
        "Every test run, every study.",
        "In addition to the studies that involved watching videos on the platform,",
        "it also contains sessions from the “First Impressions” study and the “Shorts” study",
        "(Study 4 in the paper).",
        "Those sessions involved no user decisions.",
        "Instead they played predetermined videos that were",
        "either constant or increasing in their extremeness.",
        'All are differentiated by the topicId." — Emily Hu (University of Pennsylvania)',
        "",
    ]

    overall_counts = context.summaries.counts.participant.get("overall", {})
    lines.extend(_shortfall_lines(overall_counts))
    return lines


def build_markdown_report(context: ReportContext) -> List[str]:
    """Return rendered Markdown lines summarizing the prompt statistics."""

    lines: List[str] = [
        "# Prompt feature report",
        "",
        f"Figures directory: `{_relative_path(context.figures_dir, context.output_dir)}`",
        "",
    ]

    figure_paths = [
        ("Prior history distribution", context.figures.prior_history),
        ("Slate size distribution", context.figures.n_options),
        ("Demographic coverage", context.figures.demographic),
    ]
    for caption, fig_path in figure_paths:
        rel_path = _relative_path(fig_path, context.output_dir)
        lines.extend([f"![{caption}]({rel_path})", ""])

    for section in (
        _profile_section(context),
        _prior_history_section(context),
        _n_options_section(context),
        _unique_content_section(context),
        _participant_section(context),
        _skipped_features_section(context.summaries.skipped_features),
        _coverage_section(context),
    ):
        if section:
            lines.extend(section)

    return lines


__all__ = [
    "ReportContext",
    "ReportSummaries",
    "ReportCounts",
    "ReportFigures",
    "build_markdown_report",
]
