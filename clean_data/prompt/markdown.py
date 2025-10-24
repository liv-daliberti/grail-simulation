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

"""Generate the human-readable Markdown report for prompt analytics results.

The functions here translate the structured summaries computed by the
prompt CLI into narrative sections, tables, and figure references so the
cleaning workflow can emit a ready-to-publish README alongside plots. This
reporting module is distributed under the repository's Apache 2.0 license;
consult LICENSE for the governing terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ReportFigures:
    """Paths to rendered summary figures.

    :param prior_history: Location of the prior-history distribution plot.
    :param n_options: Location of the slate-size distribution plot.
    :param demographic: Location of the demographic coverage plot.
    """

    prior_history: Path
    n_options: Path
    demographic: Path


@dataclass
class ReportCounts:
    """Aggregated count metrics used in the report.

    :param prior_history: Tallies of prior-history counts per split.
    :param n_options: Slate-size count distributions per split.
    :param demographic_missing: Mapping of demographic completeness metrics.
    :param unique_content: Coverage counts of unique videos and slates.
    :param participant: Participant totals and breakdowns by split.
    """

    prior_history: Dict[str, Dict[int, int]]
    n_options: Dict[str, Dict[int, int]]
    demographic_missing: Dict[str, Dict[str, float]]
    unique_content: Dict[str, Dict[str, int]]
    participant: Dict[str, Dict[str, Any]]


@dataclass
class ReportSummaries:
    """Feature-level summaries used to render the Markdown report.

    :param feature: Aggregated feature statistics per split.
    :param profile: Profile coverage metrics per split.
    :param counts: Structured count bundle for the report.
    :param coverage: Dataset coverage summaries by split.
    :param skipped_features: Features omitted from reporting.
    """

    feature: Dict[str, Dict[str, Dict[str, float | int]]]
    profile: Dict[str, Dict[str, float]]
    counts: ReportCounts
    coverage: Dict[str, Dict[str, Any]]
    skipped_features: List[str]


@dataclass
class ReportContext:
    """Full context bundle required to build the prompt report.

    :param output_dir: Destination directory for generated Markdown files.
    :param figures_dir: Directory containing rendered plots.
    :param summaries: Aggregated summaries describing the dataset.
    :param figures: Paths to figures referenced in the report.
    """

    output_dir: Path
    figures_dir: Path
    summaries: ReportSummaries
    figures: ReportFigures


def _relative_path(path: Path, base: Path) -> str:
    """Return ``path`` relative to ``base`` when possible for cleaner Markdown links.

    :param path: Path that should be relativised.
    :param base: Base directory used for relative calculations.
    :returns: Relative path string when possible, otherwise the absolute path.
    """

    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _format_table(header: List[str], rows: List[List[str]]) -> List[str]:
    """Return Markdown table lines for the given header and rows.

    :param header: List of column headers.
    :param rows: Sequence of rows, each already converted to strings.
    :returns: List of Markdown-formatted strings representing the table.
    """

    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["-" * (len(h) + 2) for h in header]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _profile_section(context: ReportContext) -> List[str]:
    """Render the profile availability section.

    :param context: Report context containing profile summaries.
    :returns: Markdown lines describing profile coverage per split.
    """

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


def _demographic_section(context: ReportContext) -> List[str]:
    """Render the demographic completeness section.

    :param context: Report context containing demographic counts.
    :returns: Markdown lines describing demographic coverage per split.
    """

    counts = context.summaries.counts.demographic_missing
    rows: List[List[str]] = []
    for split in ("train", "validation", "overall"):
        data = counts.get(split)
        if not data:
            continue
        total = int(data.get("total", 0))
        missing = int(data.get("missing", 0))
        share = data.get("share", 0.0) * 100
        rows.append(
            [
                split,
                str(total),
                str(missing),
                f"{share:.2f}%",
            ]
        )
    if not rows:
        return []
    return [
        "## Demographic completeness",
        "",
        *_format_table(["Split", "Rows", "Missing all demographics", "Share"], rows),
        "",
    ]


def _prior_history_section(context: ReportContext) -> List[str]:
    """Render the prior-history summary section.

    :param context: Report context containing prior history counts.
    :returns: Markdown lines describing prior-history distributions.
    """

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
    """Render the slate size summary section.

    :param context: Report context containing ``n_options`` counts.
    :returns: Markdown lines describing slate-size distributions.
    """

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
    """Render the unique content coverage section.

    :param context: Report context containing unique content counts.
    :returns: Markdown lines describing coverage per split.
    """

    counts = context.summaries.counts.unique_content
    rows: List[List[str]] = []
    for split in ("train", "validation", "overall"):
        split_counts = counts.get(split)
        if not split_counts:
            continue
        rows.append(
            [
                split,
                str(split_counts.get("current_video_ids", 0)),
                str(split_counts.get("gold_video_ids", 0)),
                str(split_counts.get("candidate_video_ids", 0)),
                str(split_counts.get("slate_combinations", 0)),
                str(
                    split_counts.get(
                        "prompt_texts",
                        split_counts.get("state_texts", 0),
                    )
                ),
            ]
        )
    return [
        "## Unique content coverage",
        "",
        *_format_table(
            [
                "Split",
                "Current videos",
                "Gold videos",
                "Candidate videos",
                "Unique slates",
                "Prompt texts",
            ],
            rows,
        ),
        "",
    ]


def _participant_section(context: ReportContext) -> List[str]:
    """Render the participant coverage section.

    :param context: Report context containing participant counts.
    :returns: Markdown lines listing participant totals per split.
    """

    counts = context.summaries.counts.participant
    rows: List[List[str]] = []
    for split in ("train", "validation", "overall"):
        split_counts = counts.get(split)
        if not split_counts:
            continue
        rows.append([split, str(split_counts.get("overall", 0))])

    lines: List[str] = [
        "## Unique participants",
        "",
    ]

    if rows:
        lines.extend(_format_table(["Split", "Participants (all issues)"], rows))
        lines.append("")

    study_keys = sorted(
        {
            study
            for split_stats in counts.values()
            for study in split_stats.get("by_study", {}).keys()
        }
    )
    if study_keys:
        def _study_header(name: str) -> str:
            """Return a human-readable table header for the study identifier.

            :param name: Raw study label (e.g. ``study2`` or ``gun_control``).
            :returns: Nicely formatted heading used in the Markdown table.
            """

            if name.lower().startswith("study"):
                suffix = name[len("study") :]
                return f"Study {suffix}" if suffix else "Study"
            return name.replace("_", " ").title()

        study_headers = ["Split"] + [_study_header(study) for study in study_keys]
        study_rows: List[List[str]] = []
        for split in ("train", "validation", "overall"):
            split_stats = counts.get(split)
            if not split_stats:
                continue
            study_counts = split_stats.get("by_study", {})
            study_rows.append(
                [split]
                + [str(study_counts.get(study, 0)) for study in study_keys]
            )
        lines.append("### Participants by study")
        lines.append("")
        lines.extend(_format_table(study_headers, study_rows))
        lines.append("")
        lines.append(
            "_Study labels: study1 = gun control (MTurk), "
            "study2 = minimum wage (MTurk), study3 = minimum wage (YouGov)._"
        )
        lines.append("")

    return lines


def _skipped_features_section(skipped_features: List[str]) -> List[str]:
    """Render the skipped-features section.

    :param skipped_features: List of feature identifiers omitted from reporting.
    :returns: Markdown lines documenting skipped features.
    """

    if not skipped_features:
        return []
    lines = ["## Features skipped due to missing data", ""]
    lines.extend(f"- {feature}" for feature in sorted(skipped_features))
    lines.append("")
    return lines


def _shortfall_lines(  # pylint: disable=too-many-locals
    overall_counts: Dict[str, Any]
) -> List[str]:
    """Summaries comparing expected and cleaned participant totals.

    :param overall_counts: Participant counts broken down by issue and study.
    :returns: Explanatory Markdown lines describing participant gaps.
    """

    overall_by_issue = overall_counts.get("by_issue", {})
    issue_labels = {
        "gun_control": "gun control",
        "minimum_wage": "minimum wage",
    }
    issue_descriptions: List[str] = []
    for key in ("gun_control", "minimum_wage"):
        if key in overall_by_issue:
            issue_descriptions.append(
                f"{overall_by_issue[key]} ({issue_labels.get(key, key.replace('_', ' '))})"
            )
    issue_line: str
    duplicate_note: Optional[str] = None
    if issue_descriptions:
        issue_line = (
            "- Cleaned dataset participants captured here: "
            + " and ".join(issue_descriptions)
            + "."
        )
        total_by_issue = sum(overall_by_issue.values())
        overall_total = overall_counts.get("overall", 0)
        overlap = total_by_issue - overall_total
        if overlap > 0:
            duplicate_note = (
                f"  {overlap} participant{'s' if overlap != 1 else ''} appear in both issues, "
                f"so the unique total is {overall_total}."
            )
    else:
        issue_line = (
            "- Cleaned dataset participants captured here (all issues): "
            f"{overall_counts.get('overall', 0)}."
        )

    lines = [
        "- Original study participants: 1,650 (Study 1 — gun rights)",
        "  1,679 (Study 2 — minimum wage MTurk), and 2,715 (Study 3 — minimum wage YouGov).",
        issue_line,
    ]
    if duplicate_note:
        lines.append(duplicate_note)
    lines.extend(
        [
            "  Study 4 (Shorts) is excluded because the released interaction logs",
            "  do not contain recommendation slates.",
            "- Shortfall summary (Studies 1–3 only):",
        ]
    )

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
    """Render the dataset coverage notes section.

    :param context: Report context containing coverage summaries.
    :returns: Markdown lines documenting dataset coverage decisions.
    """

    coverage = context.summaries.coverage
    lines = ["## Dataset coverage notes", ""]
    lines.append(
        "Statistics and charts focus on the core study sessions (study1–study3) "
        "covering the `gun_control` and `minimum_wage` issues, using the full "
        "retained rows (no within-session subsetting)."
    )
    lines.append("")
    overall_stats = coverage.get("overall", {})
    overall_total = int(overall_stats.get("total_rows", 0))
    for split in ("train", "validation"):
        stats = coverage.get(split, {})
        total = int(stats.get("total_rows", 0))
        excluded = int(stats.get("excluded_rows", 0))
        share = (total / overall_total * 100) if overall_total else 0.0
        message = f"- {split.title()}: {total} rows ({share:.1f}% of dataset)"
        if excluded:
            breakdown = stats.get("excluded_by_study", {})
            message += f"; {excluded} excluded rows"
            if breakdown:
                parts = ", ".join(
                    f"{study}: {count}"
                    for study, count in sorted(breakdown.items(), key=lambda item: item[0])
                )
                message += f"; excluded rows from {parts}"
        lines.append(message)
    lines.append("")
    lines.extend(
        [
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
    )
    overall_counts = context.summaries.counts.participant.get("overall", {})
    lines.extend(_shortfall_lines(overall_counts))
    return lines


def build_markdown_report(context: ReportContext) -> List[str]:
    """Return rendered Markdown lines summarizing the prompt statistics.

    :param context: Full set of summary statistics, counts, and plots.
    :returns: Markdown lines suitable for writing to a README.
    """

    lines: List[str] = [
        "# Prompt feature report",
        "",
    ]

    coverage_section = _coverage_section(context)
    if coverage_section:
        lines.extend(coverage_section)

    lines.extend(
        [
            f"Figures directory: `{_relative_path(context.figures_dir, context.output_dir)}`",
            "",
        ]
    )

    figure_paths = [
        ("Prior history distribution", context.figures.prior_history),
        ("Slate size distribution", context.figures.n_options),
        ("Demographic coverage", context.figures.demographic),
    ]
    for caption, fig_path in figure_paths:
        rel_path = _relative_path(fig_path, context.output_dir)
        lines.extend([f"![{caption}]({rel_path})", ""])

    for section in (
        _demographic_section(context),
        _profile_section(context),
        _prior_history_section(context),
        _n_options_section(context),
        _unique_content_section(context),
        _participant_section(context),
        _skipped_features_section(context.summaries.skipped_features),
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
