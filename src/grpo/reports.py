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

"""Markdown report builders for GRPO baselines."""

from __future__ import annotations

from pathlib import Path

from common.pipeline.io import write_markdown_lines

from .next_video import NextVideoEvaluationResult
from .opinion import OpinionEvaluationResult


def _format_rate(value: float | int | None, precision: int = 3) -> str:
    """Return a formatted rate with the requested precision.

    :param value: Numeric rate to format; ``None`` renders as an em dash.
    :param precision: Decimal precision for floating-point values.
    :returns: Human-readable string suitable for Markdown tables.
    :rtype: str
    """

    if value is None:
        return "—"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "—"


def _format_int(value: int | float | None) -> str:
    """Return an integer string with thousand separators.

    :param value: Numeric value to coerce into an integer.
    :returns: Formatted integer string or an em dash when unavailable.
    :rtype: str
    """

    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _write_catalog_readme(reports_root: Path) -> None:
    """Materialise the top-level reports README.

    :param reports_root: Root directory for GRPO reports.
    :returns: ``None``. Markdown content is written to disk.
    """

    lines = [
        "# GRPO Report Catalog",
        "",
        "Finetuned GRPO evaluation artifacts:",
        "",
        "- `next_video/` – slate-ranking metrics for the configured checkpoint.",
        "- `opinion/` – opinion regression metrics across participant studies.",
        "",
        "Regenerate via `python -m grpo.pipeline --stage full` after producing "
        "updated evaluation artifacts under `models/grpo/`.",
        "",
    ]
    write_markdown_lines(reports_root / "README.md", lines)


def _write_next_video_report(
    reports_root: Path,
    result: NextVideoEvaluationResult,
) -> None:
    """Render the next-video evaluation summary.

    :param reports_root: Root directory for the GRPO reports tree.
    :param result: Evaluation result containing next-video metrics.
    :returns: ``None``. Markdown files are written to disk.
    """

    metrics = result.metrics
    accuracy = metrics.get("accuracy_overall")
    parsed_rate = metrics.get("parsed_rate")
    format_rate = metrics.get("format_rate")
    eligible = metrics.get("n_eligible")
    total = metrics.get("n_total")

    lines = [
        "# GRPO Next-Video Baseline",
        "",
        f"- **Overall accuracy:** {_format_rate(accuracy)} on {_format_int(eligible)} "
        f"eligible slates out of {_format_int(total)} processed.",
        f"- **Parsed rate:** {_format_rate(parsed_rate)}",
        f"- **Formatted rate:** {_format_rate(format_rate)}",
        "",
        "## Accuracy by Issue",
        "",
        "| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_issue = metrics.get("group_metrics", {}).get("by_issue", {})
    for group, stats in by_issue.items():
        lines.append(
            f"| {group} | {_format_int(stats.get('n_seen'))} | "
            f"{_format_int(stats.get('n_eligible'))} | "
            f"{_format_rate(stats.get('accuracy'))} | "
            f"{_format_rate(stats.get('parsed_rate'))} | "
            f"{_format_rate(stats.get('format_rate'))} |"
        )

    lines.extend(
        [
            "",
            "## Accuracy by Participant Study",
            "",
            "| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    by_study = metrics.get("group_metrics", {}).get("by_participant_study", {})
    for group, stats in by_study.items():
        lines.append(
            f"| {group} | {_format_int(stats.get('n_seen'))} | "
            f"{_format_int(stats.get('n_eligible'))} | "
            f"{_format_rate(stats.get('accuracy'))} | "
            f"{_format_rate(stats.get('parsed_rate'))} | "
            f"{_format_rate(stats.get('format_rate'))} |"
        )

    lines.extend(
        [
            "",
            "### Notes",
            "",
            "Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.",
            "",
        ]
    )
    write_markdown_lines(reports_root / "next_video" / "README.md", lines)


def _write_opinion_report(
    reports_root: Path,
    result: OpinionEvaluationResult,
) -> None:
    """Render the opinion regression summary.

    :param reports_root: Root directory for the GRPO reports tree.
    :param result: Opinion evaluation result providing metrics.
    :returns: ``None``. Markdown files are written to disk.
    """

    lines = [
        "# GRPO Opinion Regression",
        "",
        "Opinion-shift evaluation across the canonical participant studies. "
        "Baseline metrics treat the pre-study opinion index as the prediction.",
        "",
        "## Combined Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    combined = result.combined_metrics
    combined_rows = [
        ("Eligible", _format_int(combined.get("eligible"))),
        ("MAE (post-study)", _format_rate(combined.get("mae_after"))),
        ("MAE (change)", _format_rate(combined.get("mae_change"))),
        ("Direction accuracy", _format_rate(combined.get("direction_accuracy"))),
        ("RMSE (post-study)", _format_rate(combined.get("rmse_after"))),
        ("RMSE (change)", _format_rate(combined.get("rmse_change"))),
        ("Calibration ECE", _format_rate(combined.get("calibration_ece"))),
    ]
    for label, value in combined_rows:
        lines.append(f"| {label} | {value} |")

    lines.extend(
        [
            "",
            "## Per-Study Breakdown",
            "",
            (
                "| Study | Participants | Eligible | MAE ↓ | Baseline MAE ↓ | "
                "Direction ↑ | Baseline Direction ↑ |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    studies = result.studies
    study_iter = studies.values() if hasattr(studies, "values") else studies

    for study_result in study_iter:
        metrics = study_result.metrics
        baseline = study_result.baseline
        study_label = study_result.spec.label if study_result.spec else study_result.study_label
        eligible = study_result.eligible or metrics.get("eligible")
        row = (
            f"| {study_label} | {_format_int(study_result.participants)} | "
            f"{_format_int(eligible)} | "
            f"{_format_rate(metrics.get('mae_after'))} | "
            f"{_format_rate(baseline.get('mae_after'))} | "
            f"{_format_rate(metrics.get('direction_accuracy'))} | "
            f"{_format_rate(baseline.get('direction_accuracy'))} |"
        )
        lines.append(row)

    lines.append("")
    write_markdown_lines(reports_root / "opinion" / "README.md", lines)


def generate_reports(
    *,
    repo_root: Path,
    next_video: NextVideoEvaluationResult | None,
    opinion: OpinionEvaluationResult | None,
) -> None:
    """Materialise GRPO Markdown reports.

    :param repo_root: Root of the repository where reports are rendered.
    :param next_video: Optional next-video evaluation artefacts.
    :param opinion: Optional opinion evaluation artefacts.
    :returns: ``None``. Markdown reports are generated on disk.
    """

    reports_root = repo_root / "reports" / "grpo"
    (reports_root / "next_video").mkdir(parents=True, exist_ok=True)
    (reports_root / "opinion").mkdir(parents=True, exist_ok=True)

    _write_catalog_readme(reports_root)
    if next_video is not None:
        _write_next_video_report(reports_root, next_video)
    if opinion is not None:
        _write_opinion_report(reports_root, opinion)
