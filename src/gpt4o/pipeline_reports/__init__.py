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

"""Helpers for assembling GPT-4o Markdown reports."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, cast
from importlib import import_module
from ..core.opinion import OpinionEvaluationResult
from ..pipeline.models import SweepOutcome
write_markdown_lines = import_module("common.pipeline.io").write_markdown_lines
start_markdown_report = import_module("common.reports.utils").start_markdown_report


@dataclass(frozen=True)
class ReportContext:
    """Container for locations required when generating reports."""

    reports_dir: Path
    repo_root: Path


def trigger_report_generation(
    context: ReportContext,
    outcomes: Sequence[SweepOutcome],
    selected: SweepOutcome,
    final_metrics: Mapping[str, object],
    opinion_result: OpinionEvaluationResult | None,
) -> None:
    """
    Convenience wrapper to invoke :func:`run_report_generation`.

    Accepts positional arguments to simplify call sites that already gather the
    required report inputs.
    """

    run_report_generation(
        context=context,
        outcomes=outcomes,
        selected=selected,
        final_metrics=final_metrics,
        opinion_result=opinion_result,
    )


def run_report_generation(
    *,
    context: ReportContext,
    outcomes: Sequence[SweepOutcome],
    selected: SweepOutcome,
    final_metrics: Mapping[str, object],
    opinion_result: OpinionEvaluationResult | None,
) -> None:
    """
    Build a :class:`ReportContext` and regenerate GPT-4o reports.

    :param context: Path metadata used when generating reports.
    :param outcomes: Sweep outcomes captured during evaluation.
    :param selected: Promoted sweep configuration outcome.
    :param final_metrics: Metrics mapping for the selected configuration.
    :param opinion_result: Opinion evaluation bundle associated with the configuration.
    :returns: ``None``.
    """

    generate_reports(
        context=context,
        outcomes=outcomes,
        selected=selected,
        final_metrics=final_metrics,
        opinion_result=opinion_result,
    )


def _format_rate(value: float) -> str:
    """
    Format a numeric rate with three decimal places.

    :param value: Floating-point rate to be rendered.
    :returns: String containing the value rounded to three decimal places.
    """
    return f"{value:.3f}"


def _group_highlights(payload: Mapping[str, Mapping[str, object]]) -> List[str]:
    """
    Compute textual highlights for group-level accuracy extremes.

    :param payload: Nested mapping of group identifiers to metric dictionaries.
    :returns: Bullet list summarising the highest and lowest performing groups.
    """

    entries: List[Tuple[float, str, int]] = []
    for raw_group, stats in payload.items():
        accuracy = stats.get("accuracy")
        try:
            accuracy_value = float(accuracy)
        except (TypeError, ValueError):
            continue
        eligible_raw = stats.get("n_eligible", 0)
        try:
            eligible_value = int(eligible_raw)
        except (TypeError, ValueError):
            eligible_value = 0
        group_name = str(raw_group or "unspecified")
        entries.append((accuracy_value, group_name, eligible_value))
    if not entries:
        return []
    entries.sort(key=lambda item: item[0], reverse=True)
    lines = [
        f"- Highest accuracy: {entries[0][1]} "
        f"({_format_rate(entries[0][0])}, eligible {entries[0][2]})."
    ]
    if len(entries) > 1:
        lowest = entries[-1]
        lines.append(
            f"- Lowest accuracy: {lowest[1]} "
            f"({_format_rate(lowest[0])}, eligible {lowest[2]})."
        )
    return lines


def _write_catalog_report(reports_dir: Path) -> None:
    """
    Create the top-level GPT-4o report catalog README.

    :param reports_dir: Directory where the catalog README should be written.
    :returns: ``None``.
    """

    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "README.md"
    lines = [
        "# GPT-4o Report Catalog",
        "",
        "Generated artifacts for the GPT-4o slate-selection baseline:",
        "",
        "- `next_video/` – summary metrics and fairness cuts for the selected configuration.",
        "- `opinion/` – opinion-shift regression metrics across participant studies.",
        "- `hyperparameter_tuning/` – sweep results across temperature and max token settings.",
        "",
        "Model predictions and metrics JSON files live under `models/gpt-4o/`.",
        "",
    ]
    write_markdown_lines(path, lines)


def _write_sweep_report(
    directory: Path,
    outcomes: Sequence[SweepOutcome],
    selected: SweepOutcome,
) -> None:
    """
    Write the hyper-parameter sweep report summarising all outcomes.

    :param directory: Destination directory for the report.
    :param outcomes: Sequence of sweep outcomes collected during evaluation.
    :param selected: Sweep outcome denoting the promoted configuration.
    :returns: ``None``.
    """

    path, lines = start_markdown_report(directory, title="GPT-4o Hyper-parameter Sweep")
    if not outcomes:
        lines.append("No sweep runs were executed.")
        lines.append("")
        write_markdown_lines(path, lines)
        return
    lines.append(
        "The table below captures validation accuracy on eligible slates plus "
        "formatting/parse rates for each temperature/top-p/max-token configuration. "
        "The selected configuration is marked with ✓."
    )
    lines.extend(_build_sweep_table(outcomes, selected))
    write_markdown_lines(path, lines)


def _build_sweep_table(
    outcomes: Sequence[SweepOutcome], selected: SweepOutcome
) -> List[str]:
    """
    Return the sweep table rendered as Markdown lines.

    :param outcomes: Sequence of sweep outcomes collected during evaluation.
    :param selected: Sweep outcome denoting the promoted configuration.
    :returns: Markdown lines capturing the sweep leaderboard.
    """

    lines: List[str] = [""]
    header_cells = [
        "Config",
        "Temperature",
        "Top-p",
        "Max tokens",
        "Accuracy ↑",
        "Parsed ↑",
        "Formatted ↑",
        "Selected",
    ]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for outcome in outcomes:
        mark = "✓" if outcome.config == selected.config else ""
        lines.append(
            f"| `{outcome.config.label()}` | {outcome.config.temperature:.2f} | "
            f"{outcome.config.top_p:.2f} | {outcome.config.max_tokens} | "
            f"{_format_rate(outcome.accuracy)} | "
            f"{_format_rate(outcome.parsed_rate)} | "
            f"{_format_rate(outcome.format_rate)} | {mark} |"
        )
    lines.append("")
    return lines


def _format_filter_summary(filters: object) -> str | None:
    """
    Return a Markdown bullet describing applied filters, if any.

    :param filters: Raw filters payload returned in the metrics mapping.
    :returns: Markdown bullet summarising filters or ``None``.
    """

    if not isinstance(filters, Mapping):
        return None
    entries: List[str] = []
    issues = filters.get("issues", [])
    studies = filters.get("studies", [])
    if isinstance(issues, Sequence):
        issue_filter = ", ".join(str(item) for item in issues if item)
        if issue_filter:
            entries.append(f"issues: {issue_filter}")
    if isinstance(studies, Sequence):
        study_filter = ", ".join(str(item) for item in studies if item)
        if study_filter:
            entries.append(f"studies: {study_filter}")
    if not entries:
        return None
    return "- **Filters:** " + ", ".join(entries)


def _write_next_video_report(
    directory: Path,
    selected: SweepOutcome,
    metrics: Mapping[str, object],
) -> None:
    """
    Write the next-video evaluation report for the selected configuration.

    :param directory: Destination directory for the report.
    :param selected: Sweep outcome promoted to the final evaluation stage.
    :param metrics: Mapping containing final evaluation metrics.
    :returns: ``None``.
    """

    path, lines = start_markdown_report(directory, title="GPT-4o Next-Video Baseline")
    summary_lines = [
        (
            f"- **Selected configuration:** `{selected.config.label()}` "
            f"(temperature={selected.config.temperature:.2f}, "
            f"top_p={selected.config.top_p:.2f}, max_tokens={selected.config.max_tokens})"
        ),
        (
            f"- **Accuracy:** {_format_rate(float(metrics.get('accuracy_overall', 0.0)))} "
            f"on {int(metrics.get('n_eligible', 0))} eligible slates "
            f"out of {int(metrics.get('n_total', 0))} processed."
        ),
        (
            f"- **Parsed rate:** {_format_rate(float(metrics.get('parsed_rate', 0.0)))}  "
            f"**Formatted rate:** {_format_rate(float(metrics.get('format_rate', 0.0)))}"
        ),
    ]
    filter_line = _format_filter_summary(metrics.get("filters"))
    if filter_line:
        summary_lines.append(filter_line)
    lines.extend(summary_lines)
    lines.append("")
    group_metrics = metrics.get("group_metrics", {})

    def _render_group_table(title: str, payload: Mapping[str, Mapping[str, object]]) -> None:
        """
        Append a Markdown table capturing group-level metrics.

        :param title: Section title that describes the grouping dimension.
        :param payload: Mapping from group identifiers to metrics dictionaries.
        :returns: ``None``.
        """

        if not payload:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for group, stats in payload.items():
            seen = int(stats.get("n_seen", 0))
            eligible = int(stats.get("n_eligible", 0))
            accuracy = _format_rate(float(stats.get("accuracy", 0.0)))
            parsed_rate = _format_rate(float(stats.get("parsed_rate", 0.0)))
            format_rate = _format_rate(float(stats.get("format_rate", 0.0)))
            group_name = group or "unspecified"
            line = (
                f"| {group_name} | {seen} | {eligible} | {accuracy} | "
                f"{parsed_rate} | {format_rate} |"
            )
            lines.append(line)
        lines.append("")
        highlight_lines = _group_highlights(payload)
        if highlight_lines:
            lines.append("### Highlights")
            lines.append("")
            lines.extend(highlight_lines)
            lines.append("")

    if isinstance(group_metrics, Mapping):
        by_issue = group_metrics.get("by_issue")
        if isinstance(by_issue, Mapping):
            _render_group_table(
                "Accuracy by Issue",
                cast(Mapping[str, Mapping[str, object]], by_issue),
            )
        by_study = group_metrics.get("by_participant_study")
        if isinstance(by_study, Mapping):
            _render_group_table(
                "Accuracy by Participant Study",
                cast(Mapping[str, Mapping[str, object]], by_study),
            )

    notes = metrics.get("notes")
    if isinstance(notes, str) and notes.strip():
        lines.append("### Notes")
        lines.append("")
        lines.append(notes.strip())
        lines.append("")

    write_markdown_lines(path, lines)


def _fmt_opinion_value(value: object, digits: int = 3) -> str:
    """
    Format opinion metrics with a consistent fallback.

    :param value: Raw metric value drawn from evaluation outputs.
    :param digits: Number of decimal places to display.
    :returns: String representation or ``"n/a"`` when conversion fails.
    """

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _opinion_summary_lines(
    selected: SweepOutcome, opinion: OpinionEvaluationResult
) -> List[str]:
    """
    Build headline opinion metrics for the report introduction.

    :param selected: Sweep outcome representing the promoted configuration.
    :param opinion: Aggregated opinion evaluation results.
    :returns: Markdown lines describing summary metrics.
    """

    lines = [
        (
            f"- **Selected configuration:** `{selected.config.label()}` "
            f"(temperature={selected.config.temperature:.2f}, "
            f"top_p={selected.config.top_p:.2f}, max_tokens="
            f"{selected.config.max_tokens})"
        )
    ]
    total_participants = sum(result.participants for result in opinion.studies.values())
    lines.append(f"- **Participants evaluated:** {total_participants}")
    combined = opinion.combined_metrics or {}
    if combined:
        lines.append(
            "- **Overall metrics:** "
            f"MAE={_fmt_opinion_value(combined.get('mae_after'))}, "
            f"RMSE={_fmt_opinion_value(combined.get('rmse_after'))}, "
            f"Direction accuracy={_fmt_opinion_value(combined.get('direction_accuracy'))}"
        )
    lines.append("")
    return lines


def _build_opinion_table(
    opinion: OpinionEvaluationResult,
) -> Tuple[List[str], List[Dict[str, object]]]:
    """
    Build the Markdown opinion table and CSV export rows.

    :param opinion: Aggregated opinion evaluation results.
    :returns: Tuple containing table lines and CSV row dictionaries.
    """

    table_lines: List[str] = []
    csv_rows: List[Dict[str, object]] = []
    header = [
        "Study",
        "Issue",
        "Participants",
        "Eligible",
        "MAE (after)",
        "RMSE (after)",
        "Direction ↑",
        "No-change ↑",
        "Δ Accuracy",
    ]
    table_lines.append("| " + " | ".join(header) + " |")
    table_lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for result in opinion.studies.values():
        metrics = result.metrics
        baseline = result.baseline
        direction_accuracy = metrics.get("direction_accuracy")
        baseline_direction = baseline.get("direction_accuracy")
        delta = (
            float(direction_accuracy) - float(baseline_direction)
            if isinstance(direction_accuracy, (int, float))
            and isinstance(baseline_direction, (int, float))
            else None
        )

        table_lines.append(
            "| "
            + " | ".join(
                [
                    result.study_label,
                    result.issue.replace("_", " "),
                    str(result.participants),
                    str(result.eligible),
                    _fmt_opinion_value(metrics.get("mae_after")),
                    _fmt_opinion_value(metrics.get("rmse_after")),
                    _fmt_opinion_value(direction_accuracy),
                    _fmt_opinion_value(baseline_direction),
                    _fmt_opinion_value(delta),
                ]
            )
            + " |"
        )

        csv_rows.append(
            {
                "study": result.study_key,
                "issue": result.issue,
                "participants": result.participants,
                "eligible": result.eligible,
                "mae_after": metrics.get("mae_after"),
                "rmse_after": metrics.get("rmse_after"),
                "direction_accuracy": direction_accuracy,
                "baseline_direction_accuracy": baseline_direction,
                "mae_change": metrics.get("mae_change"),
                "rmse_change": metrics.get("rmse_change"),
            }
        )

    table_lines.append("")
    return table_lines, csv_rows


def _write_opinion_csv(directory: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """
    Write the opinion metrics CSV and return its path.

    :param directory: Directory where the CSV should be stored.
    :param rows: Sequence of opinion metric rows to serialise.
    :returns: Path to the written CSV file, even when no rows were saved.
    """
    csv_path = directory / "opinion_metrics.csv"
    if not rows:
        if csv_path.exists():
            csv_path.unlink()
        return csv_path
    with csv_path.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=[
            "study",
            "issue",
            "participants",
            "eligible",
            "mae_after",
            "rmse_after",
            "direction_accuracy",
            "baseline_direction_accuracy",
            "mae_change",
            "rmse_change",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _artifact_lines(opinion: OpinionEvaluationResult, repo_root: Path) -> List[str]:
    """
    Return Markdown bullets linking to opinion artefacts.

    :param opinion: Opinion evaluation payload containing artefact paths.
    :param repo_root: Repository root for relative path rendering.
    :returns: Markdown bullet lines referencing metrics, predictions, and QA logs.
    """

    lines: List[str] = []
    for result in opinion.studies.values():
        metrics_rel = _relative_path(repo_root, result.metrics_path)
        predictions_rel = _relative_path(repo_root, result.predictions_path)
        qa_log_rel = _relative_path(repo_root, result.qa_log_path)
        lines.append(
            f"- `{result.study_key}` metrics: `{metrics_rel}` "
            f"(predictions: `{predictions_rel}`, QA log: `{qa_log_rel}`)"
        )
    lines.append("")
    return lines


def _write_opinion_report(
    directory: Path,
    *,
    selected: SweepOutcome,
    opinion: OpinionEvaluationResult | None,
    context: ReportContext,
) -> None:
    """
    Create the opinion regression summary document.

    :param directory: Destination directory for the opinion report.
    :param selected: Winning sweep configuration.
    :param opinion: Opinion evaluation payload (may be ``None``).
    :param context: Report context containing repository metadata.
    :returns: ``None``.
    """

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "README.md"
    lines: List[str] = ["# GPT-4o Opinion Shift", ""]

    if opinion is None or not opinion.studies:
        lines.append("No opinion evaluations were produced during this pipeline invocation.")
        lines.append("")
        write_markdown_lines(path, lines)
        return

    lines.extend(_opinion_summary_lines(selected, opinion))

    table_lines, csv_rows = _build_opinion_table(opinion)
    lines.extend(table_lines)
    lines.append("`opinion_metrics.csv` summarises per-study metrics.")
    lines.append("")

    _write_opinion_csv(directory, csv_rows)

    lines.append("### Artefacts")
    lines.append("")
    lines.extend(_artifact_lines(opinion, context.repo_root))

    write_markdown_lines(path, lines)


def generate_reports(
    *,
    context: ReportContext,
    outcomes: Sequence[SweepOutcome],
    selected: SweepOutcome,
    final_metrics: Mapping[str, object],
    opinion_result: OpinionEvaluationResult | None,
) -> None:
    """
    Regenerate catalog, sweep, opinion, and next-video reports.

    :param context: Report context detailing output locations.
    :param outcomes: Sequence of sweep outcomes from the latest run.
    :param selected: Outcome chosen as the final configuration.
    :param final_metrics: Metrics payload from the promoted evaluation.
    :param opinion_result: Opinion evaluation payload (may be ``None``).
    :returns: ``None``.
    """

    _write_catalog_report(context.reports_dir)
    _write_sweep_report(context.reports_dir / "hyperparameter_tuning", outcomes, selected)
    _write_next_video_report(context.reports_dir / "next_video", selected, final_metrics)
    _write_opinion_report(
        context.reports_dir / "opinion",
        selected=selected,
        opinion=opinion_result,
        context=context,
    )


def _relative_path(base: Path, target: Path) -> Path:
    """
    Render ``target`` relative to ``base`` when possible.

    :param base: Base directory representing the repository root.
    :param target: Destination path to normalise.
    :returns: ``target`` relative to ``base`` when both share a prefix; otherwise
        ``target`` unchanged.
    """

    try:
        return target.relative_to(base)
    except ValueError:
        return target


__all__ = [
    "ReportContext",
    "trigger_report_generation",
    "run_report_generation",
    "generate_reports",
]
