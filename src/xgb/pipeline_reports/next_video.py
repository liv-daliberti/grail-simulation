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

"""Next-video report builders for the XGBoost pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from common.pipeline_formatters import (
    format_count as _format_count,
    format_optional_float as _format_optional_float,
    format_ratio as _format_ratio,
    safe_float as _safe_float,
    safe_int as _safe_int,
)
from common.pipeline_io import write_markdown_lines
from common.report_utils import start_markdown_report

from ..pipeline_context import NextVideoMetricSummary, StudySelection
from .plots import _plot_xgb_curve, _plot_xgb_curve_overview
from .shared import plt


@dataclass
class _PortfolioAccumulator:
    """
    Accumulate portfolio-level statistics across studies.

    :ivar total_correct: Sum of correctly ranked slates across studies.
    :ivar total_evaluated: Total number of evaluated slates with accuracy metrics.
    :ivar total_known_hits: Sum of known-candidate hits across all studies.
    :ivar total_known_total: Total known candidates encountered in evaluations.
    :ivar accuracy_entries: Per-study accuracy values paired with study labels.
    :ivar probability_values: Recorded mean probabilities for known candidates.
    """

    total_correct: int = 0
    total_evaluated: int = 0
    total_known_hits: int = 0
    total_known_total: int = 0
    accuracy_entries: List[Tuple[float, str]] = field(default_factory=list)
    probability_values: List[float] = field(default_factory=list)

    def record(self, summary: "NextVideoMetricSummary", label: str) -> None:
        """
        Update aggregates using the supplied study summary.

        :param summary: Metrics describing accuracy, coverage, and probabilities.
        :type summary: NextVideoMetricSummary
        :param label: Human-readable identifier for the study.
        :type label: str
        """

        self._record_accuracy(summary)
        self._record_known(summary)
        self._record_accuracy_entry(summary, label)
        self._record_probability(summary)

    def _record_accuracy(self, summary: "NextVideoMetricSummary") -> None:
        """
        Track counts needed for weighted accuracy calculations.

        :param summary: Metrics describing slate-level accuracy outcomes.
        :type summary: NextVideoMetricSummary
        """

        if summary.correct is None or summary.evaluated is None:
            return
        self.total_correct += summary.correct
        self.total_evaluated += summary.evaluated

    def _record_known(self, summary: "NextVideoMetricSummary") -> None:
        """
        Track counts needed for known-candidate coverage calculations.

        :param summary: Metrics describing known-candidate hits and totals.
        :type summary: NextVideoMetricSummary
        """

        if summary.known_hits is None or summary.known_total is None:
            return
        self.total_known_hits += summary.known_hits
        self.total_known_total += summary.known_total

    def _record_accuracy_entry(self, summary: "NextVideoMetricSummary", label: str) -> None:
        """
        Store accuracy values for per-study best/worst reporting.

        :param summary: Metrics describing slate-level accuracy outcomes.
        :type summary: NextVideoMetricSummary
        :param label: Human-readable identifier for the study.
        :type label: str
        """

        if summary.accuracy is None:
            return
        self.accuracy_entries.append((summary.accuracy, label))

    def _record_probability(self, summary: "NextVideoMetricSummary") -> None:
        """
        Collect mean probabilities for known-candidate analysis.

        :param summary: Metrics describing mean probabilities per study.
        :type summary: NextVideoMetricSummary
        """

        if summary.avg_probability is None:
            return
        self.probability_values.append(summary.avg_probability)

    def to_lines(self) -> List[str]:
        """
        Render the accumulated statistics as Markdown bullet points.

        :returns: Markdown lines capturing portfolio-level observations.
        :rtype: List[str]
        """

        if not self.total_evaluated and not self.accuracy_entries:
            return []
        lines: List[str] = ["## Portfolio Summary", ""]

        weighted_accuracy = self._weighted_accuracy()
        if weighted_accuracy is not None:
            lines.append(
                f"- Weighted accuracy {_format_optional_float(weighted_accuracy)} "
                f"across {_format_count(self.total_evaluated)} evaluated slates."
            )

        weighted_coverage = self._weighted_coverage()
        if weighted_coverage is not None:
            lines.append(
                f"- Weighted known-candidate coverage {_format_optional_float(weighted_coverage)} "
                f"over {_format_count(self.total_known_total)} eligible slates."
            )

        weighted_availability = self._weighted_availability()
        if weighted_availability is not None:
            lines.append(
                f"- Known-candidate availability {_format_optional_float(weighted_availability)} "
                f"relative to all evaluated slates."
            )

        mean_probability = self._mean_probability()
        if mean_probability is not None:
            probability_count = len(self.probability_values)
            study_label = "study" if probability_count == 1 else "studies"
            lines.append(
                f"- Mean predicted probability on known candidates "
                f"{_format_optional_float(mean_probability)} "
                f"(across {probability_count} {study_label} with recorded probabilities)."
            )

        if self.accuracy_entries:
            best_accuracy, best_label = max(self.accuracy_entries, key=lambda item: item[0])
            lines.append(
                f"- Highest study accuracy: {best_label} "
                f"({_format_optional_float(best_accuracy)})."
            )
            if len(self.accuracy_entries) > 1:
                worst_accuracy, worst_label = min(self.accuracy_entries, key=lambda item: item[0])
                lines.append(
                    f"- Lowest study accuracy: {worst_label} "
                    f"({_format_optional_float(worst_accuracy)})."
                )

        lines.append("")
        return lines

    def _weighted_accuracy(self) -> Optional[float]:
        """
        Compute the weighted accuracy over all recorded studies.

        :returns: Weighted accuracy or ``None`` when insufficient data is available.
        :rtype: Optional[float]
        """

        if not self.total_evaluated:
            return None
        return self.total_correct / self.total_evaluated

    def _weighted_coverage(self) -> Optional[float]:
        """
        Compute the weighted known-candidate coverage across studies.

        :returns: Weighted coverage or ``None`` when insufficient data is available.
        :rtype: Optional[float]
        """

        if not self.total_known_total:
            return None
        return self.total_known_hits / self.total_known_total

    def _weighted_availability(self) -> Optional[float]:
        """
        Compute the availability of known candidates relative to evaluated slates.

        :returns: Weighted availability or ``None`` when insufficient data is available.
        :rtype: Optional[float]
        """

        if not self.total_evaluated:
            return None
        return self.total_known_total / self.total_evaluated

    def _mean_probability(self) -> Optional[float]:
        """
        Compute the mean predicted probability for known candidates.

        :returns: Mean probability or ``None`` when no values were recorded.
        :rtype: Optional[float]
        """

        if not self.probability_values:
            return None
        return sum(self.probability_values) / len(self.probability_values)



def _baseline_accuracy(data: Mapping[str, object]) -> Optional[float]:
    """Extract the most-frequent baseline accuracy from ``data`` when present."""

    payload = data.get("baseline_most_frequent_gold_index")
    if isinstance(payload, Mapping):
        return _safe_float(payload.get("accuracy"))
    return None


def _optional_str(value: object) -> Optional[str]:
    """Return the string form of ``value`` when truthy."""

    return str(value) if value else None


def _extract_next_video_summary(data: Mapping[str, object]) -> NextVideoMetricSummary:
    """
    Normalise next-video metrics into a reusable summary structure.

    :param data: Raw metrics dictionary loaded from ``metrics.json``.
    :type data: Mapping[str, object]
    :returns: Dataclass containing typed fields for report generation.
    :rtype: NextVideoMetricSummary
    """

    evaluated = _safe_int(data.get("evaluated"))
    known_total = _safe_int(data.get("known_candidate_total"))
    known_hits = _safe_int(data.get("known_candidate_hits"))
    return NextVideoMetricSummary(
        accuracy=_safe_float(data.get("accuracy")),
        coverage=_safe_float(data.get("coverage")),
        accuracy_eligible=_safe_float(data.get("accuracy_eligible")),
        evaluated=evaluated,
        correct=_safe_int(data.get("correct")),
        correct_eligible=_safe_int(data.get("correct_eligible")),
        known_hits=known_hits,
        known_total=known_total,
        known_availability=(
            known_total / evaluated if known_total is not None and evaluated else None
        ),
        avg_probability=_safe_float(data.get("avg_probability")),
        baseline_most_frequent_accuracy=_baseline_accuracy(data),
        random_baseline_accuracy=_safe_float(
            data.get("random_baseline_expected_accuracy")
        ),
        dataset=_optional_str(data.get("dataset_source") or data.get("dataset")),
        issue=_optional_str(data.get("issue")),
        issue_label=_optional_str(data.get("issue_label")),
        study_label=_optional_str(data.get("study_label") or data.get("study")),
    )


def _next_video_dataset_info(metrics: Mapping[str, Mapping[str, object]]) -> str:
    """
    Determine the dataset identifier referenced by evaluation metrics.

    :param metrics: Mapping from study key to metrics dictionaries.
    :type metrics: Mapping[str, Mapping[str, object]]
    :returns: Dataset identifier or ``"unknown"`` when absent.
    :rtype: str
    """

    for payload in metrics.values():
        summary = _extract_next_video_summary(payload)
        if summary.dataset:
            return summary.dataset
    return "unknown"


def _next_video_observations(metrics: Mapping[str, Mapping[str, object]]) -> List[str]:
    """
    Generate bullet-point observations comparing study metrics.

    :param metrics: Mapping from study key to metrics dictionaries.
    :type metrics: Mapping[str, Mapping[str, object]]
    :returns: Markdown-formatted bullet points summarising key observations.
    :rtype: List[str]
    """

    if not metrics:
        return []
    lines: List[str] = ["## Observations", ""]
    accuracies: List[float] = []
    elig_accuracies: List[float] = []
    coverages: List[float] = []
    availabilities: List[float] = []
    for study_key in sorted(
        metrics.keys(),
        key=lambda key: (metrics[key].get("study_label") or key).lower(),
    ):
        summary = _extract_next_video_summary(metrics[study_key])
        accuracy_text = _format_optional_float(summary.accuracy)
        elig_text = _format_optional_float(summary.accuracy_eligible)
        coverage_text = _format_optional_float(summary.coverage)
        availability_text = _format_optional_float(summary.known_availability)
        lines.append(
            f"- {summary.study_label or study_key}: accuracy {accuracy_text}, "
            f"eligible accuracy {elig_text}, coverage {coverage_text}, "
            f"known availability {availability_text}."
        )
        if summary.accuracy is not None:
            accuracies.append(summary.accuracy)
        if summary.accuracy_eligible is not None:
            elig_accuracies.append(summary.accuracy_eligible)
        if summary.coverage is not None:
            coverages.append(summary.coverage)
        if summary.known_availability is not None:
            availabilities.append(summary.known_availability)
    if accuracies:
        lines.append(
            f"- Average accuracy "
            f"{_format_optional_float(sum(accuracies) / len(accuracies))}."
        )
    if elig_accuracies:
        lines.append(
            f"- Average eligible-only accuracy "
            f"{_format_optional_float(sum(elig_accuracies) / len(elig_accuracies))}."
        )
    if coverages:
        lines.append(
            f"- Known coverage averages "
            f"{_format_optional_float(sum(coverages) / len(coverages))}."
        )
    if availabilities:
        lines.append(
            f"- Known candidate availability averages "
            f"{_format_optional_float(sum(availabilities) / len(availabilities))}."
        )
    lines.append("")
    return lines


def _next_video_portfolio_summary(
    metrics: Mapping[str, Mapping[str, object]],
) -> List[str]:
    """
    Generate weighted portfolio statistics for next-video evaluations.

    :param metrics: Mapping from study key to final evaluation metrics.
    :type metrics: Mapping[str, Mapping[str, object]]
    :returns: Markdown lines highlighting portfolio-level performance.
    :rtype: List[str]
    """

    accumulator = _PortfolioAccumulator()
    for study_key, payload in metrics.items():
        summary = _extract_next_video_summary(payload)
        label = summary.study_label or study_key
        accumulator.record(summary, label)
    return accumulator.to_lines()


def _report_study_label(
    study_key: str,
    selections: Mapping[str, StudySelection],
) -> str:
    """
    Resolve the display label for a study key using the available selections.

    :param study_key: Key identifying the participant study.
    :type study_key: str
    :param selections: Mapping from study keys to selected sweep outcomes.
    :type selections: Mapping[str, StudySelection]
    :returns: Human-readable study label.
    :rtype: str
    """

    selection = selections.get(study_key)
    if selection is not None:
        return selection.study.label
    return study_key


def _report_issue_label(
    study_key: str,
    fallback: str,
    selections: Mapping[str, StudySelection],
) -> str:
    """
    Resolve the issue label for a study row in the report table.

    :param study_key: Key identifying the participant study.
    :type study_key: str
    :param fallback: Fallback issue string from the metrics payload.
    :type fallback: str
    :param selections: Mapping from study keys to selected sweep outcomes.
    :type selections: Mapping[str, StudySelection]
    :returns: Human-readable issue label.
    :rtype: str
    """

    selection = selections.get(study_key)
    if selection is not None:
        return selection.study.issue.replace("_", " ").title()
    if fallback:
        return str(fallback).replace("_", " ").title()
    return ""


def _next_video_header_lines(
    metrics: Mapping[str, Mapping[str, object]],
    allow_incomplete: bool,
) -> Tuple[List[str], bool]:
    """
    Build introductory Markdown lines, handling the empty-metrics case.

    :param metrics: Mapping from study key to final evaluation metrics.
    :type metrics: Mapping[str, Mapping[str, object]]
    :param allow_incomplete: Flag controlling placeholder messaging when artefacts are missing.
    :type allow_incomplete: bool
    :returns: Tuple containing Markdown lines and a flag indicating whether metrics were present.
    :rtype: Tuple[List[str], bool]
    """

    if metrics:
        dataset_name = _next_video_dataset_info(metrics)
        lines = [
            "Slate-ranking accuracy for the selected XGBoost configuration.",
            "",
            f"- Dataset: `{dataset_name}`",
            "- Split: validation",
            "- Metrics: overall accuracy, eligible-only accuracy (gold present in slate), coverage of known candidates, and availability of known neighbors.",
            (
                "- Table columns capture validation accuracy, counts of correct predictions, "
                "known-candidate recall, and probability calibration for the selected slates."
            ),
            (
                "- `Known hits / total` counts successes among slates that contained a known "
                "candidate; `Known availability` is the share of evaluations with any known "
                "candidate present."
            ),
            "- `Avg prob` reports the mean predicted probability assigned to known candidate hits.",
            "",
        ]
        lines.extend(_next_video_portfolio_summary(metrics))
        return lines, True

    lines = [
        "Accuracy on the validation split for the selected slate configuration.",
        "",
        "No finalized evaluation metrics were available when this report was generated.",
    ]
    if allow_incomplete:
        lines.append(
            "Run the pipeline with `--stage finalize` once sufficient artifacts exist "
            "to refresh this table."
        )
    lines.append("")
    return lines, False


def _next_video_table_lines(
    metrics: Mapping[str, Mapping[str, object]],
    selections: Mapping[str, StudySelection],
) -> List[str]:
    """
    Render the metrics table summarising study-level performance.

    :param metrics: Mapping from study key to final evaluation metrics.
    :type metrics: Mapping[str, Mapping[str, object]]
    :param selections: Mapping from study key to selected sweep outcome.
    :type selections: Mapping[str, StudySelection]
    :returns: Markdown lines representing the metrics table.
    :rtype: List[str]
    """

    lines = [
        "| Study | Issue | Accuracy ↑ | Baseline ↑ | Random ↑ | Correct / evaluated | Coverage ↑ | "
        "Known hits / total | Known availability ↑ | Avg prob ↑ |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    ordered_keys = sorted(
        metrics.keys(),
        key=lambda key: _report_study_label(key, selections).lower(),
    )
    for study_key in ordered_keys:
        summary = _extract_next_video_summary(metrics[study_key])
        study_label = summary.study_label or _report_study_label(study_key, selections)
        fallback_issue = summary.issue or study_key
        issue_label = summary.issue_label or _report_issue_label(
            study_key,
            fallback_issue,
            selections,
        )
        resolved_issue = issue_label or _report_issue_label(
            study_key,
            "",
            selections,
        )
        row_cells = [
            study_label,
            resolved_issue,
            _format_optional_float(summary.accuracy),
            _format_optional_float(summary.baseline_most_frequent_accuracy),
            _format_optional_float(summary.random_baseline_accuracy),
            _format_ratio(summary.correct, summary.evaluated),
            _format_optional_float(summary.coverage),
            _format_ratio(summary.known_hits, summary.known_total),
            _format_optional_float(summary.known_availability),
            _format_optional_float(summary.avg_probability),
        ]
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return lines


def _next_video_curve_lines(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
    selections: Mapping[str, StudySelection],
) -> List[str]:
    """
    Render accuracy curve images, preferring overview plots with fallbacks.

    :param directory: Directory where the report and assets are written.
    :type directory: Path
    :param metrics: Mapping from study key to final evaluation metrics.
    :type metrics: Mapping[str, Mapping[str, object]]
    :param selections: Mapping from study key to selected sweep outcome.
    :type selections: Mapping[str, StudySelection]
    :returns: Markdown lines referencing generated plots.
    :rtype: List[str]
    """

    if plt is None:
        return []
    ordered_keys = sorted(
        metrics.keys(),
        key=lambda key: _report_study_label(key, selections).lower(),
    )
    overview_path = _plot_xgb_curve_overview(
        directory=directory,
        entries=[
            (_report_study_label(key, selections), metrics[key])
            for key in ordered_keys
        ],
    )
    if overview_path:
        return [
            "## Accuracy Curves",
            "",
            f"![Slate accuracy overview]({overview_path})",
            "",
        ]

    lines: List[str] = []
    for study_key in ordered_keys:
        label = _report_study_label(study_key, selections)
        rel_path = _plot_xgb_curve(
            directory=directory,
            study_label=label,
            study_key=study_key,
            payload=metrics[study_key],
        )
        if rel_path:
            if not lines:
                lines.extend(["## Accuracy Curves", ""])
            lines.extend([f"![{label}]({rel_path})", ""])
    return lines


def _loso_entries(
    loso_metrics: Mapping[str, Mapping[str, object]],
    selections: Mapping[str, StudySelection],
) -> List[tuple[str, str, str, NextVideoMetricSummary]]:
    ordered_keys = sorted(
        loso_metrics.keys(),
        key=lambda key: _report_study_label(key, selections).lower(),
    )
    entries: List[tuple[str, str, str, NextVideoMetricSummary]] = []
    for study_key in ordered_keys:
        summary = _extract_next_video_summary(loso_metrics[study_key])
        study_label = summary.study_label or _report_study_label(study_key, selections)
        issue_label = summary.issue_label or _report_issue_label(
            study_key,
            summary.issue or "",
            selections,
        )
        entries.append((study_key, study_label, issue_label, summary))
    return entries


def _loso_accuracy_summary(
    entries: Sequence[tuple[str, str, str, NextVideoMetricSummary]]
) -> List[str]:
    lines: List[str] = []
    accuracy_values = [
        (study_label, summary.accuracy)
        for _key, study_label, _issue_label, summary in entries
        if summary.accuracy is not None
    ]
    if accuracy_values:
        best_label, best_value = max(accuracy_values, key=lambda item: item[1])
        lines.append(
            f"- Highest holdout accuracy: {best_label} "
            f"({_format_optional_float(best_value)})."
        )
        if len(accuracy_values) > 1:
            worst_label, worst_value = min(accuracy_values, key=lambda item: item[1])
            lines.append(
                f"- Lowest holdout accuracy: {worst_label} "
                f"({_format_optional_float(worst_value)})."
            )
        mean_value = sum(value for _label, value in accuracy_values) / len(accuracy_values)
        lines.append(f"- Average holdout accuracy {_format_optional_float(mean_value)}.")

    # Eligible-only accuracy summary
    elig_values = [
        (study_label, summary.accuracy_eligible)
        for _key, study_label, _issue_label, summary in entries
        if summary.accuracy_eligible is not None
    ]
    if elig_values:
        best_label_e, best_value_e = max(elig_values, key=lambda item: item[1])
        lines.append(
            f"- Highest holdout eligible-only accuracy: {best_label_e} "
            f"({_format_optional_float(best_value_e)})."
        )
        if len(elig_values) > 1:
            worst_label_e, worst_value_e = min(elig_values, key=lambda item: item[1])
            lines.append(
                f"- Lowest holdout eligible-only accuracy: {worst_label_e} "
                f"({_format_optional_float(worst_value_e)})."
            )
        mean_value_e = sum(value for _label, value in elig_values) / len(elig_values)
        lines.append(
            f"- Average holdout eligible-only accuracy {_format_optional_float(mean_value_e)}."
        )
    lines.append("")
    return lines


def _next_video_loso_section(
    loso_metrics: Mapping[str, Mapping[str, object]],
    selections: Mapping[str, StudySelection],
) -> List[str]:
    """Render leave-one-study-out markdown when metrics are available."""

    if not loso_metrics:
        return []

    entries = _loso_entries(loso_metrics, selections)
    if not entries:
        return []

    lines: List[str] = ["## Cross-Study Holdouts", ""]

    lines.extend(_loso_accuracy_summary(entries))
    lines.append(
        "| Holdout study | Issue | Accuracy ↑ | Acc (eligible) ↑ | Correct / evaluated | Coverage ↑ | "
        "Known hits / total | Known availability ↑ | Avg prob ↑ |"
    )
    lines.append("| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: |")
    for _, study_label, issue_label, summary in entries:
        row = [
            study_label,
            issue_label,
            _format_optional_float(summary.accuracy),
            _format_optional_float(summary.accuracy_eligible),
            _format_ratio(summary.correct, summary.evaluated),
            _format_optional_float(summary.coverage),
            _format_ratio(summary.known_hits, summary.known_total),
            _format_optional_float(summary.known_availability),
            _format_optional_float(summary.avg_probability),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _write_next_video_report(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
    selections: Mapping[str, StudySelection],
    *,
    allow_incomplete: bool,
    loso_metrics: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """
    Create the next-video evaluation summary document.

    :param directory: Directory where the report and assets are written.
    :type directory: Path
    :param metrics: Mapping from study key to final evaluation metrics.
    :type metrics: Mapping[str, Mapping[str, object]]
    :param selections: Mapping from study key to selected sweep outcome.
    :type selections: Mapping[str, StudySelection]
    :param allow_incomplete: Flag controlling placeholder messaging when artefacts are missing.
    :type allow_incomplete: bool
    """

    path, lines = start_markdown_report(directory, title="XGBoost Next-Video Baseline")
    header_lines, has_metrics = _next_video_header_lines(metrics, allow_incomplete)
    lines.extend(header_lines)
    if not has_metrics:
        write_markdown_lines(path, lines)
        return

    lines.extend(_next_video_table_lines(metrics, selections))
    curve_lines = _next_video_curve_lines(directory, metrics, selections)
    if curve_lines:
        lines.extend(curve_lines)
    elif plt is None:  # pragma: no cover - optional dependency
        lines.extend(
            [
                "## Accuracy Curves",
                "",
                (
                    "Matplotlib is unavailable in this environment, so accuracy curves "
                    "were not rendered."
                ),
                "",
            ]
        )
    loso_section = _next_video_loso_section(loso_metrics or {}, selections)
    if loso_section:
        lines.extend(loso_section)
    elif allow_incomplete:
        lines.extend(
            [
                "## Cross-Study Holdouts",
                "",
                "Leave-one-study-out metrics were unavailable when this report was generated.",
                "",
            ]
        )
    lines.extend(_next_video_observations(metrics))

    write_markdown_lines(path, lines)


__all__ = [
    "_extract_next_video_summary",
    "_next_video_dataset_info",
    "_next_video_observations",
    "_next_video_portfolio_summary",
    "_write_next_video_report",
]
