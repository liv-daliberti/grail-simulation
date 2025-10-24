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

"""Opinion regression report helpers for the XGBoost pipeline."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

from common.pipeline_formatters import (
    format_count as _format_count,
    format_delta as _format_delta,
    format_optional_float as _format_optional_float,
    safe_float as _safe_float,
    safe_int as _safe_int,
)
from common.pipeline_io import write_markdown_lines
from common.report_utils import start_markdown_report

from ..pipeline_context import OpinionSummary
from .plots import _plot_opinion_curve, plt


@dataclass
class _WeightedMetricAccumulator:
    """
    Track weighted aggregates for a metric and its baseline counterpart.

    :ivar value_sum: Accumulated metric multiplied by participant weights.
    :ivar weight_total: Sum of participant weights applied to the metric.
    :ivar baseline_sum: Accumulated baseline metric multiplied by weights.
    :ivar baseline_weight_total: Sum of participant weights applied to the baseline.
    """

    value_sum: float = 0.0
    weight_total: float = 0.0
    baseline_sum: float = 0.0
    baseline_weight_total: float = 0.0

    def add(
        self,
        *,
        value: Optional[float],
        baseline: Optional[float],
        weight: Optional[float],
    ) -> None:
        """
        Incorporate a new measurement into the weighted aggregates.

        :param value: Metric value recorded for the study.
        :param baseline: Baseline value comparable to ``value``.
        :param weight: Participant count or other weighting factor.
        """

        if value is None or weight in (None, 0):
            return
        weight_float = float(weight)
        self.value_sum += value * weight_float
        self.weight_total += weight_float
        if baseline is not None:
            self.baseline_sum += baseline * weight_float
            self.baseline_weight_total += weight_float

    def weighted_value(self) -> Optional[float]:
        """Return the weighted metric value."""

        if self.weight_total <= 0:
            return None
        return self.value_sum / self.weight_total

    def weighted_baseline(self) -> Optional[float]:
        """Return the weighted baseline metric."""

        if self.baseline_weight_total <= 0:
            return None
        return self.baseline_sum / self.baseline_weight_total


@dataclass
class _OpinionPortfolioAccumulator:
    """
    Aggregate opinion-regression metrics across studies.

    :ivar mae_stats: Weighted aggregates for MAE metrics.
    :ivar accuracy_stats: Weighted aggregates for directional accuracy metrics.
    :ivar mae_entries: Recorded MAE values paired with study labels.
    :ivar delta_entries: Recorded MAE delta values paired with study labels.
    :ivar accuracy_entries: Recorded directional accuracy values with labels.
    :ivar accuracy_delta_entries: Recorded directional accuracy deltas with labels.
    """

    mae_stats: _WeightedMetricAccumulator = field(default_factory=_WeightedMetricAccumulator)
    accuracy_stats: _WeightedMetricAccumulator = field(
        default_factory=_WeightedMetricAccumulator
    )
    mae_entries: List[Tuple[float, str]] = field(default_factory=list)
    delta_entries: List[Tuple[float, str]] = field(default_factory=list)
    accuracy_entries: List[Tuple[float, str]] = field(default_factory=list)
    accuracy_delta_entries: List[Tuple[float, str]] = field(default_factory=list)

    def record(self, summary: OpinionSummary, label: str) -> None:
        """
        Track metrics for a single study or selection.

        :param summary: Opinion regression metrics captured for the study.
        :type summary: OpinionSummary
        :param label: Human-readable identifier for the study.
        :type label: str
        """

        participants = float(summary.participants or 0)
        mae_value = summary.mae_after
        baseline_value = summary.baseline_mae
        delta_value = summary.mae_delta
        accuracy_value = summary.accuracy_after
        baseline_accuracy = summary.baseline_accuracy
        accuracy_delta = summary.accuracy_delta

        if mae_value is not None:
            self.mae_entries.append((mae_value, label))
        self.mae_stats.add(value=mae_value, baseline=baseline_value, weight=participants)

        if delta_value is not None:
            self.delta_entries.append((delta_value, label))

        if accuracy_value is not None:
            self.accuracy_entries.append((accuracy_value, label))
        if accuracy_delta is not None:
            self.accuracy_delta_entries.append((accuracy_delta, label))
        self.accuracy_stats.add(
            value=accuracy_value,
            baseline=baseline_accuracy,
            weight=participants,
        )

    def to_lines(self, heading_level: str = "####") -> List[str]:
        """
        Render the aggregated metrics as Markdown bullet points.

        :param heading_level: Markdown heading prefix (e.g. ``"###"``) for the section.
        :type heading_level: str
        :returns: Markdown lines summarising weighted MAE and deltas.
        :rtype: List[str]
        """

        if not self.mae_entries:
            return []

        lines: List[str] = [f"{heading_level} Portfolio Summary", ""]
        participant_total = int(self.mae_stats.weight_total)
        lines.extend(self._portfolio_mae_lines(participant_total))
        lines.extend(self._portfolio_accuracy_lines())
        lines.extend(self._portfolio_mae_extremes())
        lines.extend(self._portfolio_accuracy_extremes())
        lines.extend(self._portfolio_accuracy_delta_lines())
        lines.append("")
        return lines

    def _portfolio_mae_lines(self, participants: int) -> List[str]:
        """Construct weighted MAE summary lines."""

        lines: List[str] = []
        weighted_mae = self.mae_stats.weighted_value()
        weighted_baseline = self.mae_stats.weighted_baseline()
        if weighted_mae is not None:
            lines.append(
                "- Weighted MAE "
                f"{_format_optional_float(weighted_mae)} across "
                f"{_format_count(participants)} participants."
            )
        if weighted_baseline is not None:
            mae_delta = None
            if weighted_mae is not None:
                mae_delta = weighted_baseline - weighted_mae
            delta_text = _format_delta(mae_delta) if mae_delta is not None else "—"
            lines.append(
                "- Weighted baseline MAE "
                f"{_format_optional_float(weighted_baseline)} ({delta_text} vs. final)."
            )
        return lines

    def _portfolio_accuracy_lines(self) -> List[str]:
        """Construct weighted directional-accuracy summary lines."""

        lines: List[str] = []
        weighted_accuracy = self.accuracy_stats.weighted_value()
        weighted_baseline = self.accuracy_stats.weighted_baseline()
        weight_total = int(self.accuracy_stats.weight_total)
        if weighted_accuracy is not None:
            lines.append(
                "- Weighted directional accuracy "
                f"{_format_optional_float(weighted_accuracy)} across "
                f"{_format_count(weight_total)} participants."
            )
        if weighted_baseline is not None:
            accuracy_delta = None
            if weighted_accuracy is not None:
                accuracy_delta = weighted_accuracy - weighted_baseline
            delta_text = _format_delta(accuracy_delta) if accuracy_delta is not None else "—"
            lines.append(
                "- Weighted baseline accuracy "
                f"{_format_optional_float(weighted_baseline)} ({delta_text} vs. final)."
            )
        return lines

    def _portfolio_mae_extremes(self) -> List[str]:
        """Summarise portfolio MAE extremes."""

        lines: List[str] = []
        if self.delta_entries:
            best_delta = max(self.delta_entries, key=lambda item: item[0])
            lines.append(
                "- Largest MAE reduction: "
                f"{best_delta[1]} ({_format_delta(best_delta[0])})."
            )
        if len(self.mae_entries) > 1:
            lowest = min(self.mae_entries, key=lambda item: item[0])
            highest = max(self.mae_entries, key=lambda item: item[0])
            lines.append(
                "- Lowest MAE: "
                f"{lowest[1]} ({_format_optional_float(lowest[0])}); "
                f"Highest MAE: {highest[1]} ({_format_optional_float(highest[0])})."
            )
        return lines

    def _portfolio_accuracy_extremes(self) -> List[str]:
        """Summarise portfolio accuracy extremes."""

        lines: List[str] = []
        if self.accuracy_entries:
            best_accuracy = max(self.accuracy_entries, key=lambda item: item[0])
            lines.append(
                "- Highest directional accuracy: "
                f"{best_accuracy[1]} ({_format_optional_float(best_accuracy[0])})."
            )
        if len(self.accuracy_entries) > 1:
            lowest_accuracy = min(self.accuracy_entries, key=lambda item: item[0])
            lines.append(
                "- Lowest directional accuracy: "
                f"{lowest_accuracy[1]} ({_format_optional_float(lowest_accuracy[0])})."
            )
        return lines

    def _portfolio_accuracy_delta_lines(self) -> List[str]:
        """Summarise directional-accuracy deltas."""

        lines: List[str] = []
        if self.accuracy_delta_entries:
            best_delta = max(self.accuracy_delta_entries, key=lambda item: item[0])
            lines.append(
                "- Largest directional-accuracy gain: "
                f"{best_delta[1]} ({_format_delta(best_delta[0])})."
            )
        return lines


def _extract_opinion_summary(data: Mapping[str, object]) -> OpinionSummary:
    """
    Normalise opinion regression metrics into a reusable summary structure.

    :param data: Raw metrics dictionary emitted by the opinion pipeline.
    :type data: Mapping[str, object]
    :returns: Dataclass containing typed fields for opinion reporting.
    :rtype: OpinionSummary
    """

    metrics_block = data.get("metrics", {})
    baseline = data.get("baseline", {})
    mae_after = _safe_float(metrics_block.get("mae_after"))
    baseline_mae = _safe_float(baseline.get("mae_before") or baseline.get("mae_using_before"))
    mae_delta = None
    if mae_after is not None and baseline_mae is not None:
        mae_delta = baseline_mae - mae_after
    accuracy_after = _safe_float(metrics_block.get("direction_accuracy"))
    baseline_accuracy = _safe_float(baseline.get("direction_accuracy"))
    accuracy_delta = None
    if accuracy_after is not None and baseline_accuracy is not None:
        accuracy_delta = accuracy_after - baseline_accuracy
    eligible = _safe_int(data.get("eligible"))
    if eligible is None:
        eligible = _safe_int(metrics_block.get("eligible"))
    return OpinionSummary(
        mae_after=mae_after,
        rmse_after=_safe_float(metrics_block.get("rmse_after")),
        r2_after=_safe_float(metrics_block.get("r2_after")),
        baseline_mae=baseline_mae,
        mae_delta=mae_delta,
        accuracy_after=accuracy_after,
        baseline_accuracy=baseline_accuracy,
        accuracy_delta=accuracy_delta,
        participants=_safe_int(data.get("n_participants")),
        eligible=eligible,
        dataset=str(data.get("dataset")) if data.get("dataset") else None,
        split=str(data.get("split")) if data.get("split") else None,
        label=str(data.get("label")) if data.get("label") else None,
    )


def _opinion_observations(metrics: Mapping[str, Mapping[str, object]]) -> List[str]:
    """
    Generate bullet-point observations comparing opinion metrics.

    :param metrics: Mapping from study key to opinion metrics dictionaries.
    :type metrics: Mapping[str, Mapping[str, object]]
    :returns: Markdown bullet list of opinion-focused observations.
    :rtype: List[str]
    """

    if not metrics:
        return []
    lines: List[str] = ["## Observations", ""]
    deltas: List[float] = []
    r2_scores: List[float] = []
    for study_key in sorted(metrics.keys()):
        summary = _extract_opinion_summary(metrics[study_key])
        delta_text = _format_delta(summary.mae_delta)
        mae_text = _format_optional_float(summary.mae_after)
        r2_text = _format_optional_float(summary.r2_after)
        lines.append(
            f"- {summary.label or study_key}: MAE {mae_text} "
            f"(Δ vs. baseline {delta_text}), R² {r2_text}."
        )
        if summary.mae_delta is not None:
            deltas.append(summary.mae_delta)
        if summary.r2_after is not None:
            r2_scores.append(summary.r2_after)
    if deltas:
        lines.append(
            f"- Average MAE reduction {_format_delta(sum(deltas) / len(deltas))} across "
            f"{len(deltas)} studies."
        )
    if r2_scores:
        lines.append(
            f"- Mean R² {_format_optional_float(sum(r2_scores) / len(r2_scores))}."
        )
    lines.append("")
    return lines


def _metric_distribution_line(values: List[float], label: str) -> Optional[str]:
    """
    Produce a formatted distribution line for a numeric metric.

    :param values: Numeric measurements collected across studies.
    :param label: Prefix to describe the metric being summarised.
    :returns: Markdown bullet line or ``None`` when ``values`` is empty.
    """

    if not values:
        return None
    mean_value = sum(values) / len(values)
    stdev_value = statistics.pstdev(values) if len(values) > 1 else 0.0
    min_value = min(values)
    max_value = max(values)
    return (
        f"{label} {_format_optional_float(mean_value)} "
        f"(σ {_format_optional_float(stdev_value)}, range "
        f"{_format_optional_float(min_value)} – {_format_optional_float(max_value)})."
    )


def _opinion_cross_study_diagnostics(
    metrics: Mapping[str, Mapping[str, object]],
) -> List[str]:
    """
    Summarise cross-study statistics for opinion metrics.

    :param metrics: Mapping from study key to metrics dictionaries.
    :type metrics: Mapping[str, Mapping[str, object]]
    :returns: Markdown lines containing cross-study observations.
    :rtype: List[str]
    """

    if not metrics:
        return []
    portfolio = _OpinionPortfolioAccumulator()
    summaries: List[OpinionSummary] = []
    for study_key, payload in metrics.items():
        summary = _extract_opinion_summary(payload)
        portfolio.record(summary, summary.label or study_key)
        summaries.append(summary)

    lines: List[str] = ["## Cross-Study Diagnostics", ""]
    lines.extend(portfolio.to_lines(heading_level="### Weighted Summary"))

    stat_sources: List[Tuple[List[float], str]] = [
        (
            [item.mae_after for item in summaries if item.mae_after is not None],
            "- Unweighted MAE",
        ),
        (
            [item.mae_delta for item in summaries if item.mae_delta is not None],
            "- MAE delta mean",
        ),
        (
            [item.accuracy_after for item in summaries if item.accuracy_after is not None],
            "- Directional accuracy mean",
        ),
        (
            [item.accuracy_delta for item in summaries if item.accuracy_delta is not None],
            "- Accuracy delta mean",
        ),
    ]
    for values, label in stat_sources:
        distribution_line = _metric_distribution_line(values, label)
        if distribution_line:
            lines.append(distribution_line)
    lines.append("")
    return lines


def _write_opinion_report(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
    *,
    allow_incomplete: bool,
) -> None:
    """
    Create the opinion regression summary document.

    :param directory: Directory where the report and assets are written.
    :type directory: Path
    :param metrics: Mapping from study key to opinion metrics.
    :type metrics: Mapping[str, Mapping[str, object]]
    :param allow_incomplete: Flag controlling placeholder messaging when artefacts are missing.
    :type allow_incomplete: bool
    """

    path, lines = start_markdown_report(directory, title="XGBoost Opinion Regression")
    if not metrics:
        lines.append("No opinion runs were produced during this pipeline invocation.")
        if allow_incomplete:
            lines.append(
                "Rerun the pipeline with `--stage finalize` to populate this section once "
                "opinion metrics are available."
            )
        lines.append("")
        write_markdown_lines(path, lines)
        return
    lines.append(
        "MAE / RMSE / R² / directional accuracy scores for predicting the post-study "
        "opinion index."
    )
    lines.append("")
    dataset_name = "unknown"
    split_name = "validation"
    for payload in metrics.values():
        summary = _extract_opinion_summary(payload)
        if dataset_name == "unknown" and summary.dataset:
            dataset_name = summary.dataset
        if summary.split:
            split_name = summary.split
        if dataset_name != "unknown" and summary.split:
            break
    lines.append(f"- Dataset: `{dataset_name}`")
    lines.append(f"- Split: {split_name}")
    lines.append(
        "- Metrics: MAE, RMSE, R², and directional accuracy compared against a "
        "no-change baseline (pre-study opinion)."
    )
    lines.append("")
    lines.append(
        "| Study | Participants | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | "
        "MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Baseline MAE ↓ |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for study_key, payload in sorted(metrics.items()):
        summary = _extract_opinion_summary(payload)
        study_label = summary.label or study_key
        row_segments = [
            study_label,
            _format_count(summary.participants),
            _format_optional_float(summary.accuracy_after),
            _format_optional_float(summary.baseline_accuracy),
            _format_delta(summary.accuracy_delta),
            _format_optional_float(summary.mae_after),
            _format_delta(summary.mae_delta),
            _format_optional_float(summary.rmse_after),
            _format_optional_float(summary.r2_after),
            _format_optional_float(summary.baseline_mae),
        ]
        lines.append(f"| {' | '.join(row_segments)} |")
    lines.append("")
    curve_lines: List[str] = []
    if plt is not None:
        for study_key, payload in sorted(metrics.items()):
            summary = _extract_opinion_summary(payload)
            rel_path = _plot_opinion_curve(
                directory=directory,
                study_label=summary.label or study_key,
                study_key=study_key,
                payload=payload,
            )
            if rel_path:
                if not curve_lines:
                    curve_lines.extend(["## Training Curves", ""])
                curve_lines.append(f"![{summary.label or study_key}]({rel_path})")
                curve_lines.append("")
    if curve_lines:
        lines.extend(curve_lines)
    lines.extend(_opinion_cross_study_diagnostics(metrics))
    lines.extend(_opinion_observations(metrics))
    write_markdown_lines(path, lines)


__all__ = [
    "_OpinionPortfolioAccumulator",
    "_WeightedMetricAccumulator",
    "_extract_opinion_summary",
    "_opinion_cross_study_diagnostics",
    "_opinion_observations",
    "_write_opinion_report",
]
