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
# pylint: disable=too-many-lines

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
import csv
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from common.pipeline.formatters import (
    format_count as _format_count,
    format_delta as _format_delta,
    format_optional_float as _format_optional_float,
    safe_float as _safe_float,
    safe_int as _safe_int,
)
from common.pipeline.io import write_markdown_lines
from common.reports.utils import append_image_section, start_markdown_report

from common.opinion.metrics import (
    OPINION_CSV_BASE_FIELDS,
    build_opinion_csv_base_row,
    summarise_opinion_metrics,
)

from ..context import OpinionSummary
from .plots import (
    _plot_opinion_change_heatmap,
    _plot_opinion_curve,
    _plot_opinion_error_histogram,
    _plot_opinion_post_heatmap,
    plt,
)
from .shared import LOGGER


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
        """Add a weighted measurement to the aggregate."""

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


def _difference(baseline: Optional[float], value: Optional[float]) -> Optional[float]:
    """Return ``baseline - value`` when both inputs are present."""

    if baseline is None or value is None:
        return None
    return baseline - value


def _append_if_not_none(values: List[float], value: Optional[float]) -> None:
    """Append ``value`` to ``values`` when the value is not ``None``."""

    if value is not None:
        values.append(value)


def _append_difference(
    values: List[float],
    baseline: Optional[float],
    value: Optional[float],
) -> None:
    """Append ``baseline - value`` to ``values`` when both numbers exist."""

    delta_value = _difference(baseline, value)
    if delta_value is not None:
        values.append(delta_value)


@dataclass
class _MetricTarget:
    """Destinations for metric aggregation."""

    entries: List[Tuple[float, str]]
    stats: _WeightedMetricAccumulator
    delta_entries: Optional[List[Tuple[float, str]]] = None


@dataclass
class _MetricContext:
    """Metadata associated with a metric value being recorded."""

    label: str
    participants: Optional[float]
    delta_value: Optional[float]


@dataclass
class _OpinionPortfolioAccumulator:  # pylint: disable=too-many-instance-attributes
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
    rmse_change_stats: _WeightedMetricAccumulator = field(
        default_factory=_WeightedMetricAccumulator
    )
    calibration_ece_stats: _WeightedMetricAccumulator = field(
        default_factory=_WeightedMetricAccumulator
    )
    kl_divergence_stats: _WeightedMetricAccumulator = field(
        default_factory=_WeightedMetricAccumulator
    )
    mae_entries: List[Tuple[float, str]] = field(default_factory=list)
    delta_entries: List[Tuple[float, str]] = field(default_factory=list)
    accuracy_entries: List[Tuple[float, str]] = field(default_factory=list)
    accuracy_delta_entries: List[Tuple[float, str]] = field(default_factory=list)
    rmse_change_entries: List[Tuple[float, str]] = field(default_factory=list)
    rmse_change_delta_entries: List[Tuple[float, str]] = field(default_factory=list)
    calibration_ece_entries: List[Tuple[float, str]] = field(default_factory=list)
    calibration_ece_delta_entries: List[Tuple[float, str]] = field(default_factory=list)
    kl_entries: List[Tuple[float, str]] = field(default_factory=list)
    kl_delta_entries: List[Tuple[float, str]] = field(default_factory=list)

    def _record_metric(
        self,
        value: Optional[float],
        baseline: Optional[float],
        target: _MetricTarget,
        context: _MetricContext,
    ) -> None:
        """Record an individual metric value and optional delta."""

        if value is not None:
            target.entries.append((value, context.label))
        if target.delta_entries is not None and context.delta_value is not None:
            target.delta_entries.append((context.delta_value, context.label))
        target.stats.add(
            value=value,
            baseline=baseline,
            weight=context.participants,
        )

    def record(self, summary: OpinionSummary, label: str) -> None:
        """Track metrics for a single study or selection."""

        metrics = summarise_opinion_metrics(summary, prefer_after_fields=True)
        participants = metrics.participants
        self._record_metric(
            metrics.mae,
            metrics.baseline_mae,
            _MetricTarget(
                entries=self.mae_entries,
                stats=self.mae_stats,
                delta_entries=self.delta_entries,
            ),
            _MetricContext(
                label=label,
                participants=participants,
                delta_value=metrics.mae_delta,
            ),
        )
        self._record_metric(
            metrics.accuracy,
            metrics.baseline_accuracy,
            _MetricTarget(
                entries=self.accuracy_entries,
                stats=self.accuracy_stats,
                delta_entries=self.accuracy_delta_entries,
            ),
            _MetricContext(
                label=label,
                participants=participants,
                delta_value=metrics.accuracy_delta,
            ),
        )
        self._record_metric(
            metrics.rmse_change,
            metrics.baseline_rmse_change,
            _MetricTarget(
                entries=self.rmse_change_entries,
                stats=self.rmse_change_stats,
                delta_entries=self.rmse_change_delta_entries,
            ),
            _MetricContext(
                label=label,
                participants=participants,
                delta_value=_difference(
                    metrics.baseline_rmse_change,
                    metrics.rmse_change,
                ),
            ),
        )
        self._record_metric(
            metrics.calibration_ece,
            metrics.baseline_calibration_ece,
            _MetricTarget(
                entries=self.calibration_ece_entries,
                stats=self.calibration_ece_stats,
                delta_entries=self.calibration_ece_delta_entries,
            ),
            _MetricContext(
                label=label,
                participants=participants,
                delta_value=_difference(
                    metrics.baseline_calibration_ece,
                    metrics.calibration_ece,
                ),
            ),
        )
        self._record_metric(
            metrics.kl_divergence_change,
            metrics.baseline_kl_divergence_change,
            _MetricTarget(
                entries=self.kl_entries,
                stats=self.kl_divergence_stats,
                delta_entries=self.kl_delta_entries,
            ),
            _MetricContext(
                label=label,
                participants=participants,
                delta_value=_difference(
                    metrics.baseline_kl_divergence_change,
                    metrics.kl_divergence_change,
                ),
            ),
        )

    def to_lines(self, heading: str = "#### Portfolio Summary") -> List[str]:
        """Render the aggregated metrics as Markdown bullet points."""

        if not self.mae_entries:
            return []

        lines: List[str] = []
        if heading:
            lines.extend([heading, ""])
        participant_total = int(self.mae_stats.weight_total)
        lines.extend(self._portfolio_mae_lines(participant_total))
        lines.extend(self._portfolio_accuracy_lines())
        lines.extend(self._portfolio_rmse_change_lines(participant_total))
        lines.extend(self._portfolio_calibration_lines(participant_total))
        lines.extend(self._portfolio_kl_lines(participant_total))
        lines.extend(self._portfolio_mae_extremes())
        lines.extend(self._portfolio_accuracy_extremes())
        lines.extend(self._portfolio_accuracy_delta_lines())
        lines.extend(self._portfolio_rmse_change_extremes())
        lines.extend(self._portfolio_calibration_extremes())
        lines.extend(self._portfolio_kl_extremes())
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

    def _portfolio_rmse_change_lines(self, participants: int) -> List[str]:
        """Construct weighted RMSE(change) summary lines."""

        lines: List[str] = []
        weighted_rmse = self.rmse_change_stats.weighted_value()
        weighted_baseline = self.rmse_change_stats.weighted_baseline()
        weight_total = int(self.rmse_change_stats.weight_total) or participants
        if weighted_rmse is not None:
            lines.append(
                "- Weighted RMSE (change) "
                f"{_format_optional_float(weighted_rmse)} across "
                f"{_format_count(weight_total)} participants."
            )
        if weighted_baseline is not None:
            rmse_delta = None
            if weighted_rmse is not None:
                rmse_delta = weighted_baseline - weighted_rmse
            delta_text = _format_delta(rmse_delta) if rmse_delta is not None else "—"
            lines.append(
                "- Weighted baseline RMSE (change) "
                f"{_format_optional_float(weighted_baseline)} ({delta_text} vs. final)."
            )
        return lines

    def _portfolio_calibration_lines(self, participants: int) -> List[str]:
        """Construct weighted calibration summary lines."""

        lines: List[str] = []
        weighted_ece = self.calibration_ece_stats.weighted_value()
        weighted_baseline = self.calibration_ece_stats.weighted_baseline()
        weight_total = int(self.calibration_ece_stats.weight_total) or participants
        if weighted_ece is not None:
            lines.append(
                "- Weighted calibration ECE "
                f"{_format_optional_float(weighted_ece)} across "
                f"{_format_count(weight_total)} participants."
            )
        if weighted_baseline is not None:
            ece_delta = None
            if weighted_ece is not None:
                ece_delta = weighted_baseline - weighted_ece
            delta_text = _format_delta(ece_delta) if ece_delta is not None else "—"
            lines.append(
                "- Weighted baseline ECE "
                f"{_format_optional_float(weighted_baseline)} ({delta_text} vs. final)."
            )
        return lines

    def _portfolio_kl_lines(self, participants: int) -> List[str]:
        """Construct weighted KL divergence summary lines."""

        lines: List[str] = []
        weighted_kl = self.kl_divergence_stats.weighted_value()
        weighted_baseline = self.kl_divergence_stats.weighted_baseline()
        weight_total = int(self.kl_divergence_stats.weight_total) or participants
        if weighted_kl is not None:
            lines.append(
                "- Weighted KL divergence "
                f"{_format_optional_float(weighted_kl)} across "
                f"{_format_count(weight_total)} participants."
            )
        if weighted_baseline is not None:
            kl_delta = None
            if weighted_kl is not None:
                kl_delta = weighted_baseline - weighted_kl
            delta_text = _format_delta(kl_delta) if kl_delta is not None else "—"
            lines.append(
                "- Weighted baseline KL divergence "
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

    def _portfolio_rmse_change_extremes(self) -> List[str]:
        """Summarise RMSE(change) extremes across the portfolio."""

        lines: List[str] = []
        if self.rmse_change_delta_entries:
            best_delta = max(self.rmse_change_delta_entries, key=lambda item: item[0])
            lines.append(
                "- Largest RMSE(change) reduction: "
                f"{best_delta[1]} ({_format_delta(best_delta[0])})."
            )
        if len(self.rmse_change_entries) > 1:
            lowest = min(self.rmse_change_entries, key=lambda item: item[0])
            highest = max(self.rmse_change_entries, key=lambda item: item[0])
            lines.append(
                "- Lowest RMSE(change): "
                f"{lowest[1]} ({_format_optional_float(lowest[0])}); "
                f"Highest: {highest[1]} ({_format_optional_float(highest[0])})."
            )
        return lines

    def _portfolio_calibration_extremes(self) -> List[str]:
        """Summarise calibration ECE extremes across the portfolio."""

        lines: List[str] = []
        if self.calibration_ece_delta_entries:
            best_delta = max(self.calibration_ece_delta_entries, key=lambda item: item[0])
            lines.append(
                "- Largest calibration ECE drop: "
                f"{best_delta[1]} ({_format_delta(best_delta[0])})."
            )
        if len(self.calibration_ece_entries) > 1:
            lowest = min(self.calibration_ece_entries, key=lambda item: item[0])
            highest = max(self.calibration_ece_entries, key=lambda item: item[0])
            lines.append(
                "- Lowest calibration ECE: "
                f"{lowest[1]} ({_format_optional_float(lowest[0])}); "
                f"Highest: {highest[1]} ({_format_optional_float(highest[0])})."
            )
        return lines

    def _portfolio_kl_extremes(self) -> List[str]:
        """Summarise KL divergence extremes across the portfolio."""

        lines: List[str] = []
        if self.kl_delta_entries:
            best_delta = max(self.kl_delta_entries, key=lambda item: item[0])
            lines.append(
                "- Largest KL divergence drop: "
                f"{best_delta[1]} ({_format_delta(best_delta[0])})."
            )
        if len(self.kl_entries) > 1:
            lowest = min(self.kl_entries, key=lambda item: item[0])
            highest = max(self.kl_entries, key=lambda item: item[0])
            lines.append(
                "- Lowest KL divergence: "
                f"{lowest[1]} ({_format_optional_float(lowest[0])}); "
                f"Highest: {highest[1]} ({_format_optional_float(highest[0])})."
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
    """Normalise opinion regression metrics into a reusable summary structure."""

    metrics_block = data.get("metrics") or {}
    baseline_raw = data.get("baseline") or {}
    baseline_block = baseline_raw if isinstance(baseline_raw, Mapping) else {}

    def metric_value(name: str) -> Optional[float]:
        """Fetch a metric float from the summary block when available.

        :param name: Metric identifier inside the ``metrics`` payload.
        :returns: Parsed float value or ``None`` if missing/unparseable.
        """
        return _safe_float(metrics_block.get(name))

    def baseline_value(*names: str) -> Optional[float]:
        """Fetch a baseline float from the summary block when available.

        :param names: Candidate baseline keys to attempt in order.
        :returns: Parsed float value or ``None`` if no key is present.
        """
        for candidate in names:
            if candidate in baseline_block:
                return _safe_float(baseline_block.get(candidate))
        return None

    summary_kwargs = {
        "mae_after": metric_value("mae_after"),
        "mae_change": metric_value("mae_change"),
        "rmse_after": metric_value("rmse_after"),
        "r2_after": metric_value("r2_after"),
        "rmse_change": metric_value("rmse_change"),
        "accuracy_after": metric_value("direction_accuracy"),
        "calibration_slope": metric_value("calibration_slope"),
        "calibration_intercept": metric_value("calibration_intercept"),
        "calibration_ece": metric_value("calibration_ece"),
        "kl_divergence_change": metric_value("kl_divergence_change"),
        "participants": _safe_int(data.get("n_participants")),
        "dataset": str(data.get("dataset")) if data.get("dataset") else None,
        "split": str(data.get("split")) if data.get("split") else None,
        "label": str(data.get("label")) if data.get("label") else None,
    }
    summary_kwargs["baseline_mae"] = baseline_value("mae_before", "mae_using_before")
    summary_kwargs["baseline_rmse_change"] = _safe_float(baseline_block.get("rmse_change_zero"))
    summary_kwargs["baseline_accuracy"] = _safe_float(baseline_block.get("direction_accuracy"))
    summary_kwargs["baseline_calibration_slope"] = _safe_float(
        baseline_block.get("calibration_slope_change_zero")
    )
    summary_kwargs["baseline_calibration_intercept"] = _safe_float(
        baseline_block.get("calibration_intercept_change_zero")
    )
    summary_kwargs["baseline_calibration_ece"] = _safe_float(
        baseline_block.get("calibration_ece_change_zero")
    )
    summary_kwargs["baseline_kl_divergence_change"] = _safe_float(
        baseline_block.get("kl_divergence_change_zero")
    )

    summary_kwargs["eligible"] = _safe_int(data.get("eligible"))
    if summary_kwargs["eligible"] is None:
        summary_kwargs["eligible"] = _safe_int(metrics_block.get("eligible"))

    mae_delta = _difference(summary_kwargs["baseline_mae"], summary_kwargs["mae_after"])
    accuracy_delta = None
    if (
        summary_kwargs["accuracy_after"] is not None
        and summary_kwargs["baseline_accuracy"] is not None
    ):
        accuracy_delta = summary_kwargs["accuracy_after"] - summary_kwargs["baseline_accuracy"]

    summary_kwargs["mae_delta"] = mae_delta
    summary_kwargs["accuracy_delta"] = accuracy_delta

    return OpinionSummary(**summary_kwargs)


def _opinion_observations(metrics: Mapping[str, Mapping[str, object]]) -> List[str]:
    """Generate bullet-point observations comparing opinion metrics."""

    if not metrics:
        return []
    lines: List[str] = ["## Observations", ""]
    aggregates = {
        "mae_delta": [],
        "r2": [],
        "rmse_change": [],
        "rmse_delta": [],
        "ece": [],
        "ece_delta": [],
        "kl": [],
        "kl_delta": [],
    }
    for study_key in sorted(metrics.keys()):
        summary = _extract_opinion_summary(metrics[study_key])
        label = summary.label or study_key
        lines.append(
            f"- {label}: MAE {_format_optional_float(summary.mae_after)} "
            f"(Δ vs. baseline {_format_delta(summary.mae_delta)}), "
            f"RMSE(change) {_format_optional_float(summary.rmse_change)}, "
            f"ECE {_format_optional_float(summary.calibration_ece)}, "
            f"KL {_format_optional_float(summary.kl_divergence_change)}, "
            f"R² {_format_optional_float(summary.r2_after)}."
        )
        _append_if_not_none(aggregates["mae_delta"], summary.mae_delta)
        _append_if_not_none(aggregates["r2"], summary.r2_after)
        _append_if_not_none(aggregates["rmse_change"], summary.rmse_change)
        _append_difference(
            aggregates["rmse_delta"],
            summary.baseline_rmse_change,
            summary.rmse_change,
        )
        _append_if_not_none(aggregates["ece"], summary.calibration_ece)
        _append_difference(
            aggregates["ece_delta"],
            summary.baseline_calibration_ece,
            summary.calibration_ece,
        )
        _append_if_not_none(aggregates["kl"], summary.kl_divergence_change)
        _append_difference(
            aggregates["kl_delta"],
            summary.baseline_kl_divergence_change,
            summary.kl_divergence_change,
        )
    summary_configs = [
        (
            "mae_delta",
            _format_delta,
            "- Average MAE reduction {value} across {count} studies.",
        ),
        ("r2", _format_optional_float, "- Mean R² {value}."),
        ("rmse_change", _format_optional_float, "- Mean RMSE(change) {value}."),
        ("rmse_delta", _format_optional_float, "- Mean RMSE(change) delta {value}."),
        ("ece", _format_optional_float, "- Mean calibration ECE {value}."),
        ("ece_delta", _format_optional_float, "- Mean calibration ECE delta {value}."),
        ("kl", _format_optional_float, "- Mean KL divergence {value}."),
        ("kl_delta", _format_optional_float, "- Mean KL divergence delta {value}."),
    ]
    for key, formatter, template in summary_configs:
        values = aggregates[key]
        if values:
            mean_value = sum(values) / len(values)
            lines.append(
                template.format(
                    value=formatter(mean_value),
                    count=len(values),
                )
            )
    lines.append("")
    return lines


def _metric_distribution_line(values: List[float], label: str) -> Optional[str]:
    """Return a formatted distribution line for a numeric metric."""

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


def _dataset_and_split(
    metrics: Mapping[str, Mapping[str, object]],
) -> Tuple[str, str]:
    """Return representative dataset and split names for the metrics payload."""

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
    return dataset_name, split_name


def _opinion_table_header() -> List[str]:
    """Return the Markdown header rows for the opinion metrics table."""

    columns = [
        "Study",
        "Participants",
        "Accuracy ↑",
        "Baseline ↑",
        "Δ Accuracy ↑",
        "MAE ↓",
        "Δ vs baseline ↓",
        "RMSE ↓",
        "R² ↑",
        "MAE (change) ↓",
        "RMSE (change) ↓",
        "Δ RMSE (change) ↓",
        "Calib slope",
        "Calib intercept",
        "ECE ↓",
        "Δ ECE ↓",
        "KL div ↓",
        "Δ KL ↓",
        "Baseline MAE ↓",
    ]
    header_line = f"| {' | '.join(columns)} |"
    align_tokens = ["---"] + ["---:"] * (len(columns) - 1)
    align_line = f"| {' | '.join(align_tokens)} |"
    return [header_line, align_line]


def _opinion_table_rows(metrics: Mapping[str, Mapping[str, object]]) -> List[str]:
    """Build Markdown table rows for opinion metrics."""

    rows: List[str] = []
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
            _format_optional_float(summary.mae_change),
            _format_optional_float(summary.rmse_change),
            _format_delta(_difference(summary.baseline_rmse_change, summary.rmse_change)),
            _format_optional_float(summary.calibration_slope),
            _format_optional_float(summary.calibration_intercept),
            _format_optional_float(summary.calibration_ece),
            _format_delta(
                _difference(
                    summary.baseline_calibration_ece,
                    summary.calibration_ece,
                )
            ),
            _format_optional_float(summary.kl_divergence_change),
            _format_delta(
                _difference(
                    summary.baseline_kl_divergence_change,
                    summary.kl_divergence_change,
                )
            ),
            _format_optional_float(summary.baseline_mae),
        ]
        rows.append(f"| {' | '.join(row_segments)} |")
    return rows


def _opinion_curve_lines(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
) -> List[str]:
    """Render training curve images, returning Markdown lines referencing them."""

    if plt is None:
        return []
    curve_lines: List[str] = []
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
    return curve_lines


@dataclass
class _OpinionPredictionVectors:
    """
    Collect opinion prediction sequences extracted from cached inference rows.

    :ivar actual_after: Observed post-study opinion indices.
    :ivar predicted_after: Predicted post-study opinion indices.
    :ivar actual_changes: Observed opinion deltas (post - pre).
    :ivar predicted_changes: Predicted opinion deltas.
    :ivar errors: Absolute prediction errors for the post-study index.
    """

    actual_after: List[float] = field(default_factory=list)
    predicted_after: List[float] = field(default_factory=list)
    actual_changes: List[float] = field(default_factory=list)
    predicted_changes: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)

    def append_sample(
        self,
        *,
        before: float,
        after: float,
        predicted_after: float,
        predicted_change: float,
    ) -> None:
        """
        Record a single participant snapshot.

        :param before: Baseline opinion index prior to treatment.
        :param after: Observed post-study opinion index.
        :param predicted_after: Model-predicted post-study opinion index.
        :param predicted_change: Model-predicted opinion delta.
        """

        self.actual_after.append(after)
        self.predicted_after.append(predicted_after)
        self.errors.append(abs(predicted_after - after))
        self.actual_changes.append(after - before)
        self.predicted_changes.append(predicted_change)

    def has_post_indices(self) -> bool:
        """Return ``True`` when post-study predictions are available."""

        return bool(self.actual_after and self.predicted_after)

    def has_change_series(self) -> bool:
        """Return ``True`` when change deltas are available."""

        return bool(self.actual_changes and self.predicted_changes)

    def has_errors(self) -> bool:
        """Return ``True`` when prediction errors were recorded."""

        return bool(self.errors)

    def has_observations(self) -> bool:
        """Return ``True`` when any accumulated sequences are non-empty."""

        return bool(self.actual_after or self.actual_changes or self.errors)


def _collect_opinion_prediction_vectors(
    *,
    predictions_path: Path,
    feature_space: str,
    study_key: str,
) -> _OpinionPredictionVectors | None:
    """
    Load cached prediction rows and extract series needed for diagnostic plots.

    :param predictions_path: Filesystem path to the cached prediction JSONL file.
    :param feature_space: Feature-space identifier (e.g. ``tfidf``).
    :param study_key: Participant study identifier.
    :returns: Populated :class:`_OpinionPredictionVectors` or ``None`` when empty.
    """

    def _parse_prediction_row(
        payload: Mapping[str, object],
    ) -> tuple[float, float, float, float] | None:
        before = payload.get("before")
        after_value = payload.get("after")
        pred_after = payload.get("prediction")
        if before is None or after_value is None or pred_after is None:
            return None
        try:
            before_f = float(before)
            after_f = float(after_value)
            pred_after_f = float(pred_after)
        except (TypeError, ValueError):
            return None
        pred_change = payload.get("prediction_change")
        if pred_change is None:
            pred_change_f = pred_after_f - before_f
        else:
            try:
                pred_change_f = float(pred_change)
            except (TypeError, ValueError):
                pred_change_f = pred_after_f - before_f
        return (before_f, after_f, pred_after_f, pred_change_f)

    vectors = _OpinionPredictionVectors()
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = line.strip()
            if not record:
                continue
            try:
                payload = json.loads(record)
            except json.JSONDecodeError:
                LOGGER.debug(
                    "[XGB][OPINION] Skipping malformed prediction row for %s/%s.",
                    feature_space,
                    study_key,
                )
                continue
            parsed = _parse_prediction_row(payload)
            if parsed is None:
                continue
            before_f, after_f, pred_after_f, pred_change_f = parsed
            vectors.append_sample(
                before=before_f,
                after=after_f,
                predicted_after=pred_after_f,
                predicted_change=pred_change_f,
            )

    if not vectors.has_observations():
        return None

    return vectors


def _render_opinion_prediction_plots(
    *,
    feature_dir: Path,
    study_key: str,
    vectors: _OpinionPredictionVectors,
) -> None:
    """
    Dispatch plotting helpers for the supplied prediction series.

    :param feature_dir: Directory receiving generated PNG artefacts.
    :param study_key: Study identifier used when naming outputs.
    :param vectors: Prediction sequences extracted from cached outputs.
    :returns: ``None``.
    """

    if vectors.has_post_indices():
        _plot_opinion_post_heatmap(
            actual_after=vectors.actual_after,
            predicted_after=vectors.predicted_after,
            output_path=feature_dir / f"post_heatmap_{study_key}.png",
        )
    if vectors.has_change_series():
        _plot_opinion_change_heatmap(
            actual_changes=vectors.actual_changes,
            predicted_changes=vectors.predicted_changes,
            output_path=feature_dir / f"change_heatmap_{study_key}.png",
        )
    if vectors.has_errors():
        _plot_opinion_error_histogram(
            errors=vectors.errors,
            output_path=feature_dir / f"error_histogram_{study_key}.png",
        )


def _regenerate_opinion_feature_plots(
    *,
    report_dir: Path,
    metrics: Mapping[str, Mapping[str, object]],
    predictions_root: Path | None,
) -> None:
    """
    Rebuild supplementary opinion plots from cached predictions.

    :param report_dir: Output directory for the opinion report bundle.
    :param metrics: Opinion metrics keyed by feature space and study.
    :param predictions_root: Directory containing cached prediction records.
    :returns: ``None``.
    """
    if not metrics or predictions_root is None:
        return
    for study_key, payload in metrics.items():
        feature_space = str(payload.get("feature_space") or "").lower()
        if not feature_space:
            LOGGER.debug(
                "[XGB][OPINION] Missing feature_space for study=%s; skipping plot regeneration.",
                study_key,
            )
            continue
        feature_dir = report_dir.parent / feature_space / "opinion"
        feature_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = (
            predictions_root
            / feature_space
            / study_key
            / f"opinion_xgb_{study_key}_validation_predictions.jsonl"
        )
        if not predictions_path.exists():
            LOGGER.debug(
                "[XGB][OPINION] Predictions missing for %s/%s at %s; skipping plots.",
                feature_space,
                study_key,
                predictions_path,
            )
            continue

        vectors = _collect_opinion_prediction_vectors(
            predictions_path=predictions_path,
            feature_space=feature_space,
            study_key=study_key,
        )
        if vectors is None:
            continue

        _render_opinion_prediction_plots(
            feature_dir=feature_dir,
            study_key=study_key,
            vectors=vectors,
        )


def _opinion_feature_plot_section(directory: Path) -> List[str]:
    """Embed static PNG assets produced outside the primary report."""

    primary_base = directory.parent
    candidate_bases = [primary_base]
    secondary_base = primary_base.parent
    if secondary_base != primary_base:
        candidate_bases.append(secondary_base)
    seen_paths: set[Path] = set()
    sections: List[str] = []
    for feature_space in ("tfidf", "word2vec", "sentence_transformer"):
        images: List[Path] = []
        for base_dir in candidate_bases:
            feature_dir = base_dir / feature_space / "opinion"
            if not feature_dir.exists():
                continue
            images.extend(sorted(feature_dir.glob("*.png")))
        unique_images: List[Path] = []
        for image in images:
            try:
                canonical = image.resolve()
            except FileNotFoundError:
                # Skip files that vanished since discovery.
                continue
            if canonical in seen_paths:
                continue
            seen_paths.add(canonical)
            unique_images.append(image)
        if not unique_images:
            continue
        sections.append(f"### {feature_space.upper()} Opinion Plots")
        sections.append("")
        for image in unique_images:
            append_image_section(
                sections,
                image=image,
                relative_root=directory.parent,
            )
    return sections


def _opinion_cross_study_diagnostics(
    metrics: Mapping[str, Mapping[str, object]],
) -> List[str]:
    """Summarise cross-study statistics for opinion metrics."""

    if not metrics:
        return []
    portfolio = _OpinionPortfolioAccumulator()
    summaries: List[OpinionSummary] = []
    for study_key, payload in metrics.items():
        summary = _extract_opinion_summary(payload)
        portfolio.record(summary, summary.label or study_key)
        summaries.append(summary)

    lines: List[str] = ["## Cross-Study Diagnostics", ""]
    lines.extend(portfolio.to_lines(heading="### Weighted Summary"))

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
        (
            [item.rmse_change for item in summaries if item.rmse_change is not None],
            "- RMSE(change) mean",
        ),
        (
            [
                item.baseline_rmse_change - item.rmse_change
                for item in summaries
                if item.rmse_change is not None and item.baseline_rmse_change is not None
            ],
            "- RMSE(change) delta mean",
        ),
        (
            [item.calibration_ece for item in summaries if item.calibration_ece is not None],
            "- Calibration ECE mean",
        ),
        (
            [
                item.baseline_calibration_ece - item.calibration_ece
                for item in summaries
                if item.calibration_ece is not None and item.baseline_calibration_ece is not None
            ],
            "- Calibration ECE delta mean",
        ),
        (
            [
                item.kl_divergence_change
                for item in summaries
                if item.kl_divergence_change is not None
            ],
            "- KL divergence mean",
        ),
        (
            [
                item.baseline_kl_divergence_change - item.kl_divergence_change
                for item in summaries
                if item.kl_divergence_change is not None
                and item.baseline_kl_divergence_change is not None
            ],
            "- KL divergence delta mean",
        ),
    ]
    for values, label in stat_sources:
        distribution_line = _metric_distribution_line(values, label)
        if distribution_line:
            lines.append(distribution_line)
    lines.append("")
    return lines


@dataclass(frozen=True)
class OpinionReportOptions:
    """
    Configuration bundle for generating opinion regression reports.

    :ivar allow_incomplete: Whether warnings for missing metrics should be emitted.
    :ivar title: Markdown title applied to the report.
    :ivar description_lines: Optional explanatory copy inserted after the title.
    :ivar predictions_root: Directory containing cached prediction artefacts.
    :ivar regenerate_plots: Whether supplementary plots should be rebuilt.
    """

    allow_incomplete: bool
    title: str = "XGBoost Opinion Regression"
    description_lines: Sequence[str] | None = None
    predictions_root: Path | None = None
    regenerate_plots: bool = True


def _write_opinion_report(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
    options: OpinionReportOptions,
) -> None:
    """
    Create the opinion regression summary document.

    :param directory: Destination directory for the Markdown report.
    :param metrics: Nested mapping of opinion metrics indexed by study.
    :param options: Configuration values controlling report generation.
    :returns: ``None``.
    """

    path, lines = start_markdown_report(directory, title=options.title)
    if not metrics:
        lines.append("No opinion runs were produced during this pipeline invocation.")
        if options.allow_incomplete:
            lines.append(
                "Rerun the pipeline with `--stage finalize` to populate this section once "
                "opinion metrics are available."
            )
        lines.append("")
        write_markdown_lines(path, lines)
        return
    if options.regenerate_plots:
        _regenerate_opinion_feature_plots(
            report_dir=directory,
            metrics=metrics,
            predictions_root=options.predictions_root,
        )
    description_lines = options.description_lines
    if description_lines is None:
        description_lines = [
            "This summary captures the opinion-regression baselines trained with XGBoost "
            "for the selected participant studies."
        ]
    if description_lines:
        lines.extend(description_lines)
        if description_lines[-1].strip():
            lines.append("")
    dataset_name, split_name = _dataset_and_split(metrics)
    lines.extend(
        [
            f"- Dataset: `{dataset_name}`",
            f"- Split: {split_name}",
            (
                "- Metrics track MAE, RMSE, R², directional accuracy, MAE(change), "
                "RMSE(change), calibration slope/intercept, calibration ECE, and KL "
                "divergence versus the no-change baseline."
            ),
            "- Δ columns capture improvements relative to that baseline when available.",
            "",
        ]
    )
    lines.extend(_opinion_table_header())
    lines.extend(_opinion_table_rows(metrics))
    lines.append("")
    curve_lines = _opinion_curve_lines(directory, metrics)
    if curve_lines:
        lines.extend(curve_lines)
    elif plt is None:  # pragma: no cover - optional dependency
        lines.extend(
            [
                "## Training Curves",
                "",
                (
                    "Matplotlib is unavailable in this environment, so training curves "
                    "were not rendered."
                ),
                "",
            ]
        )
    lines.extend(_opinion_feature_plot_section(directory))
    lines.extend(_opinion_cross_study_diagnostics(metrics))
    lines.extend(_opinion_observations(metrics))
    write_markdown_lines(path, lines)
    # Emit CSV dump for downstream analysis
    _write_opinion_csv(directory, metrics)
__all__ = [
    "_OpinionPortfolioAccumulator",
    "_WeightedMetricAccumulator",
    "OpinionReportOptions",
    "_extract_opinion_summary",
    "_opinion_cross_study_diagnostics",
    "_opinion_observations",
    "_write_opinion_report",
]


def _write_opinion_csv(directory: Path, metrics: Mapping[str, Mapping[str, object]]) -> None:
    """Write per-study opinion metrics to opinion_metrics.csv."""

    if not metrics:
        return
    out_path = directory / "opinion_metrics.csv"
    fieldnames = list(OPINION_CSV_BASE_FIELDS)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for study_key in sorted(metrics.keys()):
            summary = _extract_opinion_summary(metrics[study_key])
            row = build_opinion_csv_base_row(
                summary, study_label=(summary.label or study_key)
            )
            writer.writerow(row)
