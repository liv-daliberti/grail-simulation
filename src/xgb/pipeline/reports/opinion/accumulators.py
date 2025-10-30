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

"""Aggregators and helpers for summarising opinion metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from common.pipeline.formatters import (
    format_count as _format_count,
    format_delta as _format_delta,
    format_optional_float as _format_optional_float,
)
from common.opinion.metrics import summarise_opinion_metrics

from .metrics import _difference
from ...context import OpinionSummary


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
        """
        Record an individual metric value and optional delta.

        :param value: Final metric value to aggregate.
        :param baseline: Baseline value aligned with ``value`` when available.
        :param target: Storage containers (lists and accumulators) receiving updates.
        :param context: Metadata describing the current study/selection.
        """

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
        """
        Track metrics for a single study or selection.

        :param summary: Opinion summary containing the metrics to aggregate.
        :param label: Human-readable label for the study or selection.
        """

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
        """
        Render the aggregated metrics as Markdown bullet points.

        :param heading: Markdown heading inserted before the portfolio summary.
        :returns: Ordered list of Markdown lines describing the portfolio.
        """

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
        """
        Construct weighted MAE summary lines.

        :param participants: Total participant count across aggregated runs.
        :returns: Markdown bullet lines covering MAE aggregates.
        """

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
        """
        Construct weighted directional-accuracy summary lines.

        :returns: Markdown bullet lines covering accuracy aggregates.
        """

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
        """
        Construct weighted RMSE(change) summary lines.

        :param participants: Total participant count across aggregated runs.
        :returns: Markdown bullet lines covering RMSE(change) aggregates.
        """

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
        """
        Construct weighted calibration summary lines.

        :param participants: Total participant count across aggregated runs.
        :returns: Markdown bullet lines covering calibration aggregates.
        """

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
        """
        Construct weighted KL divergence summary lines.

        :param participants: Total participant count across aggregated runs.
        :returns: Markdown bullet lines covering KL divergence aggregates.
        """

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
        """
        Summarise portfolio MAE extremes.

        :returns: Markdown bullet lines highlighting MAE extremes.
        """

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
        """
        Summarise portfolio accuracy extremes.

        :returns: Markdown bullet lines highlighting accuracy extremes.
        """

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
        """
        Summarise directional-accuracy deltas.

        :returns: Markdown bullet lines covering accuracy delta extremes.
        """

        lines: List[str] = []
        if self.accuracy_delta_entries:
            best_delta = max(self.accuracy_delta_entries, key=lambda item: item[0])
            lines.append(
                "- Largest directional-accuracy gain: "
                f"{best_delta[1]} ({_format_delta(best_delta[0])})."
            )
        return lines

    def _portfolio_rmse_change_extremes(self) -> List[str]:
        """
        Summarise RMSE(change) extremes across the portfolio.

        :returns: Markdown bullet lines highlighting RMSE(change) extremes.
        """

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
        """
        Summarise calibration ECE extremes across the portfolio.

        :returns: Markdown bullet lines highlighting calibration extremes.
        """

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
        """
        Summarise KL divergence extremes across the portfolio.

        :returns: Markdown bullet lines highlighting KL divergence extremes.
        """

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


__all__ = [
    "_MetricContext",
    "_MetricTarget",
    "_OpinionPortfolioAccumulator",
    "_WeightedMetricAccumulator",
]
