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

"""Structured inputs and helper accumulators for the next-video report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING

from ...context import StudySpec
from ...utils import format_delta, format_optional_float

if TYPE_CHECKING:
    from ...context import MetricSummary
else:  # pragma: no cover - type hint fallback
    MetricSummary = Any


@dataclass
class PortfolioAggregate:
    """Weighted portfolio metrics for a feature space."""

    accuracy: Optional[float]
    baseline: Optional[float]
    random: Optional[float]
    eligible: int
    studies: int


@dataclass
class _PortfolioAccumulator:
    """Mutable helper used to compute weighted aggregates."""

    accuracy_total: float = 0.0
    accuracy_weight: int = 0
    baseline_total: float = 0.0
    baseline_weight: int = 0
    random_total: float = 0.0
    random_weight: int = 0
    studies_with_metrics: int = 0

    def add(self, summary: MetricSummary) -> None:
        """
        Record weighted contributions from ``summary`` when eligible.

        :param summary: Structured metric summary extracted from a study payload.
        """

        eligible = summary.n_eligible
        if not eligible:
            return

        recorded = False
        for total_attr, weight_attr, field_name in (
            ("accuracy_total", "accuracy_weight", "accuracy"),
            ("baseline_total", "baseline_weight", "baseline"),
            ("random_total", "random_weight", "random_baseline"),
        ):
            value = getattr(summary, field_name)
            if value is None:
                continue
            setattr(self, total_attr, getattr(self, total_attr) + value * eligible)
            setattr(self, weight_attr, getattr(self, weight_attr) + eligible)
            recorded = True

        if recorded:
            self.studies_with_metrics += 1

    def result(self) -> Optional[PortfolioAggregate]:
        """
        Return the weighted aggregates accumulated so far.

        :returns: Aggregated portfolio metrics, or ``None`` when nothing was recorded.
        """

        if not self.studies_with_metrics:
            return None

        def _average(total: float, weight: int) -> Optional[float]:
            return total / weight if weight else None

        weighted_accuracy = _average(self.accuracy_total, self.accuracy_weight)
        weighted_baseline = _average(self.baseline_total, self.baseline_weight)
        weighted_random = _average(self.random_total, self.random_weight)
        eligible_total = max(self.accuracy_weight, self.baseline_weight, self.random_weight)

        return PortfolioAggregate(
            accuracy=weighted_accuracy,
            baseline=weighted_baseline,
            random=weighted_random,
            eligible=eligible_total,
            studies=self.studies_with_metrics,
        )


@dataclass
class _ObservationAccumulator:
    """Running statistics for qualitative observations."""

    delta_sum: float = 0.0
    delta_count: int = 0
    random_sum: float = 0.0
    random_count: int = 0

    def record(self, summary: MetricSummary) -> None:
        """
        Update accumulated statistics with ``summary``.

        :param summary: Structured metric summary extracted from a study payload.
        """

        delta = (
            summary.accuracy - summary.baseline
            if summary.accuracy is not None and summary.baseline is not None
            else None
        )
        if delta is not None:
            self.delta_sum += delta
            self.delta_count += 1

        random_value = summary.random_baseline
        if random_value is not None:
            self.random_sum += random_value
            self.random_count += 1

    def averages(self) -> list[str]:
        """
        Return formatted average statistics when available.

        :returns: List of formatted mean statistics (may be empty).
        """

        extras: list[str] = []
        if self.delta_count:
            extras.append(f"mean Δ {format_delta(self.delta_sum / self.delta_count)}")
        if self.random_count:
            extras.append(
                f"mean random {format_optional_float(self.random_sum / self.random_count)}"
            )
        return extras


@dataclass
class NextVideoReportInputs:
    """
    Input bundle required to render the next-video report.

    :param output_dir: Destination directory where report artifacts are written.
    :param metrics_by_feature: Nested mapping of per-study metrics keyed by feature space.
    :param studies: Ordered study specifications covered by the report.
    :param feature_spaces: Feature spaces in the order they should appear in the report.
    :param loso_metrics: Optional leave-one-study-out metrics grouped by feature space.
    :param allow_incomplete: Whether the report may emit placeholders when data is missing.
    :param xgb_next_video_dir: Root directory containing XGB comparison metrics, if any.
    """

    output_dir: Path
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    studies: Sequence[StudySpec]
    feature_spaces: Sequence[str]
    loso_metrics: Optional[Mapping[str, Mapping[str, Mapping[str, object]]]] = None
    allow_incomplete: bool = False
    xgb_next_video_dir: Optional[Path] = None
