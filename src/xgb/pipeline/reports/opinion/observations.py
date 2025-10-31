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

"""Observation and diagnostic helpers for opinion metrics."""

from __future__ import annotations

import statistics
from typing import List, Mapping, Optional, Tuple

from common.pipeline.formatters import (
    format_delta as _format_delta,
    format_optional_float as _format_optional_float,
)

from .accumulators import _OpinionPortfolioAccumulator
from .metrics import _append_difference, _append_if_not_none
from .summaries import _extract_opinion_summary
from ...context import OpinionSummary


def _opinion_observations(metrics: Mapping[str, Mapping[str, object]]) -> List[str]:
    """
    Generate bullet-point observations comparing opinion metrics.

    :param metrics: Mapping from study identifiers to metrics payloads.
    :returns: Markdown lines summarising notable per-study observations.
    """

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
    """
    Return a formatted distribution line for a numeric metric.

    :param values: Recorded metric values being summarised.
    :param label: Human-readable descriptor for the metric.
    :returns: Markdown string describing the distribution or ``None``.
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

    :param metrics: Mapping from study identifiers to metrics payloads.
    :returns: Markdown lines covering weighted and unweighted diagnostics.
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


__all__ = [
    "_metric_distribution_line",
    "_opinion_cross_study_diagnostics",
    "_opinion_observations",
]
