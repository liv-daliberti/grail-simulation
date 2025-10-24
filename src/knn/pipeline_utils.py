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

"""Utility helpers shared by the Grail Simulation KNN pipeline stages.

Collects small formatting utilities, filesystem helpers, and conversions
from raw metric payloads into structured summaries used by the reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Tuple

from common.pipeline_formatters import safe_float, safe_int

from .pipeline_context import MetricSummary, OpinionSummary

def ensure_dir(path: Path) -> Path:
    """
    Ensure the given directory exists and return it.

    :param path: Directory path that should be created if missing.
    :type path: Path
    :returns: The original ``Path`` instance for convenient chaining.
    :rtype: Path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

def snake_to_title(value: str) -> str:
    """
    Convert a ``snake_case`` string into Title Case.

    :param value: String containing underscore-separated tokens.
    :type value: str
    :returns: Title-cased string with spaces instead of underscores.
    :rtype: str
    """
    return value.replace("_", " ").title()

def format_float(value: float) -> str:
    """
    Format a floating-point metric with three decimal places.

    :param value: Numeric metric to render.
    :type value: float
    :returns: Formatted string (``{value:.3f}``).
    :rtype: str
    """
    return f"{value:.3f}"

def format_optional_float(value: Optional[float]) -> str:
    """
    Format an optional floating-point metric.

    :param value: Metric value or ``None`` when unavailable.
    :type value: Optional[float]
    :returns: Formatted float or an em dash when ``None``.
    :rtype: str
    """
    return format_float(value) if value is not None else "—"

def format_delta(delta: Optional[float]) -> str:
    """
    Format a signed improvement metric.

    :param delta: Change relative to a baseline.
    :type delta: Optional[float]
    :returns: Signed string (``+0.000`` style) or an em dash when ``None``.
    :rtype: str
    """
    return f"{delta:+.3f}" if delta is not None else "—"

def format_count(value: Optional[int]) -> str:
    """
    Format integer counts with thousands separators.

    :param value: Count to render.
    :type value: Optional[int]
    :returns: Formatted count or an em dash when ``None``.
    :rtype: str
    """
    if value is None:
        return "—"
    return f"{value:,}"

def format_k(value: Optional[int]) -> str:
    """
    Format the selected ``k`` hyper-parameter.

    :param value: Neighbourhood size under consideration.
    :type value: Optional[int]
    :returns: String representation of ``k`` or an em dash when unset.
    :rtype: str
    """
    if value is None or value <= 0:
        return "—"
    return str(value)

def format_uncertainty_details(uncertainty: Mapping[str, object]) -> str:
    """
    Format auxiliary uncertainty metadata for reporting.

    :param uncertainty: Mapping containing extra uncertainty fields.
    :type uncertainty: Mapping[str, object]
    :returns: Parenthesised detail string or an empty string.
    :rtype: str
    """
    if not isinstance(uncertainty, Mapping):
        return ""
    detail_bits: List[str] = []
    for key in ("n_bootstrap", "n_groups", "n_rows", "seed"):
        value = uncertainty.get(key)
        if value is None:
            continue
        detail_bits.append(f"{key}={value}")
    return f" ({', '.join(detail_bits)})" if detail_bits else ""

def parse_ci(ci_value: object) -> Optional[Tuple[float, float]]:
    """
    Convert confidence-interval payloads into numeric tuples.

    :param ci_value: Mapping or sequence describing a confidence interval.
    :type ci_value: object
    :returns: Tuple containing ``(low, high)`` bounds when available.
    :rtype: Optional[Tuple[float, float]]
    """
    if isinstance(ci_value, Mapping):
        low = safe_float(ci_value.get("low"))
        high = safe_float(ci_value.get("high"))
        if low is not None and high is not None:
            return (low, high)
        return None
    if isinstance(ci_value, (tuple, list)) and len(ci_value) == 2:
        low = safe_float(ci_value[0])
        high = safe_float(ci_value[1])
        if low is not None and high is not None:
            return (low, high)
    return None

def extract_metric_summary(data: Mapping[str, object]) -> MetricSummary:
    """
    Collect reusable next-video metric fields from ``data``.

    :param data: Raw metrics dictionary emitted by the evaluation stage.
    :type data: Mapping[str, object]
    :returns: Normalised metric summary for downstream reports.
    :rtype: MetricSummary
    """
    accuracy = safe_float(data.get("accuracy_overall"))
    best_k = safe_int(data.get("best_k"))
    n_total = safe_int(data.get("n_total"))
    n_eligible = safe_int(data.get("n_eligible"))
    accuracy_ci = parse_ci(
        data.get("accuracy_ci_95") or data.get("accuracy_uncertainty", {}).get("ci95")
    )

    baseline_ci = parse_ci(
        data.get("baseline_ci_95") or data.get("baseline_uncertainty", {}).get("ci95")
    )
    baseline_data = data.get("baseline_most_frequent_gold_index", {})
    baseline = None
    if isinstance(baseline_data, Mapping):
        baseline = safe_float(baseline_data.get("accuracy"))

    random_baseline = safe_float(data.get("random_baseline_expected_accuracy"))

    return MetricSummary(
        accuracy=accuracy,
        accuracy_ci=accuracy_ci,
        baseline=baseline,
        baseline_ci=baseline_ci,
        random_baseline=random_baseline,
        best_k=best_k,
        n_total=n_total,
        n_eligible=n_eligible,
    )

def extract_opinion_summary(data: Mapping[str, object]) -> OpinionSummary:
    """
    Collect opinion regression metrics into a normalised structure.

    :param data: Raw metrics dictionary emitted by the opinion pipeline.
    :type data: Mapping[str, object]
    :returns: Normalised opinion summary for reporting.
    :rtype: OpinionSummary
    """
    best_metrics = data.get("best_metrics", {})
    baseline_metrics = data.get("baseline", {})
    mae_after = safe_float(best_metrics.get("mae_after"))
    baseline_mae = safe_float(baseline_metrics.get("mae_using_before"))
    mae_change = safe_float(best_metrics.get("mae_change"))
    rmse_after = safe_float(best_metrics.get("rmse_after"))
    r2_after = safe_float(best_metrics.get("r2_after"))
    participants = safe_int(data.get("n_participants"))
    best_k = safe_int(data.get("best_k"))
    accuracy = safe_float(best_metrics.get("direction_accuracy"))
    baseline_accuracy = safe_float(baseline_metrics.get("direction_accuracy"))
    accuracy_delta = (
        accuracy - baseline_accuracy
        if accuracy is not None and baseline_accuracy is not None
        else None
    )
    eligible = safe_int(
        best_metrics.get("eligible")
        if isinstance(best_metrics, Mapping)
        else None
    )
    if eligible is None:
        eligible = safe_int(data.get("eligible"))
    if eligible is None:
        eligible = participants

    return OpinionSummary(
        mae=mae_after,
        rmse=rmse_after,
        r2_score=r2_after,
        mae_change=mae_change,
        baseline_mae=baseline_mae,
        mae_delta=(
            mae_after - baseline_mae
            if mae_after is not None and baseline_mae is not None
            else None
        ),
        accuracy=accuracy,
        baseline_accuracy=baseline_accuracy,
        accuracy_delta=accuracy_delta,
        best_k=best_k,
        participants=participants,
        eligible=eligible,
        dataset=str(data.get("dataset")) if data.get("dataset") else None,
        split=str(data.get("split")) if data.get("split") else None,
    )

__all__ = [
    "ensure_dir",
    "extract_metric_summary",
    "extract_opinion_summary",
    "format_count",
    "format_delta",
    "format_float",
    "format_k",
    "format_optional_float",
    "format_uncertainty_details",
    "parse_ci",
    "safe_float",
    "safe_int",
    "snake_to_title",
]
