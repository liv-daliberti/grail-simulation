"""Utility helpers shared across the KNN pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Tuple

from common.pipeline_formatters import safe_float, safe_int

from .pipeline_context import MetricSummary, OpinionSummary


def ensure_dir(path: Path) -> Path:
    """Ensure ``path`` exists and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def snake_to_title(value: str) -> str:
    """Convert a snake_case string into Title Case."""

    return value.replace("_", " ").title()


def format_float(value: float) -> str:
    """Format a floating-point metric with three decimal places."""

    return f"{value:.3f}"


def format_optional_float(value: Optional[float]) -> str:
    """Format optional floating-point metrics."""

    return format_float(value) if value is not None else "—"


def format_delta(delta: Optional[float]) -> str:
    """Format a signed improvement metric."""

    return f"{delta:+.3f}" if delta is not None else "—"


def format_count(value: Optional[int]) -> str:
    """Format integer counts with thousands separators."""

    if value is None:
        return "—"
    return f"{value:,}"


def format_k(value: Optional[int]) -> str:
    """Format the selected k hyperparameter."""

    if value is None or value <= 0:
        return "—"
    return str(value)


def format_uncertainty_details(uncertainty: Mapping[str, object]) -> str:
    """Format auxiliary uncertainty metadata for reporting."""

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
    """Return a numeric confidence-interval tuple when available."""

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
    """Collect reusable slate metrics fields."""

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
    """Collect opinion regression metrics into a normalized structure."""

    best_metrics = data.get("best_metrics", {})
    baseline = data.get("baseline", {})
    mae_after = safe_float(best_metrics.get("mae_after"))
    baseline_mae = safe_float(baseline.get("mae_using_before"))
    mae_delta = (
        mae_after - baseline_mae if mae_after is not None and baseline_mae is not None else None
    )
    mae_change = safe_float(best_metrics.get("mae_change"))
    rmse_after = safe_float(best_metrics.get("rmse_after"))
    r2_after = safe_float(best_metrics.get("r2_after"))
    participants = safe_int(data.get("n_participants"))
    dataset = data.get("dataset")
    split = data.get("split")
    best_k = safe_int(data.get("best_k"))

    return OpinionSummary(
        mae=mae_after,
        rmse=rmse_after,
        r2=r2_after,
        mae_change=mae_change,
        baseline_mae=baseline_mae,
        mae_delta=mae_delta,
        best_k=best_k,
        participants=participants,
        dataset=str(dataset) if dataset else None,
        split=str(split) if split else None,
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
