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

"""Shared helpers for opinion-regression metrics across model families."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Mapping

import numpy as np


@dataclass(frozen=True)
class OpinionMetricsView:  # pylint: disable=too-many-instance-attributes
    """Normalised view across KNN and XGB opinion summary payloads."""

    participants: float
    mae: Optional[float]
    baseline_mae: Optional[float]
    mae_delta: Optional[float]
    accuracy: Optional[float]
    baseline_accuracy: Optional[float]
    accuracy_delta: Optional[float]
    rmse_change: Optional[float]
    baseline_rmse_change: Optional[float]
    calibration_ece: Optional[float]
    baseline_calibration_ece: Optional[float]
    kl_divergence_change: Optional[float]
    baseline_kl_divergence_change: Optional[float]


# Canonical CSV columns shared by KNN and XGB opinion writers.
OPINION_CSV_BASE_FIELDS: Tuple[str, ...] = (
    "study",
    "participants",
    "eligible",
    "accuracy_after",
    "baseline_accuracy",
    "accuracy_delta",
    "mae_after",
    "baseline_mae",
    "mae_delta",
    "rmse_after",
    "r2_after",
    "mae_change",
    "rmse_change",
    "rmse_change_delta",
    "calibration_slope",
    "calibration_intercept",
    "calibration_ece",
    "calibration_ece_delta",
    "kl_divergence_change",
    "kl_divergence_delta",
    "dataset",
    "split",
)


def _first_attr(summary: Any, *names: str) -> Optional[float]:
    """
    Return the first present attribute value among ``names`` on ``summary``.

    :param summary: Object exposing attributes via :func:`getattr`.
    :param names: Candidate attribute names checked in order.
    :returns: Attribute value when found, otherwise ``None``.
    """

    for name in names:
        if hasattr(summary, name):
            return getattr(summary, name)
    return None


def build_opinion_csv_base_row(summary: Any, *, study_label: str) -> Mapping[str, object]:
    """
    Produce a CSV row mapping using a normalised view across model families.

    Bridges differences in attribute names between the KNN and XGB
    ``OpinionSummary`` types so both writers can share one implementation.

    :param summary: Opinion summary object produced by a model family.
    :param study_label: Study identifier used to populate the CSV row.
    :returns: Mapping keyed by :data:`OPINION_CSV_BASE_FIELDS`.
    """

    # Normalise fields that differ by model family
    mae_after = _first_attr(summary, "mae", "mae_after")
    rmse_after = _first_attr(summary, "rmse", "rmse_after")
    r2_after = _first_attr(summary, "r2_score", "r2_after")
    accuracy_after = _first_attr(summary, "accuracy", "accuracy_after")

    rmse_change = getattr(summary, "rmse_change", None)
    baseline_rmse_change = getattr(summary, "baseline_rmse_change", None)
    calibration_ece = getattr(summary, "calibration_ece", None)
    baseline_calibration_ece = getattr(summary, "baseline_calibration_ece", None)
    kl_divergence_change = getattr(summary, "kl_divergence_change", None)
    baseline_kl_divergence_change = getattr(
        summary, "baseline_kl_divergence_change", None
    )

    rmse_change_delta = (
        baseline_rmse_change - rmse_change
        if (baseline_rmse_change is not None and rmse_change is not None)
        else None
    )
    calibration_ece_delta = (
        baseline_calibration_ece - calibration_ece
        if (baseline_calibration_ece is not None and calibration_ece is not None)
        else None
    )
    kl_divergence_delta = (
        baseline_kl_divergence_change - kl_divergence_change
        if (
            baseline_kl_divergence_change is not None
            and kl_divergence_change is not None
        )
        else None
    )

    return {
        "study": study_label,
        "participants": getattr(summary, "participants", None),
        "eligible": getattr(summary, "eligible", None),
        "accuracy_after": accuracy_after,
        "baseline_accuracy": getattr(summary, "baseline_accuracy", None),
        "accuracy_delta": getattr(summary, "accuracy_delta", None),
        "mae_after": mae_after,
        "baseline_mae": getattr(summary, "baseline_mae", None),
        "mae_delta": getattr(summary, "mae_delta", None),
        "rmse_after": rmse_after,
        "r2_after": r2_after,
        "mae_change": getattr(summary, "mae_change", None),
        "rmse_change": rmse_change,
        "rmse_change_delta": rmse_change_delta,
        "calibration_slope": getattr(summary, "calibration_slope", None),
        "calibration_intercept": getattr(summary, "calibration_intercept", None),
        "calibration_ece": calibration_ece,
        "calibration_ece_delta": calibration_ece_delta,
        "kl_divergence_change": kl_divergence_change,
        "kl_divergence_delta": kl_divergence_delta,
        "dataset": getattr(summary, "dataset", None),
        "split": getattr(summary, "split", None),
    }


@dataclass(frozen=True)
class CalibrationBinBounds:
    """Describes a calibration bin interval."""

    lower: float
    upper: float
    include_upper: bool


def _pick(summary: Any, *names: Optional[str]) -> Optional[float]:
    """
    Return the first present attribute on ``summary`` from ``names``.

    :param summary: Object exposing attributes via :func:`getattr`.
    :param names: Candidate attribute names; ``None`` entries are ignored.
    :returns: Attribute value when found, otherwise ``None``.
    """

    for name in names:
        if name and hasattr(summary, name):
            return getattr(summary, name)
    return None


def summarise_opinion_metrics(summary: Any, *, prefer_after_fields: bool) -> OpinionMetricsView:
    """
    Return a metric bundle with consistent field names across model families.

    :param summary: Opinion summary object produced by a baseline.
    :param prefer_after_fields: When ``True`` prefer post-study attribute names.
    :returns: :class:`OpinionMetricsView` containing normalised metrics.
    """

    participants_raw = getattr(summary, "participants", 0.0)
    participants = float(participants_raw or 0.0)
    if prefer_after_fields:
        mae = _pick(summary, "mae_after", "mae")
        accuracy = _pick(summary, "accuracy_after", "accuracy")
    else:
        mae = _pick(summary, "mae", "mae_after")
        accuracy = _pick(summary, "accuracy", "accuracy_after")

    return OpinionMetricsView(
        participants=participants,
        mae=mae,
        baseline_mae=_pick(summary, "baseline_mae"),
        mae_delta=_pick(summary, "mae_delta"),
        accuracy=accuracy,
        baseline_accuracy=_pick(summary, "baseline_accuracy"),
        accuracy_delta=_pick(summary, "accuracy_delta"),
        rmse_change=_pick(summary, "rmse_change"),
        baseline_rmse_change=_pick(summary, "baseline_rmse_change"),
        calibration_ece=_pick(summary, "calibration_ece"),
        baseline_calibration_ece=_pick(summary, "baseline_calibration_ece"),
        kl_divergence_change=_pick(summary, "kl_divergence_change"),
        baseline_kl_divergence_change=_pick(summary, "baseline_kl_divergence_change"),
    )


def _safe_numpy(values: Sequence[float]) -> np.ndarray:
    """
    Return a 1D numpy array constructed from ``values``.

    :param values: Sequence of numeric values.
    :returns: One-dimensional :class:`numpy.ndarray` view of ``values``.
    """

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        return array.reshape(-1)
    return array


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """
    Return a mask of entries that are finite across all arrays.

    :param arrays: Sequence of arrays with broadcast-compatible shapes.
    :returns: Boolean mask indicating indices that are finite in every array.
    """

    mask = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _direction(values: np.ndarray, *, tolerance: float = 1e-6) -> np.ndarray:
    """
    Map opinion deltas to -1, 0, +1 direction buckets with tolerance.

    :param values: Opinion delta array.
    :param tolerance: Threshold below which deltas are treated as zero.
    :returns: Array containing -1, 0, or +1 direction labels.
    """

    direction = np.zeros_like(values)
    direction[values > tolerance] = 1.0
    direction[values < -tolerance] = -1.0
    return direction


def _safe_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    """
    Return the coefficient of determination handling degenerate cases.

    :param truth: Ground-truth target values.
    :param prediction: Predicted target values.
    :returns: R² score or ``nan`` when the computation is ill-defined.
    """

    if truth.size < 2:
        return float("nan")
    variance = np.var(truth)
    if variance <= 0.0 or not math.isfinite(variance):
        return float("nan")
    ss_res = float(np.sum((truth - prediction) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _fit_calibration(
    predicted_change: np.ndarray,
    actual_change: np.ndarray,
) -> Tuple[float, float]:
    """
    Return slope and intercept fitting actual change as a function of predicted change.

    :param predicted_change: Predicted opinion deltas.
    :param actual_change: Ground-truth opinion deltas.
    :returns: Tuple of ``(slope, intercept)``; ``nan`` values indicate failure.
    """

    if predicted_change.size < 2:
        return (float("nan"), float("nan"))
    if np.allclose(predicted_change, predicted_change[0]):
        return (float("nan"), float("nan"))
    design = np.vstack([predicted_change, np.ones_like(predicted_change)]).T
    slope, intercept = np.linalg.lstsq(design, actual_change, rcond=None)[0]
    return (float(slope), float(intercept))


class _CalibrationAccumulator:
    """State helper that collects calibration bin statistics."""

    def __init__(self, predicted_change: np.ndarray, actual_change: np.ndarray) -> None:
        """
        Initialise the accumulator with opinion deltas.

        :param predicted_change: Predicted opinion delta array.
        :param actual_change: Ground-truth opinion delta array.
        """
        self._predicted_change = predicted_change
        self._actual_change = actual_change
        self._bins: list[dict] = []
        self._weighted_error = 0.0
        self._total = 0

    def process(self, *, bounds: CalibrationBinBounds) -> None:
        """
        Accumulate calibration details for the provided interval.

        :param bounds: Bin interval describing the lower/upper range to consider.
        :returns: ``None``.
        """

        count, error = _accumulate_calibration_bin(
            bins=self._bins,
            predicted_change=self._predicted_change,
            actual_change=self._actual_change,
            bounds=bounds,
        )
        self._weighted_error += error
        self._total += count

    def result(self) -> Tuple[Tuple[dict, ...], float]:
        """
        Return immutable bins and the expected calibration error.

        :returns: Tuple containing calibration bins and the expected calibration error.
        """

        if self._total == 0:
            return tuple(self._bins), float("nan")
        return tuple(self._bins), float(self._weighted_error / self._total)


def _calibration_bins(
    predicted_change: np.ndarray,
    actual_change: np.ndarray,
    *,
    max_bins: int = 10,
) -> Tuple[Tuple[dict, ...], float]:
    """
    Construct calibration bins and compute an expected calibration error.

    :param predicted_change: Predicted opinion deltas.
    :param actual_change: Ground-truth opinion deltas.
    :param max_bins: Maximum number of quantile bins to construct.
    :returns: Tuple containing calibration bin dictionaries and expected calibration error.
    """

    if predicted_change.size == 0:
        return tuple(), float("nan")

    unique_preds = np.unique(predicted_change)
    if unique_preds.size <= 1:
        return tuple(), float("nan")

    bin_count = min(max_bins, predicted_change.size)
    # Use quantile bins to avoid empty buckets when the distribution is skewed.
    edges = np.quantile(
        predicted_change,
        np.linspace(0.0, 1.0, bin_count + 1),
    )
    edges = np.unique(edges)
    if edges.size <= 1:
        return tuple(), float("nan")

    accumulator = _CalibrationAccumulator(predicted_change, actual_change)

    for lower, upper in zip(edges[:-1], edges[1:]):
        accumulator.process(
            bounds=CalibrationBinBounds(
                lower=float(lower),
                upper=float(upper),
                include_upper=bool(upper == edges[-1]),
            ),
        )

    return accumulator.result()


def _accumulate_calibration_bin(
    *,
    bins: list[dict],
    predicted_change: np.ndarray,
    actual_change: np.ndarray,
    bounds: CalibrationBinBounds,
) -> Tuple[int, float]:
    """
    Collect summary statistics for observations within a calibration bin.

    :param bins: Mutable list receiving bin summaries.
    :param predicted_change: Predicted opinion deltas.
    :param actual_change: Ground-truth opinion deltas.
    :param bounds: Interval describing the bin boundaries.
    :returns: Tuple of observation count and weighted absolute error.
    """

    lower = bounds.lower
    upper = bounds.upper
    if bounds.include_upper:
        mask = (predicted_change >= lower) & (predicted_change <= upper)
    else:
        mask = (predicted_change >= lower) & (predicted_change < upper)

    count = int(mask.sum())
    if count == 0:
        return 0, 0.0

    pred_mean = float(predicted_change[mask].mean())
    actual_mean = float(actual_change[mask].mean())
    bins.append(
        {
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_pred": pred_mean,
            "mean_actual": actual_mean,
        }
    )
    error = count * abs(pred_mean - actual_mean)
    return count, error


def _kl_divergence(
   actual_change: np.ndarray,
   predicted_change: np.ndarray,
   *,
   bins: int = 20,
) -> float:
    """
    Estimate the KL divergence between actual and predicted change histograms.

    :param actual_change: Ground-truth opinion deltas.
    :param predicted_change: Predicted opinion deltas.
    :param bins: Histogram bin count used for the distribution estimate.
    :returns: KL divergence value or ``nan`` when the estimate is ill-defined.
    """

    if actual_change.size == 0 or predicted_change.size == 0:
        return float("nan")

    combined = np.concatenate([actual_change, predicted_change])
    if np.allclose(combined, combined[0]):
        return 0.0

    hist_range = (float(combined.min()), float(combined.max()))
    bins = max(5, min(bins, combined.size))

    truth_counts, _ = np.histogram(actual_change, bins=bins, range=hist_range)
    pred_counts, _ = np.histogram(predicted_change, bins=bins, range=hist_range)

    if truth_counts.sum() == 0 or pred_counts.sum() == 0:
        return float("nan")

    eps = 1e-9
    smoothed_truth = (truth_counts.astype(np.float64) + eps) / (
        truth_counts.sum() + eps * bins
    )
    smoothed_pred = (pred_counts.astype(np.float64) + eps) / (
        pred_counts.sum() + eps * bins
    )
    divergence = float(np.sum(smoothed_truth * np.log(smoothed_truth / smoothed_pred)))
    return divergence


def _prepare_opinion_arrays(
   *,
   truth_after: Sequence[float],
   truth_before: Sequence[float],
   pred_after: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return filtered arrays with consistent shapes and finite entries.

    :param truth_after: Ground-truth post-study opinion scores.
    :param truth_before: Ground-truth pre-study opinion scores.
    :param pred_after: Predicted post-study opinion scores.
    :returns: Tuple of filtered ``(truth_after, truth_before, pred_after)`` arrays.
    :raises ValueError: If the arrays do not share identical shapes.
    """

    truth_after_arr = _safe_numpy(truth_after)
    truth_before_arr = _safe_numpy(truth_before)
    pred_after_arr = _safe_numpy(pred_after)

    if (
        truth_after_arr.shape != truth_before_arr.shape
        or truth_after_arr.shape != pred_after_arr.shape
    ):
        raise ValueError("truth/prediction arrays must share identical shapes.")

    mask = _finite_mask(truth_after_arr, truth_before_arr, pred_after_arr)
    return (
        truth_after_arr[mask],
        truth_before_arr[mask],
        pred_after_arr[mask],
    )


def _error_and_change_metrics(
   *,
   truth_after: np.ndarray,
   truth_before: np.ndarray,
   pred_after: np.ndarray,
) -> Tuple[dict[str, float], np.ndarray, np.ndarray]:
    """
    Compute point estimate errors and opinion-change aggregates.

    :param truth_after: Ground-truth post-study opinion scores.
    :param truth_before: Ground-truth pre-study opinion scores.
    :param pred_after: Predicted post-study opinion scores.
    :returns: Tuple of metrics dictionary, truth deltas, and predicted deltas.
    """

    residual = pred_after - truth_after
    mae_after = float(np.mean(np.abs(residual)))
    rmse_after = float(np.sqrt(np.mean(residual**2)))
    r2_after = float(_safe_r2(truth_after, pred_after))

    change_truth = truth_after - truth_before
    change_pred = pred_after - truth_before
    change_residual = change_pred - change_truth
    mae_change = float(np.mean(np.abs(change_residual)))
    rmse_change = float(np.sqrt(np.mean(change_residual**2)))

    metrics = {
        "mae_after": mae_after,
        "rmse_after": rmse_after,
        "r2_after": r2_after,
        "mae_change": mae_change,
        "rmse_change": rmse_change,
    }
    return metrics, change_truth, change_pred


def _direction_metrics(
   *,
   change_truth: np.ndarray,
   change_pred: np.ndarray,
   tolerance: float,
) -> dict[str, float]:
    """
    Return direction accuracy metrics when the result is finite.

    :param change_truth: Ground-truth opinion deltas.
    :param change_pred: Predicted opinion deltas.
    :param tolerance: Threshold below which deltas are treated as zero.
    :returns: Mapping containing ``direction_accuracy`` when finite.
    """

    direction_truth = _direction(change_truth, tolerance=tolerance)
    direction_pred = _direction(change_pred, tolerance=tolerance)
    accuracy = float(np.mean(direction_truth == direction_pred))
    if math.isfinite(accuracy):
        return {"direction_accuracy": accuracy}
    return {}


def _calibration_and_divergence_metrics(
   *,
   change_truth: np.ndarray,
   change_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Compute calibration summary statistics and distributional divergence.

    :param change_truth: Ground-truth opinion deltas.
    :param change_pred: Predicted opinion deltas.
    :returns: Mapping containing calibration and divergence metrics.
    """

    metrics: dict[str, Any] = {}
    slope, intercept = _fit_calibration(change_pred, change_truth)
    if math.isfinite(slope):
        metrics["calibration_slope"] = slope
    if math.isfinite(intercept):
        metrics["calibration_intercept"] = intercept

    bins, ece = _calibration_bins(change_pred, change_truth)
    if bins:
        metrics["calibration_bins"] = bins
    if math.isfinite(ece):
        metrics["calibration_ece"] = ece

    kl_divergence = _kl_divergence(change_truth, change_pred)
    if math.isfinite(kl_divergence):
        metrics["kl_divergence_change"] = kl_divergence
    return metrics


def compute_opinion_metrics(
    *,
    truth_after: Sequence[float],
    truth_before: Sequence[float],
    pred_after: Sequence[float],
    direction_tolerance: float = 1e-6,
) -> dict:
    """
    Compute opinion regression metrics shared by the KNN and XGBoost pipelines.

    :param truth_after: Ground-truth post-study opinion indices.
    :type truth_after: Sequence[float]
    :param truth_before: Ground-truth pre-study opinion indices.
    :type truth_before: Sequence[float]
    :param pred_after: Predicted post-study opinion indices.
    :type pred_after: Sequence[float]
    :param direction_tolerance: Threshold below which opinion deltas are treated as no-change.
    :type direction_tolerance: float
    :returns: Dictionary containing error metrics, calibration summaries, and sample counts.
    :rtype: dict
    """

    truth_after_arr, truth_before_arr, pred_after_arr = _prepare_opinion_arrays(
        truth_after=truth_after,
        truth_before=truth_before,
        pred_after=pred_after,
    )

    eligible = int(truth_after_arr.size)
    result: dict[str, Any] = {"eligible": eligible}
    if eligible == 0:
        result.update(
            {
                "mae_after": float("nan"),
                "rmse_after": float("nan"),
                "r2_after": float("nan"),
                "mae_change": float("nan"),
                "rmse_change": float("nan"),
            }
        )
        return result

    error_metrics, change_truth, change_pred = _error_and_change_metrics(
        truth_after=truth_after_arr,
        truth_before=truth_before_arr,
        pred_after=pred_after_arr,
    )
    result.update(error_metrics)

    direction_metrics = _direction_metrics(
        change_truth=change_truth,
        change_pred=change_pred,
        tolerance=direction_tolerance,
    )
    result.update(direction_metrics)

    calibration_metrics = _calibration_and_divergence_metrics(
        change_truth=change_truth,
        change_pred=change_pred,
    )
    result.update(calibration_metrics)

    return result


__all__ = ["OpinionMetricsView", "summarise_opinion_metrics", "compute_opinion_metrics"]
