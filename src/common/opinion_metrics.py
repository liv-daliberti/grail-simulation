"""Shared helpers for opinion-regression metrics across model families."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

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


def _pick(summary: Any, *names: Optional[str]) -> Optional[float]:
    """Return the first present attribute on ``summary`` from ``names``."""

    for name in names:
        if name and hasattr(summary, name):
            return getattr(summary, name)
    return None


def summarise_opinion_metrics(summary: Any, *, prefer_after_fields: bool) -> OpinionMetricsView:
    """Return a metric bundle with consistent field names across model families."""

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
    """Return a 1D numpy array constructed from ``values``."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        return array.reshape(-1)
    return array


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return a mask of entries that are finite across all arrays."""

    mask = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _direction(values: np.ndarray, *, tolerance: float = 1e-6) -> np.ndarray:
    """Map opinion deltas to -1, 0, +1 direction buckets with tolerance."""

    direction = np.zeros_like(values)
    direction[values > tolerance] = 1.0
    direction[values < -tolerance] = -1.0
    return direction


def _safe_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Return the coefficient of determination handling degenerate cases."""

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
    """Return slope and intercept fitting actual change as a function of predicted change."""

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
        self._predicted_change = predicted_change
        self._actual_change = actual_change
        self._bins: list[dict] = []
        self._weighted_error = 0.0
        self._total = 0

    def process(self, lower: float, upper: float, include_upper: bool) -> None:
        """Accumulate calibration details for the provided interval."""

        count, error = _accumulate_calibration_bin(
            bins=self._bins,
            predicted_change=self._predicted_change,
            actual_change=self._actual_change,
            lower=lower,
            upper=upper,
            include_upper=include_upper,
        )
        self._weighted_error += error
        self._total += count

    def result(self) -> Tuple[Tuple[dict, ...], float]:
        """Return immutable bins and the expected calibration error."""

        if self._total == 0:
            return tuple(self._bins), float("nan")
        return tuple(self._bins), float(self._weighted_error / self._total)


def _calibration_bins(
    predicted_change: np.ndarray,
    actual_change: np.ndarray,
    *,
    max_bins: int = 10,
) -> Tuple[Tuple[dict, ...], float]:
    """Construct calibration bins and compute an expected calibration error."""

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
            lower=float(lower),
            upper=float(upper),
            include_upper=bool(upper == edges[-1]),
        )

    return accumulator.result()


def _accumulate_calibration_bin(
    *,
    bins: list[dict],
    predicted_change: np.ndarray,
    actual_change: np.ndarray,
    lower: float,
    upper: float,
    include_upper: bool,
) -> Tuple[int, float]:
    """Collect summary statistics for observations within a calibration bin."""

    if include_upper:
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
    """Estimate the KL divergence between actual and predicted change histograms."""

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
    """Return filtered arrays with consistent shapes and finite entries."""

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
    """Compute point estimate errors and opinion-change aggregates."""

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
    """Return direction accuracy metrics when the result is finite."""

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
    """Compute calibration summary statistics and distributional divergence."""

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
