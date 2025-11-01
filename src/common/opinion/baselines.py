#!/usr/bin/env python
"""Baseline metrics shared by GPT-4o/GRPO/KNN/XGB opinion pipelines."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .metrics import compute_opinion_metrics


def baseline_metrics(
    truth_before: Sequence[float], truth_after: Sequence[float]
) -> Dict[str, object]:
    """Compute simple opinion baselines: global-mean-after and no-change.

    Returns MAE/RMSE for predicting the global mean of ``truth_after`` and a
    collection of calibration/direction metrics for the no-change baseline
    (using ``truth_before`` as the prediction).
    """

    after_arr = np.asarray(truth_after, dtype=np.float32)
    before_arr = np.asarray(truth_before, dtype=np.float32)
    if after_arr.size == 0:
        return {}

    baseline_mean = float(np.mean(after_arr))
    baseline_predictions = np.full_like(after_arr, baseline_mean)
    mae_mean = float(np.mean(np.abs(baseline_predictions - after_arr)))
    rmse_mean = float(np.sqrt(np.mean((baseline_predictions - after_arr) ** 2)))

    no_change = compute_opinion_metrics(
        truth_after=after_arr,
        truth_before=before_arr,
        pred_after=before_arr,
    )
    direction_accuracy = no_change.get("direction_accuracy")
    direction_accuracy = (
        float(direction_accuracy) if isinstance(direction_accuracy, (int, float)) else None
    )

    return {
        "global_mean_after": baseline_mean,
        "mae_global_mean_after": mae_mean,
        "rmse_global_mean_after": rmse_mean,
        "mae_using_before": float(no_change.get("mae_after", float("nan"))),
        "rmse_using_before": float(no_change.get("rmse_after", float("nan"))),
        "mae_change_zero": float(no_change.get("mae_change", float("nan"))),
        "rmse_change_zero": float(no_change.get("rmse_change", float("nan"))),
        "calibration_slope_change_zero": no_change.get("calibration_slope"),
        "calibration_intercept_change_zero": no_change.get("calibration_intercept"),
        "calibration_ece_change_zero": no_change.get("calibration_ece"),
        "calibration_bins_change_zero": no_change.get("calibration_bins"),
        "kl_divergence_change_zero": no_change.get("kl_divergence_change"),
        "direction_accuracy": direction_accuracy,
    }
