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

"""Prediction and metric utilities for the KNN opinion pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from common.opinion.metrics import compute_opinion_metrics

from .opinion_index import _similarity_from_distances, _transform_documents, _weighted_mean
from .opinion_models import OpinionExample, OpinionIndex


@dataclass(frozen=True)
class _EvalPlan:
    """Precomputed neighbour arrays and k settings for evaluation."""

    neighbour_distances: np.ndarray
    neighbour_indices: np.ndarray
    unique_k: List[int]
    max_k: int


@dataclass(frozen=True)
class _RowEnv:
    """Environment for per-row evaluation and accumulation."""

    index: OpinionIndex
    plan: _EvalPlan
    exclude_self: bool


@dataclass
class _Accumulators:
    """Mutable per-k accumulators updated while iterating rows."""

    per_k_predictions: Dict[int, List[float]]
    per_k_change_predictions: Dict[int, List[float]]


def _plan_evaluation(
    *,
    index: OpinionIndex,
    eval_examples: Sequence[OpinionExample],
    k_values: Sequence[int],
    exclude_self: bool,
) -> _EvalPlan | None:
    """Prepare neighbour arrays and k settings; return ``None`` when invalid."""
    requested_k = [int(k) for k in k_values if int(k) > 0]
    if not requested_k:
        return None
    max_available = len(index.targets.participant_keys) - (1 if exclude_self else 0)
    max_available = max(1, max_available)
    unique_k = sorted({k for k in requested_k if k <= max_available})
    if not unique_k:
        unique_k = [min(max_available, max(requested_k))]
    max_k = max(unique_k)

    documents = [example.document for example in eval_examples]
    matrix_eval = _transform_documents(index=index, documents=documents)
    neighbour_distances, neighbour_indices = index.neighbors.kneighbors(
        matrix_eval, n_neighbors=max_k
    )
    return _EvalPlan(
        neighbour_distances=neighbour_distances,
        neighbour_indices=neighbour_indices,
        unique_k=unique_k,
        max_k=max_k,
    )


def _accumulate_row(
    *,
    env: _RowEnv,
    acc: _Accumulators,
    example: OpinionExample,
    row_idx: int,
) -> Dict[str, Any] | None:
    """Compute predictions for a single row and update accumulators."""
    distances = env.plan.neighbour_distances[row_idx]
    indices = env.plan.neighbour_indices[row_idx]
    record_after, record_change = _row_predictions(
        index=env.index,
        example=example,
        distances=distances,
        indices=indices,
        params=_RowPredParams(
            unique_k=env.plan.unique_k,
            max_k=env.plan.max_k,
            exclude_self=env.exclude_self,
        ),
    )
    if not record_after:
        return None
    for k, pred in record_after.items():
        acc.per_k_predictions[int(k)].append(float(pred))
    for k, delta in record_change.items():
        acc.per_k_change_predictions[int(k)].append(float(delta))
    return {
        "participant_id": example.participant_id,
        "participant_study": example.participant_study,
        "issue": example.issue,
        "session_id": example.session_id,
        "before_index": example.before,
        "after_index": example.after,
        "predictions_by_k": record_after,
        "predicted_change_by_k": record_change,
    }


def predict_post_indices(
    *,
    index: OpinionIndex,
    eval_examples: Sequence[OpinionExample],
    k_values: Sequence[int],
    exclude_self: bool = False,
) -> Dict[str, Any]:
    """
    Return predictions and aggregate metrics for ``eval_examples``.

    :param index: KNN index object or registry being manipulated.
    :type index: OpinionIndex
    :param eval_examples: Iterable of evaluation examples to score with the index.
    :type eval_examples: Sequence[~knn.core.opinion_models.OpinionExample]
    :param k_values: Iterable of ``k`` values to evaluate or report.
    :type k_values: Sequence[int]
    :param exclude_self: Whether to drop the query point when collecting nearest neighbours.
    :type exclude_self: bool
    :returns: Predictions and aggregate metrics keyed by ``k``.
    :rtype: Dict[str, Any]
    """
    if not eval_examples:
        return {
            "rows": [],
            "per_k_predictions": {int(k): [] for k in k_values},
        }

    plan = _plan_evaluation(
        index=index, eval_examples=eval_examples, k_values=k_values, exclude_self=exclude_self
    )
    if plan is None:
        return {
            "rows": [],
            "per_k_predictions": {},
            "per_k_change_predictions": {},
        }

    per_k_predictions: Dict[int, List[float]] = {k: [] for k in plan.unique_k}
    per_k_change_predictions: Dict[int, List[float]] = {k: [] for k in plan.unique_k}
    env = _RowEnv(index=index, plan=plan, exclude_self=exclude_self)
    acc = _Accumulators(
        per_k_predictions=per_k_predictions,
        per_k_change_predictions=per_k_change_predictions,
    )
    rows: List[Dict[str, Any]] = []

    for row_idx, example in enumerate(eval_examples):
        row = _accumulate_row(env=env, acc=acc, example=example, row_idx=row_idx)
        if row is not None:
            rows.append(row)

    return {
        "rows": rows,
        "per_k_predictions": per_k_predictions,
        "per_k_change_predictions": per_k_change_predictions,
    }


def _metric_bundle(
    *,
    truth_after: np.ndarray,
    truth_before: np.ndarray,
    preds_arr: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for the supplied predictions.

    :param truth_after: Ground-truth post-study opinion indices.
    :type truth_after: np.ndarray
    :param truth_before: Ground-truth pre-study opinion indices.
    :type truth_before: np.ndarray
    :param preds_arr: Model predictions for the post-study indices.
    :type preds_arr: np.ndarray
    :returns: Mapping of opinion metrics computed for the provided arrays.
    :rtype: Dict[str, float]
    """

    return compute_opinion_metrics(
        truth_after=truth_after,
        truth_before=truth_before,
        pred_after=preds_arr,
    )


def _row_prediction_values(
    rows: Sequence[Dict[str, Any]],
    k: int,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract per-row prediction/ground-truth triples for a given ``k``.

    :param rows: Sequence of per-participant prediction dictionaries.
    :type rows: Sequence[Dict[str, Any]]
    :param k: ``k`` value whose predictions should be examined.
    :type k: int
    :returns: Tuple containing (actual_after, actual_before, predictions).
    :rtype: Tuple[List[float], List[float], List[float]]
    """

    actual_after: List[float] = []
    actual_before: List[float] = []
    pred_values: List[float] = []
    key_str = str(k)
    for row in rows:
        predictions_by_k = row.get("predictions_by_k") or {}
        pred_val = predictions_by_k.get(k)
        if pred_val is None:
            pred_val = predictions_by_k.get(key_str)
        if pred_val is None:
            continue
        after_raw = row.get("after_index", row.get("after"))
        before_raw = row.get("before_index", row.get("before"))
        if after_raw is None or before_raw is None:
            continue
        after_val = float(after_raw)
        before_val = float(before_raw)
        if not (math.isfinite(after_val) and math.isfinite(before_val)):
            continue
        pred_values.append(float(pred_val))
        actual_after.append(after_val)
        actual_before.append(before_val)
    return actual_after, actual_before, pred_values


def _post_prediction_series(
    rows: Sequence[Dict[str, Any]],
    k: int,
) -> Tuple[List[float], List[float]]:
    """
    Return actual/predicted post values for ``k``.

    :param rows: Sequence of per-participant prediction dictionaries.
    :type rows: Sequence[Dict[str, Any]]
    :param k: ``k`` value whose predictions should be examined.
    :type k: int
    :returns: Pair of lists containing actual and predicted post-study indices.
    :rtype: Tuple[List[float], List[float]]
    """

    actual_after, _, preds = _row_prediction_values(rows, k)
    return actual_after, preds


def _metrics_from_rows(
    predictions: Dict[int, List[float]],
    rows: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, float]]:
    """
    Return metrics computed from row-level prediction payloads.

    :param predictions: Mapping of k values to prediction lists.
    :type predictions: Dict[int, List[float]]
    :param rows: Sequence of per-participant prediction dictionaries.
    :type rows: Sequence[Dict[str, Any]]
    :returns: Mapping of k values to computed metric bundles.
    :rtype: Dict[int, Dict[str, float]]
    """

    metrics: Dict[int, Dict[str, float]] = {}
    for key in sorted({int(candidate) for candidate in predictions.keys()}):
        actual_after, actual_before, pred_values = _row_prediction_values(rows, key)
        if not pred_values:
            continue
        bundle = _metric_bundle(
            truth_after=np.asarray(actual_after, dtype=np.float32),
            truth_before=np.asarray(actual_before, dtype=np.float32),
            preds_arr=np.asarray(pred_values, dtype=np.float32),
        )
        metrics[key] = bundle
    return metrics


def _metrics_from_eval_examples(
    predictions: Dict[int, List[float]],
    eval_examples: Sequence[OpinionExample],
) -> Dict[int, Dict[str, float]]:
    """
    Return metrics computed over the full evaluation dataset.

    :param predictions: Mapping of k values to prediction lists.
    :type predictions: Dict[int, List[float]]
    :param eval_examples: Iterable of evaluation examples to score with the index.
    :type eval_examples: Sequence[~knn.core.opinion_models.OpinionExample]
    :returns: Mapping of k values to computed metric bundles.
    :rtype: Dict[int, Dict[str, float]]
    """

    truth_after = np.asarray([example.after for example in eval_examples], dtype=np.float32)
    truth_before = np.asarray([example.before for example in eval_examples], dtype=np.float32)
    metrics: Dict[int, Dict[str, float]] = {}
    for k, preds in predictions.items():
        preds_arr = np.asarray(preds, dtype=np.float32)
        if preds_arr.size == 0 or preds_arr.size != truth_after.size:
            continue
        metrics[int(k)] = _metric_bundle(
            truth_after=truth_after,
            truth_before=truth_before,
            preds_arr=preds_arr,
        )
    return metrics


def _summary_metrics(
    *,
    predictions: Dict[int, List[float]],
    eval_examples: Sequence[OpinionExample],
    rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Return MAE, RMSE, and R^2 metrics for each ``k``.

    :param predictions: Sequence of KNN prediction records emitted during evaluation.
    :type predictions: Dict[int, List[float]]
    :param eval_examples: Iterable of evaluation examples to score with the index.
    :type eval_examples: Sequence[~knn.core.opinion_models.OpinionExample]
    :param rows: Optional iterable of per-example prediction rows. When provided,
        metrics are computed only over examples that produced predictions for the
        given ``k``.
    :type rows: Optional[Sequence[Dict[str, Any]]]
    :returns: MAE, RMSE, and R^2 metrics for each ``k``.
    :rtype: Dict[int, Dict[str, float]]
    """
    if rows:
        metrics = _metrics_from_rows(predictions, rows)
        if metrics:
            return metrics
    return _metrics_from_eval_examples(predictions, eval_examples)


@dataclass(frozen=True)
class _FilterParams:
    """Neighbour filtering parameters."""

    exclude_self: bool
    max_k: int
    self_key: Tuple[str, str]


def _filter_neighbors(
    *,
    index: OpinionIndex,
    indices: np.ndarray,
    similarities: np.ndarray,
    params: _FilterParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return filtered neighbour indices/weights as compact numpy arrays."""
    filtered_indices: List[int] = []
    filtered_weights: List[float] = []
    for candidate_idx, weight in zip(indices, similarities):
        if params.exclude_self:
            participant_key = index.targets.participant_keys[int(candidate_idx)]
            if participant_key == params.self_key:
                continue
        filtered_indices.append(int(candidate_idx))
        filtered_weights.append(float(weight))
        if len(filtered_indices) >= params.max_k:
            break
    return (
        np.asarray(filtered_indices, dtype=np.int32),
        np.asarray(filtered_weights, dtype=np.float32),
    )


def _records_for_k(
    *,
    index: OpinionIndex,
    example_before: float,
    idx_arr: np.ndarray,
    wts_arr: np.ndarray,
    unique_k: Sequence[int],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Build per-k prediction and change dictionaries from neighbours."""
    record_after: Dict[int, float] = {}
    record_change: Dict[int, float] = {}
    for k in unique_k:
        if len(idx_arr) < k:
            continue
        top_indices = idx_arr[:k]
        top_weights = wts_arr[:k]
        top_targets_after = index.targets.after[top_indices]
        top_targets_before = index.targets.before[top_indices]
        predicted_change = _weighted_mean(top_targets_after - top_targets_before, top_weights)
        anchored_prediction = float(example_before) + predicted_change
        record_after[int(k)] = float(anchored_prediction)
        record_change[int(k)] = float(predicted_change)
    return record_after, record_change


@dataclass(frozen=True)
class _RowPredParams:
    """Neighbour selection and per-k evaluation parameters."""

    unique_k: Sequence[int]
    max_k: int
    exclude_self: bool


def _row_predictions(
    *,
    index: OpinionIndex,
    example: OpinionExample,
    distances: np.ndarray,
    indices: np.ndarray,
    params: _RowPredParams,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute per-k predictions for a single ``example``.

    :param index: KNN opinion index containing embeddings and neighbour data.
    :param example: Evaluation example to score.
    :param distances: Neighbour distances for the example.
    :param indices: Neighbour row indices for the example.
    :param unique_k: Unique, sorted set of ``k`` values to evaluate.
    :param max_k: Maximum neighbourhood size to consider for the example.
    :param exclude_self: Whether to exclude the query participant from neighbours.
    :returns: Tuple of dictionaries for post-study predictions and changes keyed by ``k``.
    """
    similarities = _similarity_from_distances(distances, metric=index.metric)
    idx_arr, wts_arr = _filter_neighbors(
        index=index,
        indices=indices,
        similarities=similarities,
        params=_FilterParams(
            exclude_self=params.exclude_self,
            max_k=params.max_k,
            self_key=(example.participant_id, example.participant_study),
        ),
    )
    if idx_arr.size == 0:
        return {}, {}
    return _records_for_k(
        index=index,
        example_before=float(example.before),
        idx_arr=idx_arr,
        wts_arr=wts_arr,
        unique_k=params.unique_k,
    )


@dataclass
class _CurveAccumulator:
    """Accumulate per-k metrics and track the best-performing configuration."""

    mae_by_k: Dict[str, Optional[float]] = field(default_factory=dict)
    r2_by_k: Dict[str, Optional[float]] = field(default_factory=dict)
    _best: Optional[Tuple[int, float, float]] = None

    def add(self, k: int, values: Mapping[str, float]) -> None:
        """
        Record metrics for a specific ``k`` and update best trackers.

        :param k: Neighbour count hyper-parameter.
        :param values: Mapping that includes keys like ``mae_after`` and ``r2_after``.
        :returns: ``None``.
        """
        raw_mae = float(values.get("mae_after", float("nan")))
        raw_r2 = float(values.get("r2_after", float("nan")))
        mae_value = raw_mae if math.isfinite(raw_mae) else float("inf")
        r2_value = raw_r2 if math.isfinite(raw_r2) else float("-inf")
        self.mae_by_k[str(int(k))] = raw_mae if math.isfinite(raw_mae) else None
        self.r2_by_k[str(int(k))] = raw_r2 if math.isfinite(raw_r2) else None
        if self._is_preferred(mae_value, r2_value):
            self._best = (int(k), mae_value, r2_value)

    def best_summary(self, fallback_k: int) -> Tuple[int, Optional[float], Optional[float]]:
        """
        Return the best-performing ``k`` and associated metrics (or fallback).

        :param fallback_k: Value returned for ``k`` when no metrics were recorded.
        :returns: Tuple of ``(k, mae_after, r2_after)`` where metric entries may be ``None``.
        """
        if self._best is None:
            return fallback_k, None, None
        best_k, mae_value, r2_value = self._best
        best_mae = mae_value if math.isfinite(mae_value) else None
        best_r2 = r2_value if math.isfinite(r2_value) else None
        return best_k, best_mae, best_r2

    def _is_preferred(self, mae_value: float, r2_value: float) -> bool:
        """
        Determine whether the candidate metrics should replace the current best.

        The comparison minimises MAE first and uses R^2 as a tie-breaker to favour
        higher explanatory power when MAE differences fall within numerical tolerance.

        :param mae_value: Candidate mean absolute error (finite sentinel already applied).
        :param r2_value: Candidate R^2 score (finite sentinel already applied).
        :returns: ``True`` when the candidate should become the new best summary.
        """
        if self._best is None:
            return True
        _, best_mae, best_r2 = self._best
        if mae_value < best_mae - 1e-9:
            return True
        if mae_value <= best_mae + 1e-9 and r2_value > best_r2:
            return True
        return False


def _curve_payload(
    metrics_by_k: Dict[int, Dict[str, float]],
    *,
    n_examples: int,
) -> Optional[Dict[str, Any]]:
    """
    Convert ``metrics_by_k`` into a serialisable curve bundle.

    :param metrics_by_k: Mapping from each ``k`` to its associated opinion metrics.
    :type metrics_by_k: Dict[int, Dict[str, float]]
    :param n_examples: Total number of evaluation examples summarised in the bundle.
    :type n_examples: int
    :returns: Dictionary summarising the evaluation curve, including AUC and per-k metrics.
    :rtype: Optional[Dict[str, Any]]
    """
    if not metrics_by_k:
        return None

    accumulator = _CurveAccumulator()
    sorted_items = sorted((int(k), values) for k, values in metrics_by_k.items())
    for k, values in sorted_items:
        accumulator.add(k, values)

    fallback_k = sorted_items[0][0]
    best_k, best_mae, best_r2 = accumulator.best_summary(fallback_k)

    return {
        "metric": "mae_after",
        "mae_by_k": accumulator.mae_by_k,
        "r2_by_k": accumulator.r2_by_k,
        "best_k": int(best_k),
        "best_mae": best_mae,
        "best_r2": best_r2,
        "n_examples": int(n_examples),
    }


def _baseline_metrics(eval_examples: Sequence[OpinionExample]) -> Dict[str, float]:
    """
    Return baseline error metrics for opinion prediction.

    :param eval_examples: Iterable of evaluation examples to score with the index.
    :type eval_examples: Sequence[~knn.core.opinion_models.OpinionExample]
    :returns: Baseline error metrics for opinion prediction.
    :rtype: Dict[str, float]
    """
    truth_after = np.asarray([example.after for example in eval_examples], dtype=np.float32)
    truth_before = np.asarray([example.before for example in eval_examples], dtype=np.float32)
    baseline_mean = float(truth_after.mean()) if truth_after.size else float("nan")
    baseline_predictions = np.full_like(truth_after, baseline_mean)
    mae_mean = float(mean_absolute_error(truth_after, baseline_predictions))
    rmse_mean = float(math.sqrt(mean_squared_error(truth_after, baseline_predictions)))

    no_change_metrics = compute_opinion_metrics(
        truth_after=truth_after,
        truth_before=truth_before,
        pred_after=truth_before,
    )
    baseline_direction_accuracy = no_change_metrics.get("direction_accuracy")
    if baseline_direction_accuracy is not None and not math.isnan(baseline_direction_accuracy):
        baseline_direction_accuracy = float(baseline_direction_accuracy)
    else:
        baseline_direction_accuracy = float("nan")

    return {
        "mae_global_mean_after": mae_mean,
        "rmse_global_mean_after": rmse_mean,
        "mae_using_before": float(no_change_metrics["mae_after"]),
        "rmse_using_before": float(no_change_metrics["rmse_after"]),
        "mae_change_zero": float(no_change_metrics["mae_change"]),
        "rmse_change_zero": float(no_change_metrics["rmse_change"]),
        "calibration_slope_change_zero": no_change_metrics.get("calibration_slope"),
        "calibration_intercept_change_zero": no_change_metrics.get("calibration_intercept"),
        "calibration_ece_change_zero": no_change_metrics.get("calibration_ece"),
        "calibration_bins_change_zero": no_change_metrics.get("calibration_bins"),
        "kl_divergence_change_zero": no_change_metrics.get("kl_divergence_change"),
        "global_mean_after": baseline_mean,
        "direction_accuracy": baseline_direction_accuracy,
    }


@dataclass(frozen=True)
class _PredictionResults:
    """Intermediate predictions and metrics produced during evaluation."""

    rows: Sequence[Dict[str, Any]]
    metrics_by_k: Dict[int, Dict[str, float]]


__all__ = [
    "_PredictionResults",
    "_baseline_metrics",
    "_curve_payload",
    "_post_prediction_series",
    "_summary_metrics",
    "predict_post_indices",
]
