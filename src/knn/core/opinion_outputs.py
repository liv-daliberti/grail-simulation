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

"""Output helpers for persisting opinion evaluation artefacts."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from common.opinion import OpinionSpec

from .data import DEFAULT_DATASET_SOURCE, EVAL_SPLIT
from .evaluate import resolve_reports_dir
from .opinion_models import OpinionIndex
from .opinion_plots import _plot_change_heatmap, _plot_metric, _plot_post_prediction_heatmap
from .opinion_predictions import _post_prediction_series

LOGGER = logging.getLogger("knn.opinion")


@dataclass(frozen=True)
class _OutputContext:
    """Lightweight wrapper for output-related runtime state."""

    args: Any
    spec: OpinionSpec
    index: OpinionIndex
    outputs_root: Path


@dataclass(frozen=True)
class _OutputPayload:
    """Container bundling metrics and per-example predictions for export."""

    rows: Sequence[Dict[str, Any]]
    metrics_by_k: Dict[int, Dict[str, float]]
    baseline: Dict[str, float]
    best_k: int
    curve_metrics: Optional[Dict[str, Any]] = None


def _opinion_change_series(
    rows: Sequence[Dict[str, Any]],
    best_k: int,
) -> Tuple[List[float], List[float]]:
    """
    Return paired lists of actual and predicted opinion changes.

    :param rows: Per-participant prediction rows containing change metadata.
    :type rows: Sequence[Dict[str, Any]]
    :param best_k: ``k`` value whose predictions should be compared.
    :type best_k: int
    :returns: Tuple of ``(actual_changes, predicted_changes)`` lists.
    :rtype: Tuple[List[float], List[float]]
    """

    actual_changes: List[float] = []
    predicted_changes: List[float] = []
    for row in rows:
        prediction = row.get("predictions_by_k", {}).get(best_k)
        if prediction is None:
            continue
        before = float(row["before_index"])
        actual_changes.append(float(row["after_index"]) - before)
        change_lookup = row.get("predicted_change_by_k", {})
        predicted_change = change_lookup.get(best_k)
        if predicted_change is None:
            predicted_change = float(prediction) - before
        predicted_changes.append(float(predicted_change))
    return actual_changes, predicted_changes


def _write_prediction_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """
    Serialise per-row predictions to ``path`` in JSONL format.

    :param path: Destination file path for the JSONL export.
    :type path: Path
    :param rows: Iterable of per-participant prediction payloads.
    :type rows: Sequence[Dict[str, Any]]
    """

    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            change_dict = {
                str(k): float(v) for k, v in row.get("predicted_change_by_k", {}).items()
            }
            serializable = {
                **row,
                "predictions_by_k": {str(k): float(v) for k, v in row["predictions_by_k"].items()},
                "predicted_change_by_k": change_dict,
            }
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def _build_metric_bundle(values: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Convert raw metrics into the serialisable structure persisted per ``k``.

    :param values: Mapping of metric names to computed values for a given ``k``.
    :type values: Mapping[str, Any]
    :returns: Filtered metrics bundle containing serialisable values only.
    :rtype: Dict[str, Any]
    """

    bundle = {
        "mae_after": float(values["mae_after"]),
        "rmse_after": float(values["rmse_after"]),
        "r2_after": float(values["r2_after"]),
        "mae_change": float(values["mae_change"]),
    }
    for key in (
        "rmse_change",
        "direction_accuracy",
        "calibration_slope",
        "calibration_intercept",
        "calibration_ece",
        "kl_divergence_change",
    ):
        optional_value = values.get(key)
        if optional_value is None:
            continue
        try:
            numeric_value = float(optional_value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric_value):
            continue
        bundle[key] = numeric_value
    calibration_bins = values.get("calibration_bins")
    if calibration_bins:
        bundle["calibration_bins"] = calibration_bins
    eligible_value = values.get("eligible")
    if eligible_value is not None:
        try:
            bundle["eligible"] = int(eligible_value)
        except (TypeError, ValueError):
            pass
    return bundle


def _build_plot_bundle(plots: Mapping[str, Path]) -> Dict[str, str]:
    """
    Normalise metric plot paths for persistence.

    :param plots: Mapping of plot identifiers to filesystem paths.
    :type plots: Mapping[str, Path]
    :returns: Dictionary of plot identifiers mapped to relative path strings.
    :rtype: Dict[str, str]
    """

    bundle: Dict[str, str] = {
        "mae_vs_k": str(plots["mae"]),
        "r2_vs_k": str(plots["r2"]),
        "change_heatmap": str(plots["heatmap"]),
    }
    post_heatmap_path = plots.get("post_heatmap")
    if post_heatmap_path is not None:
        bundle["post_vs_predicted_heatmap"] = str(post_heatmap_path)
    return bundle


def _compose_metrics_record(
    context: _OutputContext,
    payload: _OutputPayload,
    plots: Mapping[str, Path],
) -> Dict[str, Any]:
    """
    Build the metrics payload persisted alongside predictions.

    :param context: Runtime configuration referencing CLI args and output roots.
    :type context: _OutputContext
    :param payload: Metrics, predictions, and derived artefacts to persist.
    :type payload: _OutputPayload
    :param plots: Mapping of generated plots for inclusion in the payload.
    :type plots: Mapping[str, Path]
    :returns: Serialisable dictionary written to the metrics JSON file.
    :rtype: Dict[str, Any]
    """

    metrics_by_k = {
        str(k): _build_metric_bundle(values)
        for k, values in payload.metrics_by_k.items()
    }
    plot_bundle = _build_plot_bundle(plots)
    record: Dict[str, Any] = {
        "model": "knn_opinion",
        "feature_space": context.index.feature_space,
        "dataset": context.args.dataset or DEFAULT_DATASET_SOURCE,
        "study": context.spec.key,
        "issue": context.spec.issue,
        "label": context.spec.label,
        "split": EVAL_SPLIT,
        "n_participants": len(payload.rows),
        "metrics_by_k": metrics_by_k,
        "baseline": payload.baseline,
        "best_k": int(payload.best_k),
        "best_metrics": payload.metrics_by_k.get(int(payload.best_k), {}),
        "plots": plot_bundle,
    }
    if payload.curve_metrics:
        record["curve_metrics"] = payload.curve_metrics
    best_metrics = record["best_metrics"]
    if (
        "direction_accuracy" in best_metrics
        and best_metrics["direction_accuracy"] is not None
    ):
        record["best_direction_accuracy"] = float(best_metrics["direction_accuracy"])
    eligible_best = best_metrics.get("eligible")
    if eligible_best is not None:
        record["eligible"] = int(eligible_best)
    return record


def _write_outputs(
    *,
    context: _OutputContext,
    payload: _OutputPayload,
) -> None:
    """
    Persist per-example predictions, metrics, and plots.

    :param context: Runtime configuration referencing CLI args and output roots.
    :type context: _OutputContext
    :param payload: Metrics, predictions, and derived artefacts to persist.
    :type payload: _OutputPayload
    """
    study_dir = context.outputs_root / context.spec.key
    study_dir.mkdir(parents=True, exist_ok=True)

    actual_changes, predicted_changes = _opinion_change_series(payload.rows, payload.best_k)
    post_actual, post_pred = _post_prediction_series(payload.rows, payload.best_k)

    predictions_path = study_dir / f"opinion_knn_{context.spec.key}_{EVAL_SPLIT}.jsonl"
    _write_prediction_rows(predictions_path, payload.rows)

    reports_dir = (
        resolve_reports_dir(Path(context.args.out_dir))
        / "knn"
        / context.index.feature_space
        / "opinion"
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots = {
        "mae": reports_dir / f"mae_{context.spec.key}.png",
        "r2": reports_dir / f"r2_{context.spec.key}.png",
        "heatmap": reports_dir / f"change_heatmap_{context.spec.key}.png",
        "post_heatmap": reports_dir / f"post_heatmap_{context.spec.key}.png",
    }
    _plot_metric(
        metrics_by_k=payload.metrics_by_k,
        metric_key="mae_after",
        output_path=plots["mae"],
    )
    _plot_metric(
        metrics_by_k=payload.metrics_by_k,
        metric_key="r2_after",
        output_path=plots["r2"],
    )
    _plot_change_heatmap(
        actual_changes=actual_changes,
        predicted_changes=predicted_changes,
        output_path=plots["heatmap"],
    )
    _plot_post_prediction_heatmap(
        actual_after=post_actual,
        predicted_after=post_pred,
        output_path=plots["post_heatmap"],
    )

    metrics_path = study_dir / f"opinion_knn_{context.spec.key}_{EVAL_SPLIT}_metrics.json"
    metrics_record = _compose_metrics_record(
        context,
        payload,
        plots,
    )
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_record, handle, ensure_ascii=False, indent=2)

    LOGGER.info(
        "[OPINION] Wrote predictions=%s metrics=%s",
        predictions_path,
        metrics_path,
    )


__all__ = [
    "_OutputContext",
    "_OutputPayload",
    "_build_metric_bundle",
    "_compose_metrics_record",
    "_opinion_change_series",
    "_write_outputs",
]
