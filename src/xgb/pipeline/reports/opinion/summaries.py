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

"""Opinion metric normalisation helpers."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

from common.pipeline.formatters import safe_float as _safe_float, safe_int as _safe_int

from .metrics import _difference
from ...context import OpinionSummary


def _extract_opinion_summary(data: Mapping[str, object]) -> OpinionSummary:
    """
    Normalise opinion regression metrics into a reusable summary structure.

    :param data: Raw metrics payload for a single study or selection.
    :returns: Parsed :class:`~xgb.pipeline.context.OpinionSummary` instance.
    """

    metrics_block = data.get("metrics") or {}
    baseline_raw = data.get("baseline") or {}
    baseline_block = baseline_raw if isinstance(baseline_raw, Mapping) else {}

    def metric_value(name: str) -> Optional[float]:
        """Fetch a metric float from the summary block when available.

        :param name: Metric identifier inside the ``metrics`` payload.
        :returns: Parsed float value or ``None`` if missing/unparseable.
        """
        return _safe_float(metrics_block.get(name))

    def baseline_value(*names: str) -> Optional[float]:
        """Fetch a baseline float from the summary block when available.

        :param names: Candidate baseline keys to attempt in order.
        :returns: Parsed float value or ``None`` if no key is present.
        """
        for candidate in names:
            if candidate in baseline_block:
                return _safe_float(baseline_block.get(candidate))
        return None

    summary_kwargs = {
        "mae_after": metric_value("mae_after"),
        "mae_change": metric_value("mae_change"),
        "rmse_after": metric_value("rmse_after"),
        "r2_after": metric_value("r2_after"),
        "rmse_change": metric_value("rmse_change"),
        "accuracy_after": metric_value("direction_accuracy"),
        "calibration_slope": metric_value("calibration_slope"),
        "calibration_intercept": metric_value("calibration_intercept"),
        "calibration_ece": metric_value("calibration_ece"),
        "kl_divergence_change": metric_value("kl_divergence_change"),
        "participants": _safe_int(data.get("n_participants")),
        "dataset": str(data.get("dataset")) if data.get("dataset") else None,
        "split": str(data.get("split")) if data.get("split") else None,
        "label": str(data.get("label")) if data.get("label") else None,
    }
    summary_kwargs["baseline_mae"] = baseline_value("mae_before", "mae_using_before")
    summary_kwargs["baseline_rmse_change"] = _safe_float(baseline_block.get("rmse_change_zero"))
    summary_kwargs["baseline_accuracy"] = _safe_float(baseline_block.get("direction_accuracy"))
    summary_kwargs["baseline_calibration_slope"] = _safe_float(
        baseline_block.get("calibration_slope_change_zero")
    )
    summary_kwargs["baseline_calibration_intercept"] = _safe_float(
        baseline_block.get("calibration_intercept_change_zero")
    )
    summary_kwargs["baseline_calibration_ece"] = _safe_float(
        baseline_block.get("calibration_ece_change_zero")
    )
    summary_kwargs["baseline_kl_divergence_change"] = _safe_float(
        baseline_block.get("kl_divergence_change_zero")
    )

    summary_kwargs["eligible"] = _safe_int(data.get("eligible"))
    if summary_kwargs["eligible"] is None:
        summary_kwargs["eligible"] = _safe_int(metrics_block.get("eligible"))

    mae_delta = _difference(summary_kwargs["baseline_mae"], summary_kwargs["mae_after"])
    accuracy_delta = None
    if (
        summary_kwargs["accuracy_after"] is not None
        and summary_kwargs["baseline_accuracy"] is not None
    ):
        accuracy_delta = summary_kwargs["accuracy_after"] - summary_kwargs["baseline_accuracy"]

    summary_kwargs["mae_delta"] = mae_delta
    summary_kwargs["accuracy_delta"] = accuracy_delta

    return OpinionSummary(**summary_kwargs)


def _dataset_and_split(
    metrics: Mapping[str, Mapping[str, object]],
) -> Tuple[str, str]:
    """
    Return representative dataset and split names for the metrics payload.

    :param metrics: Mapping from study identifiers to metrics payloads.
    :returns: Tuple containing dataset and split labels.
    """

    dataset_name = "unknown"
    split_name = "validation"
    for payload in metrics.values():
        summary = _extract_opinion_summary(payload)
        if dataset_name == "unknown" and summary.dataset:
            dataset_name = summary.dataset
        if summary.split:
            split_name = summary.split
        if dataset_name != "unknown" and summary.split:
            break
    return dataset_name, split_name


__all__ = [
    "_dataset_and_split",
    "_extract_opinion_summary",
]
