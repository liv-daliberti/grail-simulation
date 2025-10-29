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

"""Plotting utilities for KNN opinion evaluations."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from common.visualization.matplotlib import plt

LOGGER = logging.getLogger("knn.opinion")


def _plot_metric(
    *,
    metrics_by_k: Dict[int, Dict[str, float]],
    metric_key: str,
    output_path: Path,
) -> None:
    """
    Save a line plot of ``metric_key`` vs. ``k`` if matplotlib is available.

    :param metrics_by_k: Mapping from each ``k`` to its associated opinion metrics.
    :type metrics_by_k: Dict[int, Dict[str, float]]
    :param metric_key: Dictionary key pointing to a metric within the payload.
    :type metric_key: str
    :param output_path: Filesystem path for the generated report or figure.
    :type output_path: Path
    """
    # pylint: disable=too-many-arguments,too-many-locals
    if plt is None:  # pragma: no cover - optional dependency
        LOGGER.warning("[OPINION] Skipping %s plot (matplotlib not installed).", metric_key)
        return

    if not metrics_by_k:
        LOGGER.warning("[OPINION] Skipping %s plot (no metrics).", metric_key)
        return

    sorted_items = sorted(metrics_by_k.items())
    k_values = [item[0] for item in sorted_items]
    values = [item[1][metric_key] for item in sorted_items]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, values, marker="o")
    plt.title(f"{metric_key} vs k")
    plt.xlabel("k")
    plt.ylabel(metric_key)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_change_heatmap(
    *,
    actual_changes: Sequence[float],
    predicted_changes: Sequence[float],
    output_path: Path,
) -> None:
    """
    Render a 2D histogram comparing actual vs. predicted opinion shifts.

    :param actual_changes: Sequence of observed opinion deltas for participants.
    :type actual_changes: Sequence[float]
    :param predicted_changes: Predicted opinion deltas returned by the model.
    :type predicted_changes: Sequence[float]
    :param output_path: Filesystem path for the generated report or figure.
    :type output_path: Path
    """
    if plt is None:  # pragma: no cover - optional dependency
        LOGGER.warning("[OPINION] Skipping opinion-change heatmap (matplotlib not installed).")
        return

    if not actual_changes or not predicted_changes:
        LOGGER.warning("[OPINION] Skipping opinion-change heatmap (no valid predictions).")
        return

    actual = np.asarray(actual_changes, dtype=np.float32)
    predicted = np.asarray(predicted_changes, dtype=np.float32)
    if actual.size == 0 or predicted.size == 0:
        LOGGER.warning("[OPINION] Skipping opinion-change heatmap (empty arrays).")
        return

    min_val = float(min(actual.min(), predicted.min()))
    max_val = float(max(actual.max(), predicted.max()))
    if math.isclose(min_val, max_val):
        span = 0.1 if math.isfinite(min_val) else 1.0
        min_val -= span
        max_val += span
    else:
        extent = max(abs(min_val), abs(max_val))
        if not math.isfinite(extent) or extent <= 1e-6:
            extent = 1.0
        min_val, max_val = -extent, extent

    bins = 40
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4.5))
    hist = plt.hist2d(
        actual,
        predicted,
        bins=bins,
        range=[[min_val, max_val], [min_val, max_val]],
        cmap="magma",
        cmin=1,
    )
    plt.colorbar(hist[3], label="Participants")
    plt.plot([min_val, max_val], [min_val, max_val], color="cyan", linestyle="--", linewidth=1.0)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
    plt.axvline(0.0, color="grey", linestyle=":", linewidth=0.8)
    plt.xlabel("Actual opinion change (post - pre)")
    plt.ylabel("Predicted opinion change")
    plt.title("Predicted vs. actual opinion change")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_post_prediction_heatmap(
    *,
    actual_after: Sequence[float],
    predicted_after: Sequence[float],
    output_path: Path,
) -> None:
    """
    Render a 2D histogram comparing actual vs. predicted post-study indices.

    :param actual_after: Sequence of observed post-study opinion indices.
    :type actual_after: Sequence[float]
    :param predicted_after: Predicted post-study opinion indices returned by the model.
    :type predicted_after: Sequence[float]
    :param output_path: Filesystem path for the generated figure.
    :type output_path: Path
    """
    if plt is None:  # pragma: no cover - optional dependency
        LOGGER.warning("[OPINION] Skipping post-vs-predicted heatmap (matplotlib not installed).")
        return

    if not actual_after or not predicted_after:
        LOGGER.warning("[OPINION] Skipping post-vs-predicted heatmap (no valid predictions).")
        return

    actual = np.asarray(actual_after, dtype=np.float32)
    predicted = np.asarray(predicted_after, dtype=np.float32)
    if actual.size == 0 or predicted.size == 0:
        LOGGER.warning("[OPINION] Skipping post-vs-predicted heatmap (empty arrays).")
        return

    min_val = float(min(actual.min(), predicted.min()))
    max_val = float(max(actual.max(), predicted.max()))
    if math.isclose(min_val, max_val):
        span = 0.05 if math.isfinite(min_val) else 1.0
        min_val -= span
        max_val += span
    else:
        span = max_val - min_val
        if not math.isfinite(span) or span <= 1e-6:
            span = 0.1
        padding = 0.05 * span
        min_val -= padding
        max_val += padding

    bins = 40
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4.5))
    hist = plt.hist2d(
        actual,
        predicted,
        bins=bins,
        range=[[min_val, max_val], [min_val, max_val]],
        cmap="magma",
        cmin=1,
    )
    plt.colorbar(hist[3], label="Participants")
    plt.plot([min_val, max_val], [min_val, max_val], color="cyan", linestyle="--", linewidth=1.0)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Actual post-study opinion index")
    plt.ylabel("Predicted post-study opinion index")
    plt.title("Predicted vs. actual post-study opinion index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


__all__ = [
    "_plot_change_heatmap",
    "_plot_metric",
    "_plot_post_prediction_heatmap",
]
