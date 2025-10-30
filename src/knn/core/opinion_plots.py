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
from pathlib import Path
from typing import Dict, Sequence

from common.visualization.matplotlib import plt
from common.opinion.plots import (
    OpinionHeatmapConfig,
    plot_opinion_change_heatmap,
    plot_post_opinion_heatmap,
)

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
    plot_opinion_change_heatmap(
        actual_changes=actual_changes,
        predicted_changes=predicted_changes,
        output_path=output_path,
        config=OpinionHeatmapConfig(
            logger=LOGGER,
            log_prefix="[OPINION]",
        ),
    )


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
    plot_post_opinion_heatmap(
        actual_after=actual_after,
        predicted_after=predicted_after,
        output_path=output_path,
        config=OpinionHeatmapConfig(
            logger=LOGGER,
            log_prefix="[OPINION]",
        ),
    )


__all__ = [
    "_plot_change_heatmap",
    "_plot_metric",
    "_plot_post_prediction_heatmap",
]
