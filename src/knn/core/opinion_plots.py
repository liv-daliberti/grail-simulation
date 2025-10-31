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
from typing import Dict

from common.visualization.matplotlib import plt
from common.opinion.plots import make_change_heatmap_plotter, make_post_heatmap_plotter

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


_plot_change_heatmap = make_change_heatmap_plotter(
    logger=LOGGER,
    log_prefix="[OPINION]",
)
_plot_change_heatmap.__doc__ = (
    "Render a 2D histogram comparing actual vs. predicted opinion shifts."
)

_plot_post_prediction_heatmap = make_post_heatmap_plotter(
    logger=LOGGER,
    log_prefix="[OPINION]",
)
_plot_post_prediction_heatmap.__doc__ = (
    "Render a 2D histogram comparing actual vs. predicted post-study indices."
)


__all__ = [
    "_plot_change_heatmap",
    "_plot_metric",
    "_plot_post_prediction_heatmap",
]
