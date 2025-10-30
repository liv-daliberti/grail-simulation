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

"""Shared opinion-plot helpers used across baseline pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from common.visualization.matplotlib import plt


_ArrayPair = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class OpinionHeatmapStyle:
    """
    Visual configuration applied to opinion heatmaps.

    :ivar bins: Histogram bin count used when rendering.
    :ivar xlabel: Label rendered on the x-axis.
    :ivar ylabel: Label rendered on the y-axis.
    :ivar title: Figure title describing the chart.
    :ivar show_zero_guides: Whether zero-baseline guide lines should be drawn.
    """

    bins: int
    xlabel: str
    ylabel: str
    title: str
    show_zero_guides: bool


@dataclass(frozen=True)
class OpinionHeatmapConfig:
    """
    Context describing how heatmaps should be rendered and logged.

    :ivar logger: Logger instance used for diagnostic output.
    :ivar log_prefix: Prefix prepended to log messages.
    :ivar style: Optional style overrides applied to the plot.
    """

    logger: Any
    log_prefix: str
    style: OpinionHeatmapStyle | None = None


@dataclass(frozen=True)
class HeatmapRenderParams:  # pylint: disable=too-many-instance-attributes
    """
    Aggregated rendering parameters forwarded to the plotting helper.

    :ivar arrays: Tuple of numpy arrays holding actual/predicted values.
    :ivar output_path: Destination path for the generated plot.
    :ivar bins: Histogram bin count used for both axes.
    :ivar bounds: Tuple describing shared axis limits.
    :ivar xlabel: X-axis label.
    :ivar ylabel: Y-axis label.
    :ivar title: Figure title.
    :ivar show_zero_guides: Whether zero-baseline guide lines should be drawn.
    """

    arrays: _ArrayPair
    output_path: Path
    bins: int
    bounds: Tuple[float, float]
    xlabel: str
    ylabel: str
    title: str
    show_zero_guides: bool


_DEFAULT_CHANGE_STYLE = OpinionHeatmapStyle(
    bins=40,
    xlabel="Actual opinion change (post - pre)",
    ylabel="Predicted opinion change",
    title="Predicted vs. actual opinion change",
    show_zero_guides=True,
)
_DEFAULT_POST_STYLE = OpinionHeatmapStyle(
    bins=40,
    xlabel="Actual post-study opinion index",
    ylabel="Predicted post-study opinion index",
    title="Predicted vs. actual post-study opinion index",
    show_zero_guides=False,
)


def _prepare_heatmap_inputs(
    *,
    actual: Sequence[float],
    predicted: Sequence[float],
    logger,
    log_prefix: str,
    descriptor: str,
) -> Optional[_ArrayPair]:
    """
    Validate sequences supplied to the plotting helpers and convert them to arrays.

    :param actual: Observed values captured during evaluation.
    :param predicted: Model predictions aligned with ``actual``.
    :param logger: Logger used for emitting informational messages.
    :param log_prefix: Prefix appended to log messages to hint at the caller.
    :param descriptor: Human-readable label describing the requested plot.
    :returns: Tuple of numpy arrays when plotting should proceed; otherwise ``None``.
    """
    if plt is None:  # pragma: no cover - optional dependency
        logger.warning("%s Skipping %s (matplotlib not installed).", log_prefix, descriptor)
        return None

    if not actual or not predicted:
        logger.warning("%s Skipping %s (no valid predictions).", log_prefix, descriptor)
        return None

    actual_array = np.asarray(actual, dtype=np.float32)
    predicted_array = np.asarray(predicted, dtype=np.float32)
    if actual_array.size == 0 or predicted_array.size == 0:
        logger.warning("%s Skipping %s (empty arrays).", log_prefix, descriptor)
        return None
    return actual_array, predicted_array


def _symmetric_bounds(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, float]:
    """
    Compute symmetric axis limits that encompass both arrays.

    :param actual: Observed opinion deltas rendered on the x-axis.
    :param predicted: Predicted opinion deltas rendered on the y-axis.
    :returns: Tuple of ``(min_val, max_val)`` describing common axis limits.
    """
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
    return min_val, max_val


def _padded_bounds(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, float]:
    """
    Compute padded axis limits based on the extrema across both arrays.

    :param actual: Observed opinion indices rendered on the x-axis.
    :param predicted: Predicted opinion indices rendered on the y-axis.
    :returns: Tuple of ``(min_val, max_val)`` describing padded axis limits.
    """
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
    return min_val, max_val


def _render_heatmap(*, params: HeatmapRenderParams) -> None:
    """
    Render a ``matplotlib.pyplot.hist2d`` chart with consistent styling.

    :param params: Aggregated render instructions.
    :returns: ``None``.
    """
    actual, predicted = params.arrays
    min_val, max_val = params.bounds
    params.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4.5))  # type: ignore[union-attr]
    hist = plt.hist2d(  # type: ignore[union-attr]
        actual,
        predicted,
        bins=params.bins,
        range=[[min_val, max_val], [min_val, max_val]],
        cmap="magma",
        cmin=1,
    )
    plt.colorbar(hist[3], label="Participants")  # type: ignore[union-attr]
    plt.plot(  # type: ignore[union-attr]
        [min_val, max_val],
        [min_val, max_val],
        color="cyan",
        linestyle="--",
        linewidth=1.0,
    )
    plt.xlim(min_val, max_val)  # type: ignore[union-attr]
    plt.ylim(min_val, max_val)  # type: ignore[union-attr]
    plt.gca().set_aspect("equal", adjustable="box")  # type: ignore[union-attr]
    if params.show_zero_guides:
        plt.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)  # type: ignore[union-attr]
        plt.axvline(0.0, color="grey", linestyle=":", linewidth=0.8)  # type: ignore[union-attr]
    plt.xlabel(params.xlabel)  # type: ignore[union-attr]
    plt.ylabel(params.ylabel)  # type: ignore[union-attr]
    plt.title(params.title)  # type: ignore[union-attr]
    plt.tight_layout()  # type: ignore[union-attr]
    plt.savefig(params.output_path, dpi=150)  # type: ignore[union-attr]
    plt.close()  # type: ignore[union-attr]


def plot_opinion_change_heatmap(
    *,
    actual_changes: Sequence[float],
    predicted_changes: Sequence[float],
    output_path: Path,
    config: OpinionHeatmapConfig,
) -> None:
    """
    Render a symmetric heatmap comparing predicted and actual opinion deltas.

    :param actual_changes: Observed change values derived from participant data.
    :param predicted_changes: Model-predicted change values aligned with ``actual_changes``.
    :param output_path: Destination path for the PNG artefact.
    :param config: Logging and styling configuration for the plot.
    :returns: ``None``.
    """

    arrays = _prepare_heatmap_inputs(
        actual=actual_changes,
        predicted=predicted_changes,
        logger=config.logger,
        log_prefix=config.log_prefix,
        descriptor="opinion-change heatmap",
    )
    if arrays is None:
        return

    style = config.style or _DEFAULT_CHANGE_STYLE
    bounds = _symmetric_bounds(*arrays)
    _render_heatmap(
        params=HeatmapRenderParams(
            arrays=arrays,
            output_path=output_path,
            bins=style.bins,
            bounds=bounds,
            xlabel=style.xlabel,
            ylabel=style.ylabel,
            title=style.title,
            show_zero_guides=style.show_zero_guides,
        )
    )


def plot_post_opinion_heatmap(
    *,
    actual_after: Sequence[float],
    predicted_after: Sequence[float],
    output_path: Path,
    config: OpinionHeatmapConfig,
) -> None:
    """
    Render a padded-range heatmap comparing predicted and actual post-study indices.

    :param actual_after: Observed post-study opinion indices.
    :param predicted_after: Predicted post-study opinion indices returned by the model.
    :param output_path: Destination path for the PNG artefact.
    :param config: Logging and styling configuration for the plot.
    :returns: ``None``.
    """

    arrays = _prepare_heatmap_inputs(
        actual=actual_after,
        predicted=predicted_after,
        logger=config.logger,
        log_prefix=config.log_prefix,
        descriptor="post-vs-predicted heatmap",
    )
    if arrays is None:
        return

    style = config.style or _DEFAULT_POST_STYLE
    bounds = _padded_bounds(*arrays)
    _render_heatmap(
        params=HeatmapRenderParams(
            arrays=arrays,
            output_path=output_path,
            bins=style.bins,
            bounds=bounds,
            xlabel=style.xlabel,
            ylabel=style.ylabel,
            title=style.title,
            show_zero_guides=style.show_zero_guides,
        )
    )


def make_change_heatmap_plotter(
    *,
    logger: Any,
    log_prefix: str,
    style: OpinionHeatmapStyle | None = None,
) -> Callable[[Sequence[float], Sequence[float], Path], None]:
    """
    Build a callable that renders opinion-change heatmaps with shared settings.

    :param logger: Logger instance used for informational messages.
    :param log_prefix: Prefix prepended to log messages emitted during plotting.
    :param style: Optional heatmap style overrides applied when plotting.
    :returns: Callable that mirrors :func:`plot_opinion_change_heatmap`.
    """

    def _plot_change_heatmap(
        *,
        actual_changes: Sequence[float],
        predicted_changes: Sequence[float],
        output_path: Path,
    ) -> None:
        plot_opinion_change_heatmap(
            actual_changes=actual_changes,
            predicted_changes=predicted_changes,
            output_path=output_path,
            config=OpinionHeatmapConfig(
                logger=logger,
                log_prefix=log_prefix,
                style=style,
            ),
        )

    return _plot_change_heatmap


def make_post_heatmap_plotter(
    *,
    logger: Any,
    log_prefix: str,
    style: OpinionHeatmapStyle | None = None,
) -> Callable[[Sequence[float], Sequence[float], Path], None]:
    """
    Build a callable that renders post-study opinion heatmaps with shared settings.

    :param logger: Logger instance used for informational messages.
    :param log_prefix: Prefix prepended to log messages emitted during plotting.
    :param style: Optional heatmap style overrides applied when plotting.
    :returns: Callable that mirrors :func:`plot_post_opinion_heatmap`.
    """

    def _plot_post_heatmap(
        *,
        actual_after: Sequence[float],
        predicted_after: Sequence[float],
        output_path: Path,
    ) -> None:
        plot_post_opinion_heatmap(
            actual_after=actual_after,
            predicted_after=predicted_after,
            output_path=output_path,
            config=OpinionHeatmapConfig(
                logger=logger,
                log_prefix=log_prefix,
                style=style,
            ),
        )

    return _plot_post_heatmap


__all__ = [
    "plot_opinion_change_heatmap",
    "plot_post_opinion_heatmap",
    "OpinionHeatmapConfig",
    "OpinionHeatmapStyle",
    "make_change_heatmap_plotter",
    "make_post_heatmap_plotter",
]
