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

"""Curve extraction and plotting helpers for XGBoost pipeline reports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

from common.report_utils import extract_curve_sections, extract_numeric_series

from .shared import LOGGER, _slugify_label, plt


def _extract_eligible_curve_steps(
    curve_block: Mapping[str, object]
) -> Tuple[List[int], List[float]]:
    """Extract eligible-only accuracy curve series if present."""
    eligible_map = (
        curve_block.get("eligible_accuracy_by_round")
        or curve_block.get("eligible_accuracy_by_step")
    )
    if not isinstance(eligible_map, Mapping):
        return ([], [])
    return extract_numeric_series(eligible_map)


def _plot_eligible_overlay(axis, payload: Mapping[str, object]) -> None:
    """Overlay eligible-only accuracy curves onto an existing axis when available."""
    bundle = _load_curve_bundle(payload)
    if not isinstance(bundle, Mapping):
        return
    sections = extract_curve_sections(bundle)
    if sections is None:
        return
    eval_curve, train_curve = sections
    eligible_eval_x, eligible_eval_y = _extract_eligible_curve_steps(eval_curve)
    if eligible_eval_x and eligible_eval_y:
        axis.plot(
            eligible_eval_x,
            eligible_eval_y,
            linestyle=":",
            marker="o",
            label="validation (eligible)",
        )
    if isinstance(train_curve, Mapping):
        eligible_train_x, eligible_train_y = _extract_eligible_curve_steps(train_curve)
        if eligible_train_x and eligible_train_y:
            axis.plot(
                eligible_train_x,
                eligible_train_y,
                linestyle=":",
                marker="o",
                label="training (eligible)",
            )


def _extract_curve_steps(curve_block: Mapping[str, object]) -> Tuple[List[int], List[float]]:
    """
    Extract sorted evaluation steps and accuracies from a curve payload.

    :param curve_block: Curve payload containing accuracy checkpoints keyed by step or round.
    :type curve_block: Mapping[str, object]
    :returns: Pair of ``(steps, accuracies)`` sorted by evaluation index.
    :rtype: Tuple[List[int], List[float]]
    """

    accuracy_map = (
        curve_block.get("accuracy_by_round") or curve_block.get("accuracy_by_step")
    )
    if not isinstance(accuracy_map, Mapping):
        return ([], [])
    return extract_numeric_series(accuracy_map)


def _extract_accuracy_curves(
    payload: Mapping[str, object]
) -> Optional[Tuple[List[int], List[float], List[int], List[float], str, str]]:
    """
    Pull validation and training cumulative accuracy curves from ``payload``.

    Args:
        payload: Metrics payload, potentially referencing embedded or on-disk curves.

    Returns:
        Tuple containing validation x/y pairs followed by training x/y pairs, or ``None`` when
        the payload does not expose the necessary series.
    """

    curve_bundle = _load_curve_bundle(payload)
    if not isinstance(curve_bundle, Mapping):
        return None
    axis_label = str(curve_bundle.get("axis_label") or "Evaluated examples")
    y_label = str(curve_bundle.get("y_label") or "Cumulative accuracy")
    sections = extract_curve_sections(curve_bundle)
    if sections is None:
        return None
    eval_curve, train_curve = sections
    eval_x, eval_y = _extract_curve_steps(eval_curve)
    if not eval_x:
        return None

    train_x, train_y = (
        _extract_curve_steps(train_curve) if train_curve is not None else ([], [])
    )

    return (eval_x, eval_y, train_x, train_y, axis_label, y_label)


def _extract_mae_curves(
    payload: Mapping[str, object]
) -> Optional[Tuple[List[int], List[float], List[int], List[float], str, str]]:
    """
    Pull validation and training MAE curves from ``payload``.

    Args:
        payload: Metrics payload, potentially referencing embedded or on-disk curves.

    Returns:
        Tuple containing validation x/y pairs followed by training x/y pairs, or ``None`` when
        the payload does not expose the necessary series.
    """

    curve_bundle = _load_curve_bundle(payload)
    if not isinstance(curve_bundle, Mapping):
        return None
    axis_label = str(curve_bundle.get("axis_label") or "Boosting rounds")
    y_label = str(curve_bundle.get("y_label") or "Mean absolute error")
    sections = extract_curve_sections(curve_bundle)
    if sections is None:
        return None
    eval_curve, train_curve = sections
    eval_source = eval_curve.get("mae_by_round") or eval_curve.get("mae_by_step")
    if not isinstance(eval_source, Mapping):
        return None
    eval_x, eval_y = extract_numeric_series(eval_source)
    if not eval_x:
        return None

    train_source = (
        (train_curve.get("mae_by_round") or train_curve.get("mae_by_step"))
        if train_curve is not None
        else None
    )
    train_x, train_y = (
        extract_numeric_series(train_source)
        if isinstance(train_source, Mapping)
        else ([], [])
    )

    return (eval_x, eval_y, train_x, train_y, axis_label, y_label)


@dataclass
class _CurveSeries:
    """Container for paired validation/training curve data."""

    label: str
    eval_x: List[int]
    eval_y: List[float]
    train_x: List[int]
    train_y: List[float]
    x_label: str
    y_label: str

    def has_training(self) -> bool:
        """Return ``True`` when training curves are available."""

        return bool(self.train_x and self.train_y)


def _build_curve_series(
    label: str,
    payload: Mapping[str, object],
    extractor: Callable[
        [Mapping[str, object]],
        Optional[Tuple[List[int], List[float], List[int], List[float], str, str]],
    ],
) -> Optional[_CurveSeries]:
    """
    Construct a ``_CurveSeries`` using the provided extraction callable.

    :param label: Title to associate with the curve.
    :param payload: Metrics payload containing the curve data.
    :param extractor: Function retrieving paired validation/training series.
    :returns: Populated ``_CurveSeries`` or ``None`` when extraction fails.
    """

    curves = extractor(payload)
    if curves is None:
        return None
    eval_x, eval_y, train_x, train_y, x_label, y_label = curves
    return _CurveSeries(label, eval_x, eval_y, train_x, train_y, x_label, y_label)


def _collect_curve_series(
    entries: Sequence[Tuple[str, Mapping[str, object]]],
    extractor: Callable[
        [Mapping[str, object]],
        Optional[Tuple[List[int], List[float], List[int], List[float], str, str]],
    ],
) -> List[_CurveSeries]:
    """
    Gather curve series for each ``(label, payload)`` pair.

    :param entries: Iterable of labelled payloads.
    :param extractor: Callable converting each payload into curve data.
    :returns: List of successfully extracted curve series.
    """

    collected: List[_CurveSeries] = []
    for label, payload in entries:
        series = _build_curve_series(label, payload, extractor)
        if series is not None:
            collected.append(series)
    return collected


def _plot_curve_on_axis(
    axis,
    series: _CurveSeries,
    *,
    legend_loc: str = "best",
) -> None:
    """
    Plot validation and training curves on ``axis``.

    :param axis: Matplotlib axis receiving the plot.
    :param series: Curve data to render.
    :param legend_loc: Preferred legend location.
    """

    axis.plot(series.eval_x, series.eval_y, marker="o", label="validation")
    if series.has_training():
        axis.plot(
            series.train_x,
            series.train_y,
            marker="o",
            linestyle="--",
            label="training",
        )
    axis.set_title(series.label)
    axis.set_xlabel(series.x_label)
    axis.set_ylabel(series.y_label)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axis.legend(loc=legend_loc)


def _load_curve_bundle(payload: Mapping[str, object]) -> Optional[Mapping[str, object]]:
    """
    Load the stored curve metrics bundle, reading from disk when required.

    :param payload: Metrics dictionary potentially containing in-memory or on-disk curves.
    :type payload: Mapping[str, object]
    :returns: Curve metrics mapping or ``None`` when unavailable.
    :rtype: Optional[Mapping[str, object]]
    """

    curve_bundle = payload.get("curve_metrics")
    if isinstance(curve_bundle, Mapping):
        return curve_bundle
    curve_path = payload.get("curve_metrics_path")
    if not curve_path:
        return None
    try:
        with open(curve_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, Mapping):
                return loaded
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - logging aid
        LOGGER.warning("Unable to read curve metrics from %s: %s", curve_path, exc)
    return None


def _format_scalar(value: object) -> str:
    """Return a short numeric string for plot labels or an em dash."""
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{value_float:.3f}"


def _decorate_curve_labels(
    entries: Sequence[Tuple[str, Mapping[str, object]]]
) -> List[Tuple[str, Mapping[str, object]]]:
    """Return entries with labels decorated by scalar metrics.

    This helper reduces locals in callers that need to build decorated lists.
    """

    decorated: List[Tuple[str, Mapping[str, object]]] = []
    for label, payload in entries:
        acc = _format_scalar(payload.get("accuracy"))
        elig = _format_scalar(payload.get("accuracy_eligible"))
        decorated.append((f"{label} (acc {acc}, elig {elig})", payload))
    return decorated


def _compute_grid(num_items: int) -> Tuple[int, int]:
    """Compute a reasonable (rows, cols) grid for ``n`` plots."""

    cols = min(3, max(1, num_items))
    rows = int(math.ceil(num_items / cols))
    return rows, cols


def _plot_xgb_curve(
    *,
    directory: Path,
    study_label: str,
    study_key: str,
    payload: Mapping[str, object],
) -> Optional[str]:
    """
    Persist a training/validation accuracy curve plot for a study.

    :param directory: Report directory where plots are stored.
    :type directory: Path
    :param study_label: Human-readable study label.
    :type study_label: str
    :param study_key: Study identifier used for slug generation.
    :type study_key: str
    :param payload: Metrics payload containing curve information.
    :type payload: Mapping[str, object]
    :returns: Relative path to the generated image or ``None`` when plotting fails.
    :rtype: Optional[str]
    """

    if plt is None:  # pragma: no cover - optional dependency
        return None
    label_base = study_label or study_key or "study"
    acc = _format_scalar(payload.get("accuracy"))
    elig = _format_scalar(payload.get("accuracy_eligible"))
    label = f"{label_base} (acc {acc}, elig {elig})"
    series = _build_curve_series(label, payload, _extract_accuracy_curves)
    if series is None:
        return None

    curves_dir = directory / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify_label(series.label, fallback="study")
    plot_path = curves_dir / f"{slug}.png"

    fig, axis = plt.subplots(figsize=(6, 3.5))  # type: ignore[attr-defined]
    _plot_curve_on_axis(axis, series)
    # Overlay eligible-only series when available
    _plot_eligible_overlay(axis, payload)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)  # type: ignore[attr-defined]
    plt.close(fig)  # type: ignore[attr-defined]
    try:
        return plot_path.relative_to(directory).as_posix()
    except ValueError:
        return plot_path.as_posix()


def _plot_xgb_curve_overview(
    *,
    directory: Path,
    entries: Sequence[Tuple[str, Mapping[str, object]]],
) -> Optional[str]:
    """
    Render a multi-panel overview of validation/training accuracy curves.

    Args:
        directory: Report directory receiving the generated image.
        entries: Iterable of ``(label, payload)`` pairs supplying curve data.

    Returns:
        Relative path to the overview figure or ``None`` when curves are unavailable.
    """

    if plt is None:  # pragma: no cover - optional dependency
        return None

    # Decorate labels with scalar metrics for quick comparison.
    decorated = _decorate_curve_labels(entries)

    series_list = _collect_curve_series(decorated, _extract_accuracy_curves)
    if not series_list:
        return None

    curves_dir = directory / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    rows, cols = _compute_grid(len(series_list))
    fig, axes = plt.subplots(  # type: ignore[attr-defined]
        rows,  # type: ignore[arg-type]
        cols,  # type: ignore[arg-type]
        figsize=(4.6 * cols, 3.2 * rows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    axes_flat = [axis for row_axes in axes for axis in row_axes]
    for axis, series, entry in zip(axes_flat, series_list, entries):
        _plot_curve_on_axis(axis, series, legend_loc="lower right")
        # entry is (label, payload)
        _plot_eligible_overlay(axis, entry[1])
    for axis in axes_flat[len(series_list):]:
        axis.axis("off")

    fig.tight_layout()
    overview_path = curves_dir / "accuracy_overview.png"
    fig.savefig(overview_path, dpi=135)  # type: ignore[attr-defined]
    plt.close(fig)  # type: ignore[attr-defined]
    try:
        return overview_path.relative_to(directory).as_posix()
    except ValueError:
        return overview_path.as_posix()


def _plot_opinion_curve(  # pylint: disable=too-many-locals,too-many-return-statements
    *,
    directory: Path,
    study_label: str,
    study_key: str,
    payload: Mapping[str, object],
) -> Optional[str]:
    """
    Persist a MAE training/validation curve plot for the opinion regressor.

    :param directory: Report directory where plots are stored.
    :type directory: Path
    :param study_label: Human-readable study label.
    :type study_label: str
    :param study_key: Study identifier used for slug generation.
    :type study_key: str
    :param payload: Metrics payload containing curve information.
    :type payload: Mapping[str, object]
    :returns: Relative path to the generated image or ``None`` when unavailable.
    :rtype: Optional[str]
    """

    if plt is None:  # pragma: no cover - optional dependency
        return None
    label = study_label or study_key or "study"
    series = _build_curve_series(label, payload, _extract_mae_curves)
    if series is None:
        return None

    curves_dir = directory / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify_label(series.label, fallback="study")
    plot_path = curves_dir / f"{slug}_mae.png"

    fig, axis = plt.subplots(figsize=(6, 3.5))  # type: ignore[attr-defined]
    _plot_curve_on_axis(axis, series)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)  # type: ignore[attr-defined]
    plt.close(fig)  # type: ignore[attr-defined]
    try:
        return plot_path.relative_to(directory).as_posix()
    except ValueError:
        return plot_path.as_posix()


__all__ = [
    "_CurveSeries",
    "_build_curve_series",
    "_collect_curve_series",
    "_extract_accuracy_curves",
    "_extract_mae_curves",
    "_plot_curve_on_axis",
    "_plot_opinion_curve",
    "_plot_xgb_curve",
    "_plot_xgb_curve_overview",
]
