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

"""Opinion report builders for the modular KNN pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from ..context import StudySpec
from ...core.opinion_plots import (
    _plot_change_heatmap,
    _plot_metric,
    _plot_post_prediction_heatmap,
)
from .opinion_csv import _write_knn_opinion_csv
from .opinion_portfolio import (
    _OpinionPortfolioStats,
    _knn_opinion_cross_study_diagnostics,
)
from .opinion_sections import (
    _opinion_dataset_info,
    _opinion_feature_sections,
    _opinion_heatmap_section,
    _opinion_report_intro,
    _opinion_takeaways,
)
from .shared import LOGGER


@dataclass(frozen=True)
class OpinionReportOptions:
    """
    Bundle of optional parameters for building the opinion report.

    :param allow_incomplete: When ``True``, allow placeholder output when metrics are missing.
    :type allow_incomplete: bool
    :param title: Markdown title inserted at the top of the report.
    :type title: str
    :param description_lines: Optional Markdown description beneath the title.
    :type description_lines: Optional[Sequence[str]]
    :param metrics_line: Optional override for the metrics summary bullet.
    :type metrics_line: Optional[str]
    :param predictions_root: Root directory containing cached opinion predictions.
    :type predictions_root: Optional[Path]
    :param regenerate_plots: Rebuild matplotlib artefacts before emitting the report.
    :type regenerate_plots: bool
    :param asset_subdir: Subdirectory name used when storing plot artefacts.
    :type asset_subdir: str
    """

    allow_incomplete: bool = False
    title: str = "# KNN Opinion Shift Study"
    description_lines: Optional[Sequence[str]] = None
    metrics_line: Optional[str] = None
    predictions_root: Optional[Path] = None
    regenerate_plots: bool = True
    asset_subdir: str = "opinion"


@dataclass
class _PredictionVectors:
    """Container for vectors used when regenerating opinion plots."""

    actual_changes: List[float]
    predicted_changes: List[float]
    actual_after: List[float]
    predicted_after: List[float]


def _build_opinion_report(
    *,
    output_path: Path,
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
    options: OpinionReportOptions | None = None,
) -> None:
    """
    Compose the opinion regression report at ``output_path``.

    :param output_path: Filesystem path for the generated report or figure.
    :type output_path: Path
    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[~knn.pipeline.context.StudySpec]
    :param options: Optional bundle controlling report presentation.
    :type options: OpinionReportOptions | None
    :returns: ``None``. The Markdown report (and CSV) are written to disk.
    :rtype: None
    """
    options = options or OpinionReportOptions()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        if not options.allow_incomplete:
            raise RuntimeError("No opinion metrics available to build the opinion report.")
        placeholder = [
            options.title,
            "",
            (
                "Opinion regression metrics are not available yet. "
                "Execute the finalize stage to refresh these results."
            ),
            "",
            "This placeholder was generated with `--allow-incomplete` enabled.",
            "",
        ]
        output_path.write_text("\n".join(placeholder), encoding="utf-8")
        return
    if options.regenerate_plots:
        _refresh_opinion_plots(
            output_root=output_path.parent.parent,
            metrics=metrics,
            predictions_root=options.predictions_root,
            asset_subdir=options.asset_subdir,
        )
    dataset_name, split = _opinion_dataset_info(metrics)
    intro_lines = options.description_lines
    if intro_lines is None:
        intro_lines = [
            (
                "This study evaluates a second KNN baseline that predicts each "
                "participant's post-study opinion index."
            )
        ]
    metrics_text = options.metrics_line or (
        "- Metrics: MAE / RMSE / R² / directional accuracy / MAE (change) / "
        "RMSE (change) / calibration slope & intercept / calibration ECE / "
        "KL divergence, compared against a no-change baseline."
    )
    lines: List[str] = []
    lines.extend(
        _opinion_report_intro(
            dataset_name,
            split,
            title=options.title,
            description_lines=intro_lines,
            metrics_line=metrics_text,
        )
    )
    lines.extend(_opinion_feature_sections(metrics, studies, asset_subdir=options.asset_subdir))
    lines.extend(
        _opinion_heatmap_section(
            output_path,
            metrics,
            asset_subdir=options.asset_subdir,
        )
    )
    lines.extend(_knn_opinion_cross_study_diagnostics(metrics, studies))
    lines.extend(_opinion_takeaways(metrics, studies))
    output_path.write_text("\n".join(lines), encoding="utf-8")
    # Emit CSV dump combining all feature spaces and studies
    _write_knn_opinion_csv(output_path.parent, metrics, studies)


def _refresh_opinion_plots(
    *,
    output_root: Path,
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    predictions_root: Optional[Path],
    asset_subdir: str,
) -> None:
    """
    Regenerate opinion matplotlib artefacts so reports remain self-contained.

    :param output_root: Root directory where opinion assets should be written.
    :type output_root: Path
    :param metrics: Cached metrics organised by feature space and study key.
    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param predictions_root: Directory containing cached prediction JSONL files.
    :type predictions_root: Optional[Path]
    """

    if not metrics:
        return
    for feature_space, per_feature in metrics.items():
        if not per_feature:
            continue
        feature_dir = _ensure_feature_dir(
            output_root,
            feature_space,
            asset_subdir=asset_subdir,
        )
        for study_key, payload in per_feature.items():
            _refresh_study_plots(
                feature_dir=feature_dir,
                feature_space=feature_space,
                study_key=study_key,
                payload=payload,
                predictions_root=predictions_root,
            )


def _ensure_feature_dir(output_root: Path, feature_space: str, *, asset_subdir: str) -> Path:
    """Create and return the feature-specific output directory."""
    feature_dir = output_root / feature_space / asset_subdir
    feature_dir.mkdir(parents=True, exist_ok=True)
    return feature_dir


def _refresh_study_plots(
    *,
    feature_dir: Path,
    feature_space: str,
    study_key: str,
    payload: Mapping[str, object],
    predictions_root: Optional[Path],
) -> None:
    """Rebuild opinion plots for a single feature-space/study pair."""
    _plot_numeric_metrics(feature_dir, study_key, payload.get("metrics_by_k"))
    best_k = _extract_best_k(payload.get("best_k"))
    if best_k is None or predictions_root is None:
        return
    predictions_path = predictions_root / feature_space / study_key / (
        f"opinion_knn_{study_key}_validation.jsonl"
    )
    rows = _load_prediction_rows(predictions_path, feature_space, study_key)
    if not rows:
        return
    vectors = _build_prediction_vectors(rows, best_k)
    _render_prediction_plots(feature_dir, study_key, vectors)


def _plot_numeric_metrics(
    feature_dir: Path,
    study_key: str,
    metrics_by_k: object,
) -> None:
    """Emit MAE and R² plots when numeric sweep metrics are available."""
    numeric_metrics = _coerce_metric_lookup(metrics_by_k)
    if not numeric_metrics:
        return
    _plot_metric(
        metrics_by_k=numeric_metrics,
        metric_key="mae_after",
        output_path=feature_dir / f"mae_{study_key}.png",
    )
    _plot_metric(
        metrics_by_k=numeric_metrics,
        metric_key="r2_after",
        output_path=feature_dir / f"r2_{study_key}.png",
    )


def _coerce_metric_lookup(metrics_by_k: object) -> Mapping[int, Mapping[str, float]]:
    """Return a mapping keyed by integer k values when coercion succeeds."""
    if not isinstance(metrics_by_k, Mapping):
        return {}
    numeric_metrics: dict[int, Mapping[str, float]] = {}
    for k_value, bundle in metrics_by_k.items():
        try:
            numeric_metrics[int(k_value)] = bundle
        except (TypeError, ValueError):
            continue
    return numeric_metrics


def _extract_best_k(candidate: object) -> int | None:
    """Coerce the ``best_k`` payload to an integer when possible."""
    try:
        return int(candidate)
    except (TypeError, ValueError):
        return None


def _load_prediction_rows(
    predictions_path: Path, feature_space: str, study_key: str
) -> List[Mapping[str, object]]:
    """Read cached prediction rows if the JSONL file exists."""
    if not predictions_path.exists():
        LOGGER.warning(
            "Skipping heatmap regeneration for %s/%s; predictions missing at %s.",
            feature_space,
            study_key,
            predictions_path,
        )
        return []
    rows: List[Mapping[str, object]] = []
    try:
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError:
                    LOGGER.debug(
                        "Skipping malformed prediction row for %s/%s.",
                        feature_space,
                        study_key,
                    )
    except OSError:
        LOGGER.warning(
            "Unable to read cached predictions at %s for %s/%s.",
            predictions_path,
            feature_space,
            study_key,
        )
    return rows


def _build_prediction_vectors(
    rows: Sequence[Mapping[str, object]],
    best_k: int,
) -> _PredictionVectors:
    """Aggregate prediction vectors used for change and post heatmaps."""
    vectors = _PredictionVectors(
        actual_changes=[],
        predicted_changes=[],
        actual_after=[],
        predicted_after=[],
    )
    k_token = str(best_k)
    for row in rows:
        parsed = _parse_prediction_row(row, best_k, k_token)
        if parsed is None:
            continue
        before_val, after_val, pred_after_val, pred_change_val = parsed
        if (
            before_val is not None
            and after_val is not None
            and pred_change_val is not None
        ):
            vectors.actual_changes.append(after_val - before_val)
            vectors.predicted_changes.append(pred_change_val)
        if after_val is not None and pred_after_val is not None:
            vectors.actual_after.append(after_val)
            vectors.predicted_after.append(pred_after_val)
    return vectors


def _parse_prediction_row(
    row: Mapping[str, object],
    best_k: int,
    k_token: str,
) -> tuple[float | None, float | None, float | None, float | None] | None:
    """Extract floats describing a single prediction row."""
    before_val = _coerce_float(row.get("before_index"))
    after_val = _coerce_float(row.get("after_index"))
    preds = row.get("predictions_by_k") or {}
    pred_changes = row.get("predicted_change_by_k") or {}
    pred_after_raw = preds.get(k_token, preds.get(best_k))
    pred_after_val = _coerce_float(pred_after_raw)
    pred_change_raw = pred_changes.get(k_token)
    pred_change_val = _coerce_float(pred_change_raw)
    if pred_change_val is None and pred_after_val is not None and before_val is not None:
        pred_change_val = pred_after_val - before_val
    if (
        before_val is None
        and after_val is None
        and pred_after_val is None
        and pred_change_val is None
    ):
        return None
    return before_val, after_val, pred_after_val, pred_change_val


def _coerce_float(candidate: object) -> float | None:
    """Return a float when ``candidate`` is numeric; otherwise ``None``."""
    try:
        if candidate is None:
            return None
        return float(candidate)
    except (TypeError, ValueError):
        return None


def _render_prediction_plots(
    feature_dir: Path,
    study_key: str,
    vectors: _PredictionVectors,
) -> None:
    """Emit heatmaps for opinion change and post-study predictions."""
    if vectors.actual_changes and vectors.predicted_changes:
        _plot_change_heatmap(
            actual_changes=vectors.actual_changes,
            predicted_changes=vectors.predicted_changes,
            output_path=feature_dir / f"change_heatmap_{study_key}.png",
        )
    if vectors.actual_after and vectors.predicted_after:
        _plot_post_prediction_heatmap(
            actual_after=vectors.actual_after,
            predicted_after=vectors.predicted_after,
            output_path=feature_dir / f"post_heatmap_{study_key}.png",
        )


__all__ = ["_OpinionPortfolioStats", "OpinionReportOptions", "_build_opinion_report"]
