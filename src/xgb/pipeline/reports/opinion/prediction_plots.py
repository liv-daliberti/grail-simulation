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

"""Prediction-driven plots for the opinion regression report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping

from common.reports.utils import append_image_section

from ..plots import (
    _plot_opinion_change_heatmap,
    _plot_opinion_error_histogram,
    _plot_opinion_post_heatmap,
)
from ..shared import LOGGER


@dataclass
class _OpinionPredictionVectors:
    """
    Collect opinion prediction sequences extracted from cached inference rows.

    :ivar actual_after: Observed post-study opinion indices.
    :ivar predicted_after: Predicted post-study opinion indices.
    :ivar actual_changes: Observed opinion deltas (post - pre).
    :ivar predicted_changes: Predicted opinion deltas.
    :ivar errors: Absolute prediction errors for the post-study index.
    """

    actual_after: List[float] = field(default_factory=list)
    predicted_after: List[float] = field(default_factory=list)
    actual_changes: List[float] = field(default_factory=list)
    predicted_changes: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)

    def append_sample(
        self,
        *,
        before: float,
        after: float,
        predicted_after: float,
        predicted_change: float,
    ) -> None:
        """
        Record a single participant snapshot.

        :param before: Baseline opinion index prior to treatment.
        :param after: Observed post-study opinion index.
        :param predicted_after: Model-predicted post-study opinion index.
        :param predicted_change: Model-predicted opinion delta.
        :returns: ``None``. Appends the sample to all relevant sequences.
        """

        self.actual_after.append(after)
        self.predicted_after.append(predicted_after)
        self.errors.append(abs(predicted_after - after))
        self.actual_changes.append(after - before)
        self.predicted_changes.append(predicted_change)

    def has_post_indices(self) -> bool:
        """Return ``True`` when post-study predictions are available.

        :returns: ``True`` when both actual and predicted post indices exist.
        """

        return bool(self.actual_after and self.predicted_after)

    def has_change_series(self) -> bool:
        """Return ``True`` when change deltas are available.

        :returns: ``True`` when both actual and predicted change sequences exist.
        """

        return bool(self.actual_changes and self.predicted_changes)

    def has_errors(self) -> bool:
        """Return ``True`` when prediction errors were recorded.

        :returns: ``True`` when any error values have been accumulated.
        """

        return bool(self.errors)

    def has_observations(self) -> bool:
        """Return ``True`` when any accumulated sequences are non-empty.

        :returns: ``True`` when at least one sequence contains data.
        """

        return bool(self.actual_after or self.actual_changes or self.errors)


def _collect_opinion_prediction_vectors(
    *,
    predictions_path: Path,
    feature_space: str,
    study_key: str,
) -> _OpinionPredictionVectors | None:
    """
    Load cached prediction rows and extract series needed for diagnostic plots.

    :param predictions_path: Filesystem path to the cached prediction JSONL file.
    :param feature_space: Feature-space identifier (e.g. ``tfidf``).
    :param study_key: Participant study identifier.
    :returns: Populated :class:`_OpinionPredictionVectors` or ``None`` when empty.
    """

    def _parse_prediction_row(
        payload: Mapping[str, object],
    ) -> tuple[float, float, float, float] | None:
        """Extract numeric opinion values from a cached prediction row.

        :param payload: JSON-decoded prediction record.
        :returns: Tuple ``(before, after, predicted_after, predicted_change)``
            or ``None`` when parsing fails.
        """
        before = payload.get("before")
        after_value = payload.get("after")
        pred_after = payload.get("prediction")
        if before is None or after_value is None or pred_after is None:
            return None
        try:
            before_f = float(before)
            after_f = float(after_value)
            pred_after_f = float(pred_after)
        except (TypeError, ValueError):
            return None
        pred_change = payload.get("prediction_change")
        if pred_change is None:
            pred_change_f = pred_after_f - before_f
        else:
            try:
                pred_change_f = float(pred_change)
            except (TypeError, ValueError):
                pred_change_f = pred_after_f - before_f
        return (before_f, after_f, pred_after_f, pred_change_f)

    vectors = _OpinionPredictionVectors()
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = line.strip()
            if not record:
                continue
            try:
                payload = json.loads(record)
            except json.JSONDecodeError:
                LOGGER.debug(
                    "[XGB][OPINION] Skipping malformed prediction row for %s/%s.",
                    feature_space,
                    study_key,
                )
                continue
            parsed = _parse_prediction_row(payload)
            if parsed is None:
                continue
            before_f, after_f, pred_after_f, pred_change_f = parsed
            vectors.append_sample(
                before=before_f,
                after=after_f,
                predicted_after=pred_after_f,
                predicted_change=pred_change_f,
            )

    if not vectors.has_observations():
        return None

    return vectors


def _render_opinion_prediction_plots(
    *,
    feature_dir: Path,
    study_key: str,
    vectors: _OpinionPredictionVectors,
) -> None:
    """
    Dispatch plotting helpers for the supplied prediction series.

    :param feature_dir: Directory receiving generated PNG artefacts.
    :param study_key: Study identifier used when naming outputs.
    :param vectors: Prediction sequences extracted from cached outputs.
    :returns: ``None``.
    """

    if vectors.has_post_indices():
        _plot_opinion_post_heatmap(
            actual_after=vectors.actual_after,
            predicted_after=vectors.predicted_after,
            output_path=feature_dir / f"post_heatmap_{study_key}.png",
        )
    if vectors.has_change_series():
        _plot_opinion_change_heatmap(
            actual_changes=vectors.actual_changes,
            predicted_changes=vectors.predicted_changes,
            output_path=feature_dir / f"change_heatmap_{study_key}.png",
        )
    if vectors.has_errors():
        _plot_opinion_error_histogram(
            errors=vectors.errors,
            output_path=feature_dir / f"error_histogram_{study_key}.png",
        )


def _regenerate_opinion_feature_plots(
    *,
    report_dir: Path,
    metrics: Mapping[str, Mapping[str, object]],
    predictions_root: Path | None,
) -> None:
    """
    Rebuild supplementary opinion plots from cached predictions.

    :param report_dir: Output directory for the opinion report bundle.
    :param metrics: Opinion metrics keyed by feature space and study.
    :param predictions_root: Directory containing cached prediction records.
    :returns: ``None``.
    """
    if not metrics or predictions_root is None:
        return
    for study_key, payload in metrics.items():
        feature_space = str(payload.get("feature_space") or "").lower()
        if not feature_space:
            LOGGER.debug(
                "[XGB][OPINION] Missing feature_space for study=%s; skipping plot regeneration.",
                study_key,
            )
            continue
        feature_dir = report_dir.parent / feature_space / "opinion"
        feature_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = (
            predictions_root
            / feature_space
            / study_key
            / f"opinion_xgb_{study_key}_validation_predictions.jsonl"
        )
        if not predictions_path.exists():
            LOGGER.debug(
                "[XGB][OPINION] Predictions missing for %s/%s at %s; skipping plots.",
                feature_space,
                study_key,
                predictions_path,
            )
            continue

        vectors = _collect_opinion_prediction_vectors(
            predictions_path=predictions_path,
            feature_space=feature_space,
            study_key=study_key,
        )
        if vectors is None:
            continue

        _render_opinion_prediction_plots(
            feature_dir=feature_dir,
            study_key=study_key,
            vectors=vectors,
        )


def _opinion_feature_plot_section(directory: Path) -> List[str]:
    """
    Embed static PNG assets produced outside the primary report.

    :param directory: Opinion report directory whose siblings may contain plots.
    :returns: Markdown lines referencing discovered static PNG artefacts.
    """

    primary_base = directory.parent
    candidate_bases = [primary_base]
    secondary_base = primary_base.parent
    if secondary_base != primary_base:
        candidate_bases.append(secondary_base)
    seen_paths: set[Path] = set()
    sections: List[str] = []
    for feature_space in ("tfidf", "word2vec", "sentence_transformer"):
        images: List[Path] = []
        for base_dir in candidate_bases:
            feature_dir = base_dir / feature_space / "opinion"
            if not feature_dir.exists():
                continue
            images.extend(sorted(feature_dir.glob("*.png")))
        unique_images: List[Path] = []
        for image in images:
            try:
                canonical = image.resolve()
            except FileNotFoundError:
                # Skip files that vanished since discovery.
                continue
            if canonical in seen_paths:
                continue
            seen_paths.add(canonical)
            unique_images.append(image)
        if not unique_images:
            continue
        sections.append(f"## {feature_space.upper()} Opinion Plots")
        sections.append("")
        for image in unique_images:
            append_image_section(
                sections,
                image=image,
                relative_root=directory.parent,
            )
    return sections


__all__ = [
    "_OpinionPredictionVectors",
    "_collect_opinion_prediction_vectors",
    "_opinion_feature_plot_section",
    "_regenerate_opinion_feature_plots",
    "_render_opinion_prediction_plots",
]
