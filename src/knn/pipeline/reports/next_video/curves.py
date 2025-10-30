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

"""Plot generation helpers for next-video learning-curve sections."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from common.reports.utils import extract_curve_sections, extract_numeric_series

from ...context import StudySpec
from ..shared import LOGGER
from .helpers import _ordered_feature_spaces

try:  # pragma: no cover - optional dependency
    from common.visualization.matplotlib import plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]


def _extract_curve_series(curve_block: Mapping[str, object]) -> Tuple[List[int], List[float]]:
    """
    Return sorted k/accuracy pairs extracted from ``curve_block``.

    :param curve_block: Markdown block describing a single KNN performance curve.
    :returns: Sorted k/accuracy pairs extracted from ``curve_block``.
    """
    accuracy_map = curve_block.get("accuracy_by_k")
    if not isinstance(accuracy_map, Mapping):
        return ([], [])
    return extract_numeric_series(accuracy_map)


def _plot_knn_curve_bundle(
    *,
    base_dir: Path,
    feature_space: str,
    study: StudySpec,
    metrics: Mapping[str, object],
) -> Optional[str]:
    """
    Save a train/validation accuracy curve plot for ``study`` when possible.

    :param base_dir: Base directory that contains task-specific output subdirectories.
    :param feature_space: Feature space identifier such as ``tfidf`` or ``word2vec``.
    :param study: Study specification for the item currently being processed.
    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :returns: Relative path to the generated curve plot.
    """
    if plt is None:  # pragma: no cover - optional dependency
        return None
    sections = extract_curve_sections(metrics.get("curve_metrics"))
    if sections is None:
        return None
    eval_curve, train_curve = sections
    eval_x, eval_y = _extract_curve_series(eval_curve)
    if not eval_x:
        return None
    train_x, train_y = (
        _extract_curve_series(train_curve) if train_curve is not None else ([], [])
    )

    curves_dir = base_dir / "curves" / feature_space
    curves_dir.mkdir(parents=True, exist_ok=True)
    plot_path = curves_dir / f"{study.study_slug}.png"

    fig, axis = plt.subplots(figsize=(6, 3.5))  # type: ignore[attr-defined]
    axis.plot(eval_x, eval_y, marker="o", label="validation")
    if train_x and train_y:
        axis.plot(train_x, train_y, marker="o", linestyle="--", label="training")
    axis.set_title(f"{study.label} – {feature_space.upper()}")
    axis.set_xlabel("k")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(eval_x)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)  # type: ignore[attr-defined]
    plt.close(fig)  # type: ignore[attr-defined]
    try:
        return plot_path.relative_to(base_dir).as_posix()
    except ValueError:
        return plot_path.as_posix()


def _next_video_curve_sections(
    *,
    output_dir: Path,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> list[str]:
    """
    Render Markdown sections embedding train/validation accuracy curves.

    :param output_dir: Directory where the rendered report should be written.
    :param metrics_by_feature: Nested mapping of metrics grouped by feature space and study.
    :param studies: Sequence of study specifications targeted by the workflow.
    :returns: Markdown sections that document next-video learning curves.
    """
    if plt is None:  # pragma: no cover - optional dependency
        LOGGER.warning(
            "Matplotlib not available; skipping KNN accuracy curve plots. "
            "Install the visualization extras (pip install -e '.[visualization]') "
            "and rerun the finalize stage to refresh them.",
        )
        return [
            "_Accuracy curves skipped because matplotlib is not installed. "
            "Install the visualization extras and rerun "
            "`python -m knn.pipeline --stage finalize` to generate plots._",
            "",
        ]

    sections: list[str] = []
    ordered_spaces = _ordered_feature_spaces((), metrics_by_feature)

    for feature_space in ordered_spaces:
        feature_metrics = metrics_by_feature.get(feature_space, {})
        if not feature_metrics:
            continue
        image_lines: list[str] = []
        for study in studies:
            study_metrics = feature_metrics.get(study.key)
            if not study_metrics:
                continue
            rel_path = _plot_knn_curve_bundle(
                base_dir=output_dir,
                feature_space=feature_space,
                study=study,
                metrics=study_metrics,
            )
            if rel_path:
                image_lines.extend(
                    [
                        f"### {study.label} ({feature_space.upper()})",
                        "",
                        f"![Accuracy curve]({rel_path})",
                        "",
                    ]
                )
        sections.extend(image_lines)
    return sections


__all__ = [
    "_extract_curve_series",
    "_next_video_curve_sections",
    "_plot_knn_curve_bundle",
]
