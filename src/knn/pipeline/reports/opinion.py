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

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from ..context import StudySpec
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
    """

    allow_incomplete: bool = False
    title: str = "# KNN Opinion Shift Study"
    description_lines: Optional[Sequence[str]] = None
    metrics_line: Optional[str] = None


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
    lines.extend(_opinion_feature_sections(metrics, studies))
    lines.extend(_opinion_heatmap_section(output_path, metrics))
    lines.extend(_knn_opinion_cross_study_diagnostics(metrics, studies))
    lines.extend(_opinion_takeaways(metrics, studies))
    output_path.write_text("\n".join(lines), encoding="utf-8")
    # Emit CSV dump combining all feature spaces and studies
    _write_knn_opinion_csv(output_path.parent, metrics, studies)


__all__ = ["_OpinionPortfolioStats", "OpinionReportOptions", "_build_opinion_report"]
