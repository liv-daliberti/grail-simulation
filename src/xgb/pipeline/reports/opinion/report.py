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

"""Report assembly for XGBoost opinion regression outputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from common.opinion.metrics import (
    OPINION_CSV_BASE_FIELDS,
    build_opinion_csv_base_row,
)
from common.pipeline.io import write_markdown_lines
from common.reports.utils import start_markdown_report

from .curves import _opinion_curve_lines
from .observations import (
    _opinion_cross_study_diagnostics,
    _opinion_observations,
)
from .prediction_plots import (
    _opinion_feature_plot_section,
    _regenerate_opinion_feature_plots,
)
from .summaries import _dataset_and_split, _extract_opinion_summary
from .tables import _opinion_table_header, _opinion_table_rows
from ..plots import plt


@dataclass(frozen=True)
class OpinionReportOptions:
    """
    Configuration bundle for generating opinion regression reports.

    :ivar allow_incomplete: Whether warnings for missing metrics should be emitted.
    :ivar title: Markdown title applied to the report.
    :ivar description_lines: Optional explanatory copy inserted after the title.
    :ivar predictions_root: Directory containing cached prediction artefacts.
    :ivar regenerate_plots: Whether supplementary plots should be rebuilt.
    """

    allow_incomplete: bool
    title: str = "XGBoost Opinion Regression"
    description_lines: Sequence[str] | None = None
    predictions_root: Path | None = None
    regenerate_plots: bool = True


def _write_opinion_report(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
    options: OpinionReportOptions,
) -> None:
    """
    Create the opinion regression summary document.

    :param directory: Destination directory for the Markdown report.
    :param metrics: Nested mapping of opinion metrics indexed by study.
    :param options: Configuration values controlling report generation.
    :returns: ``None``.
    """

    path, lines = start_markdown_report(directory, title=options.title)
    if not metrics:
        lines.append("No opinion runs were produced during this pipeline invocation.")
        if options.allow_incomplete:
            lines.append(
                "Rerun the pipeline with `--stage finalize` to populate this section once "
                "opinion metrics are available."
            )
        lines.append("")
        write_markdown_lines(path, lines)
        return
    if options.regenerate_plots:
        _regenerate_opinion_feature_plots(
            report_dir=directory,
            metrics=metrics,
            predictions_root=options.predictions_root,
        )
    description_lines = options.description_lines
    if description_lines is None:
        description_lines = [
            "This summary captures the opinion-regression baselines trained with XGBoost "
            "for the selected participant studies."
        ]
    if description_lines:
        lines.extend(description_lines)
        if description_lines[-1].strip():
            lines.append("")
    dataset_name, split_name = _dataset_and_split(metrics)
    lines.extend(
        [
            f"- Dataset: `{dataset_name}`",
            f"- Split: {split_name}",
            (
                "- Metrics track MAE, RMSE, R², directional accuracy, MAE(change), "
                "RMSE(change), calibration slope/intercept, calibration ECE, and KL "
                "divergence versus the no-change baseline."
            ),
            "- Δ columns capture improvements relative to that baseline when available.",
            "",
        ]
    )
    lines.extend(_opinion_table_header())
    lines.extend(_opinion_table_rows(metrics))
    lines.append("")
    curve_lines = _opinion_curve_lines(directory, metrics)
    if curve_lines:
        lines.extend(curve_lines)
    elif plt is None:  # pragma: no cover - optional dependency
        lines.extend(
            [
                "## Training Curves",
                "",
                (
                    "Matplotlib is unavailable in this environment, so training curves "
                    "were not rendered."
                ),
                "",
            ]
        )
    lines.extend(_opinion_feature_plot_section(directory))
    lines.extend(_opinion_cross_study_diagnostics(metrics))
    lines.extend(_opinion_observations(metrics))
    write_markdown_lines(path, lines)
    # Emit CSV dump for downstream analysis
    _write_opinion_csv(directory, metrics)


def _write_opinion_csv(directory: Path, metrics: Mapping[str, Mapping[str, object]]) -> None:
    """Write per-study opinion metrics to ``opinion_metrics.csv``.

    :param directory: Opinion report directory receiving the CSV export.
    :param metrics: Mapping from study identifiers to metrics payloads.
    :returns: ``None``. Serialises the metrics to the CSV file when available.
    """

    if not metrics:
        return
    out_path = directory / "opinion_metrics.csv"
    fieldnames = list(OPINION_CSV_BASE_FIELDS)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for study_key in sorted(metrics.keys()):
            summary = _extract_opinion_summary(metrics[study_key])
            row = build_opinion_csv_base_row(
                summary, study_label=(summary.label or study_key)
            )
            writer.writerow(row)


__all__ = [
    "OpinionReportOptions",
    "_write_opinion_csv",
    "_write_opinion_report",
]
