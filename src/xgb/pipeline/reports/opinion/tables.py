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

"""Markdown table construction for opinion metrics."""

from __future__ import annotations

from typing import List, Mapping

from common.pipeline.formatters import (
    format_count as _format_count,
    format_delta as _format_delta,
    format_optional_float as _format_optional_float,
)

from .metrics import _difference
from .summaries import _extract_opinion_summary


def _opinion_table_header() -> List[str]:
    """
    Return the Markdown header rows for the opinion metrics table.

    :returns: Markdown table header lines with alignment tokens.
    """

    columns = [
        "Study",
        "Participants",
        "Accuracy ↑",
        "Baseline ↑",
        "Δ Accuracy ↑",
        "MAE ↓",
        "Δ vs baseline ↓",
        "RMSE ↓",
        "R² ↑",
        "MAE (change) ↓",
        "RMSE (change) ↓",
        "Δ RMSE (change) ↓",
        "Calib slope",
        "Calib intercept",
        "ECE ↓",
        "Δ ECE ↓",
        "KL div ↓",
        "Δ KL ↓",
        "Baseline MAE ↓",
    ]
    header_line = f"| {' | '.join(columns)} |"
    align_tokens = ["---"] + ["---:"] * (len(columns) - 1)
    align_line = f"| {' | '.join(align_tokens)} |"
    return [header_line, align_line]


def _opinion_table_rows(metrics: Mapping[str, Mapping[str, object]]) -> List[str]:
    """
    Build Markdown table rows for opinion metrics.

    :param metrics: Mapping from study identifiers to metrics payloads.
    :returns: Markdown rows ready for inclusion in the opinion report.
    """

    rows: List[str] = []
    for study_key, payload in sorted(metrics.items()):
        summary = _extract_opinion_summary(payload)
        study_label = summary.label or study_key
        row_segments = [
            study_label,
            _format_count(summary.participants),
            _format_optional_float(summary.accuracy_after),
            _format_optional_float(summary.baseline_accuracy),
            _format_delta(summary.accuracy_delta),
            _format_optional_float(summary.mae_after),
            _format_delta(summary.mae_delta),
            _format_optional_float(summary.rmse_after),
            _format_optional_float(summary.r2_after),
            _format_optional_float(summary.mae_change),
            _format_optional_float(summary.rmse_change),
            _format_delta(_difference(summary.baseline_rmse_change, summary.rmse_change)),
            _format_optional_float(summary.calibration_slope),
            _format_optional_float(summary.calibration_intercept),
            _format_optional_float(summary.calibration_ece),
            _format_delta(
                _difference(
                    summary.baseline_calibration_ece,
                    summary.calibration_ece,
                )
            ),
            _format_optional_float(summary.kl_divergence_change),
            _format_delta(
                _difference(
                    summary.baseline_kl_divergence_change,
                    summary.kl_divergence_change,
                )
            ),
            _format_optional_float(summary.baseline_mae),
        ]
        rows.append(f"| {' | '.join(row_segments)} |")
    return rows


__all__ = [
    "_opinion_table_header",
    "_opinion_table_rows",
]
