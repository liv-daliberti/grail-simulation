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

"""Public entry points for the KNN pipeline report builders."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..context import ReportBundle
from .catalog import _build_catalog_report
from .hyperparameter import (
    HyperparameterCommonContext,
    HyperparameterReportConfig,
    NextVideoSectionConfig,
    OpinionSectionConfig,
    _build_hyperparameter_report,
)
from .next_video import NextVideoReportInputs, _build_next_video_report
from .opinion import OpinionReportOptions, _build_opinion_report
from .features import build_feature_report
from .shared import parse_k_sweep

def _derive_k_sweep(bundle: ReportBundle) -> tuple[int, ...]:
    """Prefer extracting k-sweep from actual sweep outcomes; fall back to config.

    This avoids drift when the report stage runs with different environment
    defaults than the sweeps stage (e.g., training script overrides).
    """
    values: set[int] = set()
    outcomes = getattr(bundle, "sweep_outcomes", ()) or ()
    if isinstance(outcomes, str) or not isinstance(outcomes, Iterable):
        outcomes = ()
    for outcome in outcomes:
        metrics = getattr(outcome, "metrics", {}) or {}
        hparams = metrics.get("knn_hparams", {}) if isinstance(metrics, dict) else {}
        k_list = hparams.get("k_sweep") if isinstance(hparams, dict) else None
        if isinstance(k_list, (list, tuple)):
            for item in k_list:
                parsed = None
                try:
                    parsed = int(item)
                except (TypeError, ValueError):
                    parsed = None
                if parsed is not None:
                    values.add(parsed)
    if values:
        return tuple(sorted(values))
    return parse_k_sweep(bundle.k_sweep)

__all__ = ["generate_reports"]


def generate_reports(repo_root: Path, report_bundle: ReportBundle) -> None:
    """
    Write refreshed Markdown reports under ``reports/knn``.

    :param repo_root: Repository root directory used for path resolution.
    :type repo_root: Path
    :param report_bundle: Aggregated data structure containing everything needed to emit reports.
    :type report_bundle: ~knn.pipeline.context.ReportBundle
    """
    reports_root = repo_root / "reports" / "knn"
    feature_spaces = report_bundle.feature_spaces
    k_sweep_values = _derive_k_sweep(report_bundle)
    allow_incomplete = report_bundle.allow_incomplete

    _build_catalog_report(
        reports_root,
        include_next_video=report_bundle.include_next_video,
        include_opinion=report_bundle.include_opinion,
    )

    if report_bundle.include_next_video or report_bundle.include_opinion:
        common_context = HyperparameterCommonContext(
            studies=report_bundle.studies,
            feature_spaces=feature_spaces,
            k_sweep=k_sweep_values,
            sentence_model=report_bundle.sentence_model,
        )
        next_video_config = (
            NextVideoSectionConfig(
                selections=report_bundle.selections,
                sweep_outcomes=report_bundle.sweep_outcomes,
            )
            if report_bundle.include_next_video
            else None
        )
        opinion_config = (
            OpinionSectionConfig(
                selections=report_bundle.opinion_selections,
                sweep_outcomes=report_bundle.opinion_sweep_outcomes,
            )
            if report_bundle.include_opinion
            else None
        )
        _build_hyperparameter_report(
            HyperparameterReportConfig(
                output_dir=reports_root / "hyperparameter_tuning",
                common=common_context,
                allow_incomplete=allow_incomplete,
                next_video=next_video_config,
                opinion=opinion_config,
            )
        )
    if report_bundle.include_next_video:
        _build_next_video_report(
            NextVideoReportInputs(
                output_dir=reports_root / "next_video",
                metrics_by_feature=report_bundle.metrics_by_feature,
                studies=report_bundle.studies,
                feature_spaces=feature_spaces,
                loso_metrics=report_bundle.loso_metrics,
                allow_incomplete=allow_incomplete,
                xgb_next_video_dir=(repo_root / "models" / "xgb" / "next_video"),
            )
        )

    if report_bundle.include_opinion:
        _build_opinion_report(
            output_path=reports_root / "opinion" / "README.md",
            metrics=report_bundle.opinion_metrics,
            studies=report_bundle.studies,
            options=OpinionReportOptions(
                allow_incomplete=allow_incomplete,
            ),
        )
    if report_bundle.include_opinion_from_next:
        _build_opinion_report(
            output_path=reports_root / "opinion_from_next" / "README.md",
            metrics=report_bundle.opinion_from_next_metrics,
            studies=report_bundle.studies,
            options=OpinionReportOptions(
                allow_incomplete=allow_incomplete,
                title="# KNN Opinion Shift Study (Next-Video Config)",
                description_lines=[
                    "This section reuses the selected next-video recommendation configuration "
                    "to estimate post-study opinion change.",
                ],
            ),
        )
    build_feature_report(repo_root, report_bundle)
