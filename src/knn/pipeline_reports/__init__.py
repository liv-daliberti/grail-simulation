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

from pathlib import Path

from ..pipeline_context import ReportBundle
from .catalog import _build_catalog_report
from .hyperparameter import HyperparameterReportConfig, _build_hyperparameter_report
from .next_video import NextVideoReportInputs, _build_next_video_report
from .opinion import _build_opinion_report
from .shared import parse_k_sweep

__all__ = ["generate_reports"]


def generate_reports(repo_root: Path, report_bundle: ReportBundle) -> None:
    """
    Write refreshed Markdown reports under ``reports/knn``.

    :param repo_root: Repository root directory used for path resolution.
    :type repo_root: Path
    :param report_bundle: Aggregated data structure containing everything needed to emit reports.
    :type report_bundle: ReportBundle
    """
    reports_root = repo_root / "reports" / "knn"
    feature_spaces = report_bundle.feature_spaces
    k_sweep_values = parse_k_sweep(report_bundle.k_sweep)
    allow_incomplete = report_bundle.allow_incomplete

    _build_catalog_report(
        reports_root,
        include_next_video=report_bundle.include_next_video,
        include_opinion=report_bundle.include_opinion,
    )

    if report_bundle.include_next_video or report_bundle.include_opinion:
        _build_hyperparameter_report(
            HyperparameterReportConfig(
                output_dir=reports_root / "hyperparameter_tuning",
                selections=(
                    report_bundle.selections if report_bundle.include_next_video else {}
                ),
                sweep_outcomes=(
                    report_bundle.sweep_outcomes
                    if report_bundle.include_next_video
                    else ()
                ),
                studies=report_bundle.studies,
                k_sweep=k_sweep_values,
                feature_spaces=feature_spaces,
                sentence_model=report_bundle.sentence_model,
                opinion_selections=(
                    report_bundle.opinion_selections
                    if report_bundle.include_opinion
                    else {}
                ),
                opinion_sweep_outcomes=(
                    report_bundle.opinion_sweep_outcomes
                    if report_bundle.include_opinion
                    else ()
                ),
                allow_incomplete=allow_incomplete,
                include_next_video=report_bundle.include_next_video,
                include_opinion=report_bundle.include_opinion,
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
            )
        )

    if report_bundle.include_opinion:
        _build_opinion_report(
            output_path=reports_root / "opinion" / "README.md",
            metrics=report_bundle.opinion_metrics,
            studies=report_bundle.studies,
            allow_incomplete=allow_incomplete,
        )
