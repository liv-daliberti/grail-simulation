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

"""High-level entry points for orchestrating XGBoost report generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ..context import (
    OpinionStudySelection,
    OpinionSweepOutcome,
    StudySelection,
    SweepOutcome,
)
from .catalog import _write_catalog_report
from .hyperparameter import _write_hyperparameter_report
from .next_video import _write_next_video_report
from .opinion import _write_opinion_report
from .shared import _write_disabled_report
from .features import _write_feature_report


@dataclass(frozen=True)
class SweepReportData:
    """Bundle describing sweep outcomes, selections, and evaluation metrics."""

    outcomes: Sequence[SweepOutcome] = ()
    selections: Mapping[str, StudySelection] = field(default_factory=dict)
    final_metrics: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    loso_metrics: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


@dataclass(frozen=True)
class OpinionReportData:
    """Bundle describing opinion sweep data and aggregated metrics."""

    metrics: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    outcomes: Sequence[OpinionSweepOutcome] = ()
    selections: Mapping[str, OpinionStudySelection] = field(default_factory=dict)
    title: str = "XGBoost Opinion Regression"
    description_lines: Sequence[str] | None = None


@dataclass(frozen=True)
class ReportSections:
    """Describe which report sections should be generated."""

    include_next_video: bool = True
    opinion: OpinionReportData | None = None
    opinion_from_next: OpinionReportData | None = None


def _write_reports(
    *,
    reports_dir: Path,
    sweeps: SweepReportData,
    allow_incomplete: bool,
    sections: ReportSections = ReportSections(),
) -> None:
    """
    Write the full report bundle capturing sweep and evaluation artefacts.

    :param reports_dir: Base directory receiving generated Markdown files.
    :type reports_dir: Path
    :param sweeps: Sweep outcomes, selections, and evaluation metrics.
    :type sweeps: SweepReportData
    :param allow_incomplete: Flag controlling whether missing artefacts are tolerated.
    :type allow_incomplete: bool
    :param include_next_video: Flag enabling next-video sections.
    :type include_next_video: bool
    :param opinion: Optional bundle describing opinion sweeps and metrics.
    :type opinion: OpinionReportData | None
    """

    reports_dir.mkdir(parents=True, exist_ok=True)

    include_next_video = sections.include_next_video
    opinion = sections.opinion
    opinion_from_next = sections.opinion_from_next
    include_opinion = opinion is not None
    legacy_hyper_file = reports_dir / "hyperparameter_tuning.md"
    legacy_next_file = reports_dir / "next_video.md"
    if legacy_hyper_file.exists():
        legacy_hyper_file.unlink()
    if legacy_next_file.exists():
        legacy_next_file.unlink()

    _write_catalog_report(
        reports_dir,
        include_next_video=include_next_video,
        include_opinion=include_opinion,
    )
    if include_next_video or include_opinion:
        _write_hyperparameter_report(
            reports_dir / "hyperparameter_tuning",
            sweeps,
            allow_incomplete=allow_incomplete,
            include_next_video=include_next_video,
            opinion=opinion if include_opinion else None,
        )
    else:
        _write_disabled_report(
            reports_dir / "hyperparameter_tuning",
            "Hyper-parameter Tuning",
            "Sweep stages were disabled for this run.",
        )
    if include_next_video:
        _write_next_video_report(
            reports_dir / "next_video",
            sweeps.final_metrics,
            sweeps.selections,
            allow_incomplete=allow_incomplete,
            loso_metrics=sweeps.loso_metrics,
        )
    else:
        _write_disabled_report(
            reports_dir / "next_video",
            "Next-Video Evaluation",
            "Next-video evaluation was skipped because the task was not selected.",
        )
    if include_opinion:
        _write_opinion_report(
            reports_dir / "opinion",
            opinion.metrics,
            allow_incomplete=allow_incomplete,
            title=opinion.title,
            description_lines=opinion.description_lines,
        )
    else:
        _write_disabled_report(
            reports_dir / "opinion",
            "Opinion Regression",
            "Opinion sweeps were disabled for this run.",
        )
    if opinion_from_next is not None:
        _write_opinion_report(
            reports_dir / "opinion_from_next",
            opinion_from_next.metrics,
            allow_incomplete=allow_incomplete,
            title=opinion_from_next.title,
            description_lines=opinion_from_next.description_lines,
        )
    _write_feature_report(
        reports_dir / "additional_features",
        sweeps,
        include_next_video=include_next_video,
        opinion=opinion if include_opinion else None,
    )


__all__ = ["OpinionReportData", "SweepReportData", "ReportSections", "_write_reports"]
