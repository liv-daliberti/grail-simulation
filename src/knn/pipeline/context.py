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

"""Aggregated context and types for the KNN pipeline.

This module re-exports the public context types used across the pipeline while
keeping implementations split into focused modules (pipeline, evaluation,
config, sweeps, tasks, and report bundles). This avoids duplication and keeps
attribute counts small for linting.
"""

from __future__ import annotations

from common.pipeline.types import StudySpec
from .context_config import SweepConfig
from .context_evaluation import EvaluationContext, EvaluationOutputs, EvaluationWord2VecPaths
from .context_pipeline import PipelineContext
from .context_reports import (
    ReportBundle,
    _PredictionRoots as PredictionRoots,
    _PresentationFlags as PresentationFlags,
    _ReportMetrics as ReportMetrics,
    _ReportOutcomes as ReportOutcomes,
    _ReportPresentation as ReportPresentation,
    _ReportSelections as ReportSelections,
)
from .context_summaries import MetricSummary, OpinionSummary
from .context_sweeps import (
    OpinionStudySelection,
    OpinionSweepOutcome,
    StudySelection,
    SweepOutcome,
)
from .context_tasks import OpinionSweepTask, SweepTask, SweepTaskContext

# Public alias for constructing grouped opinion summary inputs
OpinionSummaryInputs = OpinionSummary.Inputs

# Ensure canonical module for Sphinx: present re-exported classes as
# originating from this aggregator module without breaking dataclasses.
for _t in (
    StudySpec,
    ReportBundle,
    ReportSelections,
    ReportOutcomes,
    ReportMetrics,
    ReportPresentation,
    PresentationFlags,
    PredictionRoots,
    StudySelection,
    SweepOutcome,
    OpinionStudySelection,
    OpinionSweepOutcome,
    EvaluationContext,
    EvaluationOutputs,
    EvaluationWord2VecPaths,
    PipelineContext,
    SweepTask,
    SweepTaskContext,
    OpinionSweepTask,
    SweepConfig,
):
    try:  # pragma: no cover - defensive: some may be functions/aliases
        _t.__module__ = __name__
    except (TypeError, AttributeError):
        # Some built-in/extension types or aliases may not allow assignment
        # to __module__; safely skip those.
        pass

__all__ = [
    "MetricSummary",
    "EvaluationOutputs",
    "EvaluationWord2VecPaths",
    "OpinionStudySelection",
    "OpinionSummary",
    "OpinionSummaryInputs",
    "OpinionSweepOutcome",
    "OpinionSweepTask",
    "PipelineContext",
    "ReportBundle",
    "ReportSelections",
    "ReportOutcomes",
    "ReportMetrics",
    "ReportPresentation",
    "PresentationFlags",
    "PredictionRoots",
    "StudySelection",
    "StudySpec",
    "SweepConfig",
    "SweepOutcome",
    "SweepTask",
    "SweepTaskContext",
    "EvaluationContext",
]
