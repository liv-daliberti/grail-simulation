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

"""Data structures describing the Grail Simulation XGBoost pipeline.

Defines sweep configuration objects, execution contexts, and selection
results exchanged between the sweep, evaluation, and reporting stages.

This package splits the original monolithic module into focused submodules
while preserving the public API at ``xgb.pipeline.context``.
"""

from __future__ import annotations

# Re-export types from submodules to maintain backwards-compatible imports

from .sweep import (
    FinalEvalContext,
    StudySelection,
    SweepConfig,
    SweepOutcome,
    SweepRunContext,
    SweepTask,
)
from .opinion import (
    OpinionStageConfig,
    OpinionStudySelection,
    OpinionSweepOutcome,
    OpinionSweepRunContext,
    OpinionSweepTask,
    OpinionDataSettings,
    OpinionVectorizerSettings,
    OpinionXgbSettings,
)
from .metrics_next_video import (
    NextVideoMetricSummary,
    _NextVideoCore,
    _NextVideoMeta,
)
from .opinion_summary import (
    OpinionSummary,
    _OpinionAfter,
    _OpinionBaseline,
    _OpinionCalibration,
    _OpinionDeltas,
    _OpinionMeta,
)

# Also re-export StudySpec used pervasively alongside these types
from common.pipeline.types import StudySpec

__all__ = [
    "FinalEvalContext",
    "NextVideoMetricSummary",
    "OpinionStageConfig",
    "OpinionSummary",
    "OpinionStudySelection",
    "OpinionSweepOutcome",
    "OpinionSweepTask",
    "OpinionSweepRunContext",
    "StudySelection",
    "StudySpec",
    "OpinionDataSettings",
    "OpinionVectorizerSettings",
    "OpinionXgbSettings",
    "SweepConfig",
    "SweepOutcome",
    "SweepRunContext",
    "SweepTask",
]

