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

"""Aggregated exports for opinion study helpers."""

from .metrics import (
    OpinionMetricsView,
    compute_opinion_metrics,
    summarise_opinion_metrics,
)
from .prompts import format_opinion_user_prompt
from .models import (
    DEFAULT_SPECS,
    OpinionCalibrationMetrics,
    OpinionExample,
    OpinionExampleInputs,
    OpinionSpec,
    build_opinion_example,
    exclude_eval_participants,
    ensure_train_examples,
    float_or_none,
    make_opinion_inputs,
    log_participant_counts,
    make_opinion_example,
    make_opinion_example_from_values,
    opinion_example_kwargs,
)
from .results import OpinionArtifacts, OpinionEvaluationResult, OpinionStudyResult
from .sweep_types import (
    AccuracySummary,
    BaseOpinionSweepOutcome,
    BaseOpinionSweepTask,
    BaseSweepTask,
    MetricsArtifact,
    SWEEP_PUBLIC,
)

__all__ = [
    *SWEEP_PUBLIC,
    "DEFAULT_SPECS",
    "OpinionCalibrationMetrics",
    "OpinionExample",
    "OpinionExampleInputs",
    "OpinionMetricsView",
    "OpinionArtifacts",
    "OpinionEvaluationResult",
    "OpinionStudyResult",
    "OpinionSpec",
    "build_opinion_example",
    "format_opinion_user_prompt",
    "compute_opinion_metrics",
    "exclude_eval_participants",
    "ensure_train_examples",
    "float_or_none",
    "make_opinion_inputs",
    "log_participant_counts",
    "make_opinion_example",
    "make_opinion_example_from_values",
    "opinion_example_kwargs",
    "summarise_opinion_metrics",
]
