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

"""Opinion evaluation helpers and runners."""

from common.opinion import DEFAULT_SPECS  # re-export for compatibility

from .helpers import (
    baseline_metrics,
    clip_prediction,
    document_from_example,
    float_or_none,
)
from .models import (
    CombinedAccumulator,
    OpinionArtifacts,
    OpinionEvaluationResult,
    OpinionSettings,
    OpinionStudyResult,
    StudyPredictionBatch,
)
from .runner import OpinionEvaluationRunner, run_opinion_evaluations
from .settings import (
    build_settings,
    parse_tokens,
    resolve_spec_keys,
)

__all__ = [
    "DEFAULT_SPECS",
    "CombinedAccumulator",
    "OpinionArtifacts",
    "OpinionEvaluationRunner",
    "OpinionEvaluationResult",
    "OpinionSettings",
    "OpinionStudyResult",
    "StudyPredictionBatch",
    "baseline_metrics",
    "build_settings",
    "clip_prediction",
    "document_from_example",
    "float_or_none",
    "parse_tokens",
    "resolve_spec_keys",
    "run_opinion_evaluations",
]
