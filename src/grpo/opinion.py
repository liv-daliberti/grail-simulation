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

"""Opinion-shift evaluation for finetuned GRPO checkpoints.

This module now serves as a thin compatibility layer that re-exports the
public API from smaller submodules to keep file size and linting healthy.
"""

from __future__ import annotations

from .opinion_types import (
    OpinionArtifacts,
    OpinionDatasetSpec,
    OpinionEvaluationControls,
    OpinionEvaluationResult,
    OpinionEvaluationSettings,
    OpinionInferenceContext,
    OpinionPromptSettings,
    OpinionStudyContext,
    OpinionStudyFiles,
    OpinionStudyResult,
    OpinionStudySummary,
)
from .opinion_types import __all__ as _TYPES_ALL
from .opinion_runner import run_opinion_evaluation

# Export list consumed by grpo.pipeline for star re-exports.
# Keep in sync with grpo.opinion_types.__all__ without repeating names here.
OPINION_PUBLIC_EXPORTS = tuple(_TYPES_ALL)

__all__ = list(_TYPES_ALL) + ["run_opinion_evaluation", "OPINION_PUBLIC_EXPORTS"]
