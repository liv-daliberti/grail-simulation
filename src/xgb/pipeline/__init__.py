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

"""Pipeline orchestration for the XGBoost baselines.

This package coordinates sweep execution, evaluation, and report emission for
the slate-ranking and opinion-regression XGBoost workflows. The heavy lifting
now lives in submodules (``settings``, ``stages``, ``runner``) to keep this
module lightweight and focused on the public surface.
"""

from __future__ import annotations

import logging

from common.pipeline.stage import prepare_sweep_execution as _prepare_sweep_execution

from .context import (
    OpinionStageConfig,
    OpinionSweepRunContext,
    SweepRunContext,
)
from .runner import main
from .cli import (
    _resolve_study_specs,  # re-export for tests
    _build_sweep_configs,
)
from .sweeps.next_video import (
    _prepare_sweep_tasks,
    _execute_sweep_tasks,
)
from .sweeps.opinion import (
    _prepare_opinion_sweep_tasks,
    _execute_opinion_sweep_tasks,
)
from .sweeps import (
    _emit_combined_sweep_plan,
    _merge_sweep_outcomes,
    _merge_opinion_sweep_outcomes,
    _select_best_configs,
    _select_best_opinion_configs,
    _load_final_metrics_from_disk,
    _load_loso_metrics_from_disk,
    _load_opinion_metrics_from_disk,
    _load_opinion_from_next_metrics_from_disk,
)
from .evaluate import (
    _run_final_evaluations,
    _run_cross_study_evaluations,
    _run_opinion_stage,
    _run_opinion_from_next_stage,
)
from .reports import _write_reports

LOGGER = logging.getLogger("xgb.pipeline")

__all__ = [
    "main",
    "SweepRunContext",
    "OpinionStageConfig",
    "OpinionSweepRunContext",
]

# Re-export for tests and integration shims that patch this symbol.
prepare_sweep_execution = _prepare_sweep_execution

if __name__ == "__main__":  # pragma: no cover
    main()
