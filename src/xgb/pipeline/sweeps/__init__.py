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

"""Sweep orchestration helpers for the Grail Simulation XGBoost pipeline."""

from __future__ import annotations

from .common import (
    DEFAULT_OPINION_FEATURE_SPACE,
    LOGGER,
    xgboost,
    _gpu_tree_method_supported,
    _inject_study_metadata,
    _load_metrics,
    _load_metrics_with_log,
    _run_xgb_cli,
)
from .next_video import (
    _execute_sweep_task,
    _execute_sweep_tasks,
    _iter_sweep_tasks,
    _load_final_metrics_from_disk,
    _load_loso_metrics_from_disk,
    _merge_sweep_outcomes,
    _prepare_sweep_tasks,
    _run_sweeps,
    _select_best_configs,
    _sweep_outcome_from_metrics,
)
from .opinion import (
    execute_opinion_sweep_tasks,
    _build_opinion_vectorizer_config,
    _execute_opinion_sweep_task,
    _execute_opinion_sweep_tasks,
    _iter_opinion_sweep_tasks,
    _load_opinion_from_next_metrics_from_disk,
    _load_opinion_metrics_from_disk,
    _merge_opinion_sweep_outcomes,
    _opinion_sweep_outcome_from_metrics,
    _prepare_opinion_sweep_tasks,
    _select_best_opinion_configs,
)
from .planning import (
    _emit_combined_sweep_plan,
    _emit_sweep_plan,
    _format_opinion_sweep_task_descriptor,
    _format_sweep_task_descriptor,
)

__all__ = [
    "DEFAULT_OPINION_FEATURE_SPACE",
    "LOGGER",
    "xgboost",
    "execute_opinion_sweep_tasks",
    "_build_opinion_vectorizer_config",
    "_emit_combined_sweep_plan",
    "_emit_sweep_plan",
    "_execute_opinion_sweep_task",
    "_execute_opinion_sweep_tasks",
    "_execute_sweep_task",
    "_execute_sweep_tasks",
    "_format_opinion_sweep_task_descriptor",
    "_format_sweep_task_descriptor",
    "_gpu_tree_method_supported",
    "_inject_study_metadata",
    "_iter_opinion_sweep_tasks",
    "_iter_sweep_tasks",
    "_load_final_metrics_from_disk",
    "_load_loso_metrics_from_disk",
    "_load_metrics",
    "_load_metrics_with_log",
    "_load_opinion_from_next_metrics_from_disk",
    "_load_opinion_metrics_from_disk",
    "_merge_opinion_sweep_outcomes",
    "_merge_sweep_outcomes",
    "_opinion_sweep_outcome_from_metrics",
    "_prepare_opinion_sweep_tasks",
    "_prepare_sweep_tasks",
    "_run_sweeps",
    "_run_xgb_cli",
    "_select_best_configs",
    "_select_best_opinion_configs",
    "_sweep_outcome_from_metrics",
]
