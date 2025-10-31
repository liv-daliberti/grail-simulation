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

"""Top-level exports for the KNN evaluation helpers."""

from __future__ import annotations

from .curves import compute_auc_from_curve, plot_elbow
from .k_selection import parse_k_values, select_best_k
from .outputs import resolve_reports_dir
from .pipeline import evaluate_issue, run_eval
from .utils import bin_nopts, bucket_from_pos, canon

__all__ = [
    "bin_nopts",
    "bucket_from_pos",
    "canon",
    "compute_auc_from_curve",
    "evaluate_issue",
    "parse_k_values",
    "plot_elbow",
    "resolve_reports_dir",
    "run_eval",
    "select_best_k",
]
