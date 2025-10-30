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

"""Subpackage exposing the XGBoost opinion report helpers."""

from __future__ import annotations

from .accumulators import _OpinionPortfolioAccumulator, _WeightedMetricAccumulator
from .observations import _opinion_cross_study_diagnostics, _opinion_observations
from .report import OpinionReportOptions, _write_opinion_report
from .summaries import _extract_opinion_summary

__all__ = [
    "_OpinionPortfolioAccumulator",
    "_WeightedMetricAccumulator",
    "OpinionReportOptions",
    "_extract_opinion_summary",
    "_opinion_cross_study_diagnostics",
    "_opinion_observations",
    "_write_opinion_report",
]
