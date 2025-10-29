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

"""Compatibility shim for :mod:`xgb.pipeline.reports.opinion`."""

from __future__ import annotations

from ..pipeline.reports.opinion import (  # type: ignore[F401]
    _OpinionPortfolioAccumulator,
    _WeightedMetricAccumulator,
    _extract_opinion_summary,
    _opinion_cross_study_diagnostics,
    _opinion_observations,
    _write_opinion_report,
)

__all__ = [
    "_OpinionPortfolioAccumulator",
    "_WeightedMetricAccumulator",
    "_extract_opinion_summary",
    "_opinion_cross_study_diagnostics",
    "_opinion_observations",
    "_write_opinion_report",
]
