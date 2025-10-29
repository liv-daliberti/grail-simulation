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

"""Compatibility layer exposing the legacy ``xgb.pipeline_reports`` API.

The production modules now live under :mod:`xgb.pipeline.reports`.  Tests and
downstream scripts still rely on the legacy import paths, so we re-export the
public entry points here to ease the transition to the reorganised package.
"""

from __future__ import annotations

from ..pipeline import reports as _reports
from ..pipeline.reports import (
    OpinionReportData,
    ReportSections,
    SweepReportData,
    _extract_next_video_summary,
    _extract_opinion_summary,
    _format_count,
    _format_delta,
    _format_float,
    _format_optional_float,
    _format_ratio,
    _slugify_label,
    _format_shell_command,
    _write_catalog_report,
    _write_disabled_report,
    _write_feature_report,
    _write_hyperparameter_report,
    _write_next_video_report,
    _write_opinion_report,
    _write_reports,
    _xgb_next_video_command,
    _xgb_opinion_command,
)

__all__ = list(_reports.__all__)
