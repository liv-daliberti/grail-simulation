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

"""Public interface for the XGBoost pipeline report builders."""

from __future__ import annotations

from common.pipeline.formatters import (
    format_count as _format_count,
    format_delta as _format_delta,
    format_float as _format_float,
    format_optional_float as _format_optional_float,
    format_ratio as _format_ratio,
)

from .catalog import _write_catalog_report
from .features import _write_feature_report
from .hyperparameter import _write_hyperparameter_report
from .next_video import _extract_next_video_summary, _write_next_video_report
from .opinion import (
    OpinionReportOptions as _OpinionReportOptions,
    _extract_opinion_summary,
    _write_opinion_report,
)
from .runner import OpinionReportData, ReportSections, SweepReportData, _write_reports
from .shared import (
    _format_shell_command,
    _slugify_label,
    _write_disabled_report,
    _xgb_next_video_command,
    _xgb_opinion_command,
)

OpinionReportOptions = _OpinionReportOptions

__all__ = [
    name
    for name in (
        "_format_float",
        "_format_optional_float",
        "_format_delta",
        "_format_count",
        "_format_ratio",
        "_extract_next_video_summary",
        "_write_next_video_report",
        "_extract_opinion_summary",
        "_write_opinion_report",
        "OpinionReportOptions",
        "_write_reports",
        "_write_catalog_report",
        "_write_feature_report",
        "_write_hyperparameter_report",
        "_slugify_label",
        "_format_shell_command",
        "_xgb_next_video_command",
        "_xgb_opinion_command",
        "SweepReportData",
        "OpinionReportData",
        "ReportSections",
        "_write_disabled_report",
    )
    if name in globals()
]
