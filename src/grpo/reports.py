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

"""GRPO-specific wrapper around the shared RLHF reporting helpers."""

from __future__ import annotations

from pathlib import Path

from grpo.next_video import NextVideoEvaluationResult
from grpo.opinion import OpinionEvaluationResult
from common.rlhf.reports import ReportOptions, generate_reports as _generate_reports

DEFAULT_BASELINE_LABEL = "GRPO"
DEFAULT_REGENERATE_HINT = (
    "Regenerate via `python -m grpo.pipeline --stage full` after producing "
    "updated evaluation artifacts under `models/grpo/`."
)


def generate_reports(
    *,
    repo_root: Path,
    next_video: NextVideoEvaluationResult | None,
    opinion: OpinionEvaluationResult | None,
    options: ReportOptions | None = None,
) -> None:
    """
    Materialise Markdown reports for GRPO baselines.

    :param repo_root: Repository root used to resolve report output locations.
    :param next_video: In-memory next-video evaluation results (or ``None`` when skipped).
    :param opinion: In-memory opinion evaluation results (or ``None`` when skipped).
    :param options: Optional report configuration (reports subdir, labels, hints).
    :returns: ``None``.
    """

    resolved_options = options or ReportOptions(
        "grpo",  # reports_subdir
        DEFAULT_BASELINE_LABEL,
        DEFAULT_REGENERATE_HINT,
    )
    _generate_reports(
        repo_root=repo_root,
        next_video=next_video,
        opinion=opinion,
        options=resolved_options,
    )
