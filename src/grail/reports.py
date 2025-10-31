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

"""Markdown report builders for GRAIL baselines."""

from __future__ import annotations

from pathlib import Path

from grpo.next_video import NextVideoEvaluationResult
from grpo.opinion import OpinionEvaluationResult
from common.rlhf.reports import ReportOptions, generate_reports as _base_generate_reports

__all__ = [
    "DEFAULT_REGENERATE_HINT",
    "generate_reports",
]

DEFAULT_REGENERATE_HINT = (
    "Regenerate via `python -m grail.pipeline --stage full` after producing "
    "updated evaluation artifacts under `models/grail/`."
)


def generate_reports(
    *,
    repo_root: Path,
    next_video: NextVideoEvaluationResult | None,
    opinion: OpinionEvaluationResult | None,
    regenerate_hint: str | None = DEFAULT_REGENERATE_HINT,
) -> None:
    """Materialise Markdown reports for the GRAIL discriminator-augmented runs.

    :param repo_root: Root of the repository where reports are rendered.
    :param next_video: Optional next-video evaluation artefacts.
    :param opinion: Optional opinion evaluation artefacts.
    :param regenerate_hint: Optional sentence describing how to refresh artefacts.
    :returns: ``None``. Markdown reports are generated on disk.
    """

    _base_generate_reports(
        repo_root=repo_root,
        next_video=next_video,
        opinion=opinion,
        options=ReportOptions(
            "grail",  # reports_subdir
            "GRAIL",  # baseline_label
            regenerate_hint,
        ),
    )
