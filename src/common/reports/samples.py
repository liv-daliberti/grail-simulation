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

"""Render sample generative responses from existing evaluation artefacts.

This module assembles small, human-readable examples showing the exact
question shown to a model and the model's structured response using
the <think>…</think> and <answer>…</answer> tags (with an optional
<opinion> label for opinion-shift predictions).

Outputs are written under a new subdirectory in the reports tree:

  reports/<family>/sample_generative_responses/README.md

where <family> is one of "grpo", "grail", or "gpt4o".
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from common.pipeline.io import write_markdown_lines

from .samples_report import build_sample_report_lines


def write_sample_responses_report(
    *,
    reports_root: Path,
    family_label: str,
    next_video_files: Sequence[Path],
    opinion_files: Sequence[Path],
    per_issue: int = 5,
) -> None:
    """Assemble and write the sample generative responses report."""

    samples_dir = reports_root / "sample_generative_responses"
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / "README.md"

    lines: List[str] = build_sample_report_lines(
        family_label=family_label,
        next_video_files=next_video_files,
        opinion_files=opinion_files,
        per_issue=per_issue,
    )
    write_markdown_lines(path, lines)


__all__ = [
    "write_sample_responses_report",
]
