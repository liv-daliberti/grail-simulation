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

"""Training-curve rendering helpers for opinion reports."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping

from .summaries import _extract_opinion_summary
from ..plots import _plot_opinion_curve, plt


def _opinion_curve_lines(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
) -> List[str]:
    """
    Render training curve images, returning Markdown lines referencing them.

    :param directory: Report directory receiving rendered figures.
    :param metrics: Mapping from study identifiers to metrics payloads.
    :returns: Markdown lines referencing the generated training curves.
    """

    if plt is None:
        return []
    curve_lines: List[str] = []
    for study_key, payload in sorted(metrics.items()):
        summary = _extract_opinion_summary(payload)
        rel_path = _plot_opinion_curve(
            directory=directory,
            study_label=summary.label or study_key,
            study_key=study_key,
            payload=payload,
        )
        if rel_path:
            if not curve_lines:
                curve_lines.extend(["## Training Curves", ""])
            curve_lines.append(f"![{summary.label or study_key}]({rel_path})")
            curve_lines.append("")
    return curve_lines


__all__ = [
    "_opinion_curve_lines",
]
