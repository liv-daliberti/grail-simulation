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

"""Lightweight helpers for constructing Markdown-based pipeline reports."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Tuple


def start_markdown_report(
    directory: Path,
    *,
    title: str,
    filename: str = "README.md",
) -> Tuple[Path, List[str]]:
    """Create the report directory and seed Markdown lines with the title."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    lines: List[str] = [f"# {title}", ""]
    return path, lines


def extract_numeric_series(curve_map: Mapping[str, object]) -> Tuple[List[int], List[float]]:
    """Return sorted integer keys and float metrics from ``curve_map``."""
    points: List[Tuple[int, float]] = []
    for raw_step, raw_value in curve_map.items():
        try:
            step_val = int(raw_step)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        points.append((step_val, value))
    if not points:
        return ([], [])
    points.sort(key=lambda item: item[0])
    step_values, metric_values = zip(*points)
    return (list(step_values), list(metric_values))


__all__ = ["extract_numeric_series", "start_markdown_report"]
