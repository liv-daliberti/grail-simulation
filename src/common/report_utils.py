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
from typing import List, Mapping, Optional, Tuple


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


def append_catalog_sections(
    lines: List[str],
    *,
    include_next_video: bool,
    include_opinion: bool,
    reports_prefix: str,
) -> None:
    """
    Append the shared catalog bullets describing downstream report artefacts.

    :param lines: Mutable list of Markdown lines to extend.
    :param include_next_video: Whether next-video metrics should be mentioned.
    :param include_opinion: Whether opinion metrics should be mentioned.
    :param reports_prefix: Directory prefix (``knn`` or ``xgb``) used for report paths.
    """
    lines.append(
        "- `additional_features/README.md` — overview of the extra text fields appended to prompts."
    )
    if include_next_video or include_opinion:
        lines.append("")
    if include_next_video:
        lines.append(
            "- `hyperparameter_tuning/README.md` — sweep leaderboards and the per-study winners."
        )
        lines.append(
            "- `next_video/README.md` — validation accuracy, confidence intervals, baseline deltas,"
            " and training versus validation accuracy curves for the production slate task."
        )
    if include_opinion:
        lines.append(
            "- `opinion/README.md` — post-study opinion regression metrics, plus heatmaps under "
            f"`reports/{reports_prefix}/opinion/`."
        )


def extend_with_catalog_sections(
    lines: List[str],
    *,
    include_next_video: bool,
    include_opinion: bool,
    reports_prefix: str,
) -> None:
    """
    Append the shared catalog sections, ensuring a blank line separator first.

    :param lines: Mutable list of Markdown lines to extend.
    :param include_next_video: Whether next-video metrics should be mentioned.
    :param include_opinion: Whether opinion metrics should be mentioned.
    :param reports_prefix: Directory prefix (``knn`` or ``xgb``) used for report paths.
    """

    if lines and lines[-1] != "":
        lines.append("")
    append_catalog_sections(
        lines,
        include_next_video=include_next_video,
        include_opinion=include_opinion,
        reports_prefix=reports_prefix,
    )


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

def extract_curve_sections(
    curve_bundle: object,
) -> Optional[Tuple[Mapping[str, object], Optional[Mapping[str, object]]]]:
    """
    Return the validation and training curve mappings from ``curve_bundle``.

    :returns: Tuple containing the validation mapping and an optional training mapping.
    """

    if not isinstance(curve_bundle, Mapping):
        return None
    eval_curve = curve_bundle.get("eval")
    if not isinstance(eval_curve, Mapping):
        return None
    train_curve = curve_bundle.get("train")
    train_mapping: Optional[Mapping[str, object]] = (
        train_curve if isinstance(train_curve, Mapping) else None
    )
    return (eval_curve, train_mapping)


def append_image_section(
    lines: List[str],
    *,
    image: Path,
    relative_root: Optional[Path] = None,
) -> None:
    """Append Markdown for ``image`` using a path relative to ``relative_root`` when possible."""

    try:
        if relative_root is not None:
            rel_path = image.relative_to(relative_root).as_posix()
        else:
            rel_path = image.as_posix()
    except ValueError:
        rel_path = image.as_posix()
    label = image.stem.replace("_", " ").title()
    lines.append(f"![{label}]({rel_path})")
    lines.append("")


__all__ = [
    "append_image_section",
    "append_catalog_sections",
    "extend_with_catalog_sections",
    "extract_curve_sections",
    "extract_numeric_series",
    "start_markdown_report",
]
