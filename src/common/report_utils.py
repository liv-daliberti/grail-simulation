"""Utility helpers for Markdown report writers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Tuple


def start_markdown_report(
    directory: Path,
    *,
    title: str,
    filename: str = "README.md",
) -> Tuple[Path, List[str]]:
    """Create ``directory`` and return the report path plus initial lines."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    lines: List[str] = [f"# {title}", ""]
    return path, lines


def extract_numeric_series(curve_map: Mapping[str, object]) -> Tuple[List[int], List[float]]:
    """Return sorted integer keys and float values from ``curve_map``."""

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
    xs, ys = zip(*points)
    return (list(xs), list(ys))


__all__ = ["extract_numeric_series", "start_markdown_report"]
