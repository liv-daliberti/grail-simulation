"""Unit tests for :mod:`common.report_utils`."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from common.report_utils import (
    extend_with_catalog_sections,
    extract_curve_sections,
    extract_numeric_series,
    start_markdown_report,
)


def test_start_markdown_report_creates_directory(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "knn"
    path, lines = start_markdown_report(report_dir, title="My Report")

    assert path == report_dir / "README.md"
    assert path.parent.is_dir()
    assert lines == ["# My Report", ""]


def test_extend_with_catalog_sections_inserts_separator() -> None:
    lines = ["Existing heading"]

    extend_with_catalog_sections(
        lines,
        include_next_video=True,
        include_opinion=True,
        reports_prefix="knn",
    )

    assert lines[1] == ""
    assert "- `additional_features/README.md` — overview of the extra text fields appended to prompts." in lines
    assert any("hyperparameter_tuning/README.md" in entry for entry in lines)
    assert any("reports/knn/opinion/" in entry for entry in lines)


def test_extract_numeric_series_filters_invalid_entries() -> None:
    curve_map = OrderedDict(
        {
            "3": "4.25",
            "1": 2,
            "bad-step": "value",
            "2": "3.5",
            "4": None,
        }
    )

    steps, metrics = extract_numeric_series(curve_map)

    assert steps == [1, 2, 3]
    assert metrics == [2.0, 3.5, 4.25]


def test_extract_curve_sections_handles_missing_training() -> None:
    bundle = {"eval": {"1": 0.5, "2": 0.6}}
    sections = extract_curve_sections(bundle)
    assert sections == (bundle["eval"], None)

    full_bundle = {"eval": {"1": 0.5}, "train": {"1": 0.4}}
    sections = extract_curve_sections(full_bundle)
    assert sections == (full_bundle["eval"], full_bundle["train"])

    assert extract_curve_sections(42) is None
