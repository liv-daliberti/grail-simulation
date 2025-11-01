#!/usr/bin/env python
"""Tests for shared JSON/Markdown IO helpers used across pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.pipeline.io import (
    iter_jsonl_rows,
    load_metrics_json,
    write_jsonl_rows,
    write_markdown_lines,
    write_metrics_json,
    write_segmented_markdown_log,
)


def test_write_and_load_metrics_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    payload = {"a": 1, "b": {"c": 2}}
    write_metrics_json(path, payload)
    # File should end with a newline and be readable as JSON
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert load_metrics_json(path) == payload


def test_iter_jsonl_rows_handles_blanks_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    lines = [
        "\n",
        json.dumps({"x": 1}) + "\n",
        "not json\n",
        json.dumps({"y": 2}) + "\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")

    rows = list(iter_jsonl_rows(path))
    assert rows == [{"x": 1}, {"y": 2}]

    with pytest.raises(json.JSONDecodeError):
        list(iter_jsonl_rows(path, ignore_errors=False))


def test_write_jsonl_rows_produces_valid_lines(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    rows = [{"foo": 1}, {"bar": 2}]
    write_jsonl_rows(path, rows)
    back = list(iter_jsonl_rows(path))
    assert back == rows


def test_write_segmented_markdown_log_renders_header_and_entries(tmp_path: Path) -> None:
    path = tmp_path / "qa.log"
    entries = ["first", "second\n"]
    write_segmented_markdown_log(path, title="Title", entries=entries)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Title\n\n")
    # Ensure separator inserted between entries and trailing newline logic works
    assert "first\n\n---\n\nsecond\n\n---\n\n" in text


def test_write_markdown_lines_appends_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "readme.md"
    write_markdown_lines(path, ["a", "b"])
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.splitlines() == ["a", "b"]
