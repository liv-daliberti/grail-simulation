"""Low-impact unit tests for clean_data.io helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from clean_data.io import (
    maybe_literal_eval,
    read_csv_if_exists,
    read_survey_with_fallback,
    resolve_capsule_data_root,
)

pytestmark = pytest.mark.clean_data


def test_resolve_capsule_data_root_recognizes_direct_capsule(tmp_path: Path) -> None:
    platform_dir = tmp_path / "platform session data"
    platform_dir.mkdir(parents=True)
    (platform_dir / "sessions.json").write_text("[]", encoding="utf-8")

    resolved = resolve_capsule_data_root(tmp_path)

    assert resolved == tmp_path


def test_resolve_capsule_data_root_finds_data_subdirectory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "platform session data"
    data_dir.mkdir(parents=True)
    (data_dir / "sessions.json").write_text("[]", encoding="utf-8")

    resolved = resolve_capsule_data_root(tmp_path)

    assert resolved == tmp_path / "data"


def test_resolve_capsule_data_root_returns_none_when_missing(tmp_path: Path) -> None:
    assert resolve_capsule_data_root(tmp_path) is None


def test_read_csv_if_exists_returns_dataframe(tmp_path: Path) -> None:
    csv_path = tmp_path / "example.csv"
    csv_path.write_text("col\nvalue\n", encoding="utf-8")

    frame = read_csv_if_exists(csv_path)

    assert list(frame.columns) == ["col"]
    assert frame.iloc[0]["col"] == "value"


def test_read_csv_if_exists_returns_empty_for_missing_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing.csv"

    frame = read_csv_if_exists(csv_path)

    assert frame.empty


def test_read_survey_with_fallback_prefers_urlid_file(tmp_path: Path) -> None:
    first = tmp_path / "survey1.csv"
    first.write_text("id,name\n1,Alice\n", encoding="utf-8")
    second = tmp_path / "survey2.csv"
    second.write_text("urlid,name\nvideo://123,Bob\n", encoding="utf-8")

    frame = read_survey_with_fallback(first, second)

    assert list(frame.columns) == ["urlid", "name"]
    assert frame.iloc[0]["urlid"] == "video://123"


def test_read_survey_with_fallback_returns_empty_when_none_exist(tmp_path: Path) -> None:
    frame = read_survey_with_fallback(tmp_path / "missing1.csv", tmp_path / "missing2.csv")
    assert frame.empty


@pytest.mark.parametrize(
    "raw,expected",
    [
        (" true ", True),
        ("FALSE", False),
        ("NaN", None),
        ("[1, 2, 3]", [1, 2, 3]),
        ('{"a": 1}', {"a": 1}),
        ("", ""),
        ("plain", "plain"),
    ],
)
def test_maybe_literal_eval_handles_common_values(raw: str, expected) -> None:
    assert maybe_literal_eval(raw) == expected
