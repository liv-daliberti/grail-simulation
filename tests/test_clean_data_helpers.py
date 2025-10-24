"""Micro-tests for clean_data.helpers utilities."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
import pytest

from clean_data import helpers

pytestmark = pytest.mark.clean_data


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        ("", True),
        (" NaN ", True),
        ("null", True),
        ("value", False),
    ],
)
def test_is_nanlike_detects_missing_tokens(value, expected: bool) -> None:
    assert helpers._is_nanlike(value) is expected  # pylint: disable=protected-access


def test_as_list_json_accepts_strings_and_lists() -> None:
    assert helpers._as_list_json("[1, 2]") == [1, 2]  # pylint: disable=protected-access
    assert helpers._as_list_json(["a", "b"]) == ["a", "b"]  # pylint: disable=protected-access
    assert helpers._as_list_json("not json") == []  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ-extra", "dQw4w9WgXcQ"),
        ("short", "short"),
        (123, ""),
    ],
)
def test_strip_session_video_id_normalises_inputs(raw, expected: str) -> None:
    assert helpers._strip_session_video_id(raw) == expected  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "value,expected",
    [
        (123.0, "123"),
        ("123.0", "123"),
        ("123.5", "123.5"),
        ("", ""),
        ("nan", ""),
        (None, ""),
        ("001", "1"),
    ],
)
def test_normalize_urlid_handles_numeric_strings(value, expected: str) -> None:
    assert helpers._normalize_urlid(value) == expected  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "value,expected",
    [
        (" worker01 ", "worker01"),
        ("NaN", ""),
        (None, ""),
    ],
)
def test_normalize_identifier_strips_tokens(value, expected: str) -> None:
    assert helpers._normalize_identifier(value) == expected  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "value,expected",
    [
        (42, 42),
        (3.14, 3.14),
        (" 15 ", 15),
        (" 2.5 ", 2.5),
        ("", None),
        ("text", "text"),
    ],
)
def test_coerce_session_value_casts_strings(value, expected) -> None:
    assert helpers._coerce_session_value(value) == expected  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        (math.nan, True),
        ("", True),
        ("NA", True),
        ("value", False),
    ],
)
def test_is_missing_value_spots_blanks(value, expected: bool) -> None:
    assert helpers._is_missing_value(value) is expected  # pylint: disable=protected-access


def test_parse_timestamp_ns_supports_numeric_and_iso() -> None:
    ts_ms = 1_610_000_000_000
    expected_ns = int(pd.to_datetime(ts_ms, unit="ms", utc=True).value)
    assert helpers._parse_timestamp_ns(ts_ms) == expected_ns  # pylint: disable=protected-access

    iso = "2021-01-13T00:00:00Z"
    iso_ns = int(datetime(2021, 1, 13, tzinfo=timezone.utc).timestamp() * 1_000_000_000)
    assert helpers._parse_timestamp_ns(iso) == iso_ns  # pylint: disable=protected-access

    assert helpers._parse_timestamp_ns("invalid") is None  # pylint: disable=protected-access

