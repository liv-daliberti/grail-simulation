"""Unit tests for :mod:`common.text` helpers."""

from __future__ import annotations

import pytest

from common.text import (
    CANON_RE,
    YTID_RE,
    canon_text,
    canon_video_id,
    resolve_paths_from_env,
    split_env_list,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello, World!", "helloworld"),
        ("  Mixed CASE 123! ", "mixedcase123"),
        ("", ""),
        (None, ""),
    ],
)
def test_canon_text_normalises_strings(raw: str | None, expected: str) -> None:
    assert canon_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ&t=43", "dQw4w9WgXcQ"),
        ("not_an_id", "not_an_id"),
        (None, ""),
        (12345, ""),
    ],
)
def test_canon_video_id_extracts_identifier(raw: object | None, expected: str) -> None:
    assert canon_video_id(raw) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("path1:path2, path3\tpath4", ["path1", "path2", "path3", "path4"]),
        (" single ", ["single"]),
        (None, []),
        ("", []),
    ],
)
def test_split_env_list_supports_mixed_separators(value: str | None, expected: list[str]) -> None:
    assert split_env_list(value) == expected


def test_resolve_paths_from_env_aggregates_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV_A", "foo bar")
    monkeypatch.setenv("ENV_B", "baz,qux")

    paths = resolve_paths_from_env(["ENV_A", "ENV_B", "ENV_MISSING"])
    assert paths == ["foo", "bar", "baz", "qux"]

    monkeypatch.delenv("ENV_A", raising=False)
    monkeypatch.delenv("ENV_B", raising=False)
    assert resolve_paths_from_env(["ENV_A", "ENV_B"]) == []


def test_compiled_regexes_are_reusable() -> None:
    """Sanity checks that shared regex objects behave as expected."""

    assert CANON_RE.sub("", "hello-world!") == "helloworld"
    match = YTID_RE.search("https://youtu.be/dQw4w9WgXcQ")
    assert match and match.group(1) == "dQw4w9WgXcQ"
