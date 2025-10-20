"""Unit tests for :mod:`gpt4o.utils` helpers."""

from __future__ import annotations

import sys
import types

import pytest

from tests.helpers.datasets_stub import ensure_datasets_stub

pytestmark = pytest.mark.gpt4o


def _install_dependency_stubs() -> None:
    """Ensure optional external dependencies are stubbed for imports."""

    ensure_datasets_stub()

    if "openai" not in sys.modules:
        openai_stub = types.ModuleType("openai")

        class _AzureOpenAI:
            def __init__(self, **_kwargs):
                pass

        openai_stub.AzureOpenAI = _AzureOpenAI
        sys.modules["openai"] = openai_stub


_install_dependency_stubs()

from gpt4o.utils import (  # pylint: disable=wrong-import-position
    canon_text,
    canon_video_id,
    is_nan_like,
    resolve_paths_from_env,
    split_env_list,
    truthy,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello, World!", "helloworld"),
        ("   MiXeD Case 123!  ", "mixedcase123"),
        (None, ""),
        ("", ""),
    ],
)
def test_canon_text_normalises_input(raw: str | None, expected: str) -> None:
    assert canon_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("not_an_id", "not_an_id"),
        (None, ""),
    ],
)
def test_canon_video_id_extracts_valid_identifier(raw: str | None, expected: str) -> None:
    assert canon_video_id(raw) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("", True),
        (" NaN ", True),
        ("n/a", True),
        (0, False),
        ("valid", False),
    ],
)
def test_is_nan_like_identifies_missing_values(value: object | None, expected: bool) -> None:
    assert is_nan_like(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        (0, False),
        (1, True),
        ("YES", True),
        ("  no  ", False),
        ("TrUe", True),
        ("42", False),
    ],
)
def test_truthy_matches_dataset_conventions(value: object | None, expected: bool) -> None:
    assert truthy(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("path1:path2, path3  path4", ["path1", "path2", "path3", "path4"]),
        (" single ", ["single"]),
        (None, []),
        ("", []),
    ],
)
def test_split_env_list_supports_multiple_separators(value: str | None, expected: list[str]) -> None:
    assert split_env_list(value) == expected


def test_resolve_paths_from_env_collects_multiple_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV_A", "foo:bar")
    monkeypatch.setenv("ENV_B", "baz qux")
    # Ensure unrelated variables do not interfere
    monkeypatch.delenv("ENV_C", raising=False)

    paths = resolve_paths_from_env(["ENV_A", "ENV_B", "ENV_C"])
    assert paths == ["foo", "bar", "baz", "qux"]

    # Removing env vars should change the resolved paths.
    monkeypatch.delenv("ENV_A", raising=False)
    monkeypatch.delenv("ENV_B", raising=False)
    assert resolve_paths_from_env(["ENV_A", "ENV_B"]) == []
