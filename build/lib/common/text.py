"""Canonical text helpers reused across baseline implementations."""

from __future__ import annotations

import re
from typing import Iterable

import os

_YTID_EXPR = r"([A-Za-z0-9_-]{11})"
_CANON_EXPR = r"[^a-z0-9]+"

YTID_RE = re.compile(_YTID_EXPR)
CANON_RE = re.compile(_CANON_EXPR)


def canon_text(text: str | None) -> str:
    """Return a lowercased alphanumeric canonical representation."""

    if not text:
        return ""
    return CANON_RE.sub("", text.lower().strip())


def canon_video_id(value: object | None) -> str:
    """Normalise a YouTube id when present in ``value``."""

    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    match = YTID_RE.search(value)
    if match:
        return match.group(1)
    return value.strip()


def split_env_list(value: str | None) -> list[str]:
    """Parse a colon/comma/space separated list from an environment string."""

    if not value:
        return []
    return [
        token
        for chunk in re.split(r"[:,\s]+", value)
        if (token := chunk.strip())
    ]


def resolve_paths_from_env(variable_names: Iterable[str]) -> list[str]:
    """Collect file or directory paths from a set of environment variables."""

    paths: list[str] = []
    for name in variable_names:
        paths.extend(split_env_list(os.environ.get(name)))
    return paths


__all__ = ["CANON_RE", "YTID_RE", "canon_text", "canon_video_id", "resolve_paths_from_env", "split_env_list"]
