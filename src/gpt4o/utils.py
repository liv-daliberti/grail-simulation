"""Shared utility helpers for the GPT-4o slate baseline."""

from __future__ import annotations

import os
import re
from typing import Iterable, Optional

ANS_TAG = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
INDEX_ONLY = re.compile(r"^\s*(?:option\s*)?(\d+)\s*$", re.I)
YTID_RE = re.compile(r"([A-Za-z0-9_-]{11})")
CANON_RE = re.compile(r"[^a-z0-9]+")


def canon_text(text: str | None) -> str:
    """Return a lowercased alphanumeric canonical representation."""

    if not text:
        return ""
    return CANON_RE.sub("", text.lower().strip())


def canon_video_id(video_id: str | None) -> str:
    """Normalise a YouTube id if possible."""

    if not video_id:
        return ""
    if isinstance(video_id, str):
        match = YTID_RE.search(video_id)
        if match:
            return match.group(1)
    return str(video_id).strip()


def is_nan_like(value: object | None) -> bool:
    """Return True when the provided value should be treated as missing."""

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null", "na", "n/a"}
    return False


def truthy(value: object | None) -> bool:
    """Return True for typical boolean truthy values used in the dataset."""

    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    string_value = str(value).strip().lower()
    return string_value in {"1", "true", "t", "yes", "y"}


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
    """Collect file or directory paths from a set of env var names."""

    paths: list[str] = []
    for name in variable_names:
        paths.extend(split_env_list(os.environ.get(name)))
    return paths

