"""Shared utility helpers for the GPT-4o slate baseline."""

from __future__ import annotations

import re

from common.text import (
    canon_text as _canon_text,
    canon_video_id as _canon_video_id,
    resolve_paths_from_env as _resolve_paths_from_env,
    split_env_list as _split_env_list,
)

ANS_TAG = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
INDEX_ONLY = re.compile(r"^\s*(?:option\s*)?(\d+)\s*$", re.I)


def canon_text(value: str | None) -> str:
    """Normalise ``value`` using the shared canonical text helper."""

    return _canon_text(value)


def canon_video_id(value: str | None) -> str:
    """Extract a canonical YouTube id from ``value``."""

    return _canon_video_id(value)


def split_env_list(raw: str | None) -> list[str]:
    """Split ``raw`` using the separators understood by the common helper."""

    return _split_env_list(raw)


def resolve_paths_from_env(env_vars: list[str]) -> list[str]:
    """Return resolved filesystem paths aggregated from ``env_vars``."""

    return _resolve_paths_from_env(env_vars)


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
