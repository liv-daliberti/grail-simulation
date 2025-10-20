"""Shared utility helpers for the GPT-4o slate baseline."""

from __future__ import annotations

import re

from common.text import (
    CANON_RE,
    YTID_RE,
    canon_text,
    canon_video_id,
    resolve_paths_from_env,
    split_env_list,
)

ANS_TAG = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
INDEX_ONLY = re.compile(r"^\s*(?:option\s*)?(\d+)\s*$", re.I)


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
