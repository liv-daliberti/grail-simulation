#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility helpers shared across the GPT-4o baseline implementation."""

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
