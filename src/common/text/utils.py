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

"""String normalisation helpers shared across experiments."""

from __future__ import annotations

import re
from typing import Iterable

import os

_YTID_EXPR = r"([A-Za-z0-9_-]{11})"
_CANON_EXPR = r"[^a-z0-9]+"

YTID_RE = re.compile(_YTID_EXPR)
CANON_RE = re.compile(_CANON_EXPR)


def canon_text(text: str | None) -> str:
    """
    Return a lowercased alphanumeric representation of ``text``.

    :param text: Input string to normalise.
    :returns: Canonical token containing only lowercase alphanumeric characters.
    """
    if not text:
        return ""
    return CANON_RE.sub("", text.lower().strip())


def canon_video_id(value: object | None) -> str:
    """
    Extract a canonical YouTube id from ``value`` when possible.

    :param value: Raw video identifier or URL.
    :returns: 11-character YouTube ID or an empty string when unavailable.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    match = YTID_RE.search(value)
    if match:
        return match.group(1)
    return value.strip()


def split_env_list(value: str | None) -> list[str]:
    """
    Parse colon/comma/space separated environment list values.

    :param value: Raw environment variable string.
    :returns: List of non-empty tokens extracted from ``value``.
    """
    if not value:
        return []
    return [
        token
        for chunk in re.split(r"[:,\s]+", value)
        if (token := chunk.strip())
    ]


def resolve_paths_from_env(variable_names: Iterable[str]) -> list[str]:
    """
    Aggregate filesystem paths from the given environment variables.

    :param variable_names: Iterable of environment variable names to read.
    :returns: Combined list of resolved filesystem paths.
    """
    paths: list[str] = []
    for name in variable_names:
        paths.extend(split_env_list(os.environ.get(name)))
    return paths


__all__ = [
    "CANON_RE",
    "YTID_RE",
    "canon_text",
    "canon_video_id",
    "resolve_paths_from_env",
    "split_env_list",
]
