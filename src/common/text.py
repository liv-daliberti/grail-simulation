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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

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

    Return a lowercased alphanumeric canonical representation.



    :param text: Value provided for ``text``.

    :type text: str | None

    :returns: Result produced by ``canon_text``.

    :rtype: str

    """


    if not text:
        return ""
    return CANON_RE.sub("", text.lower().strip())


def canon_video_id(value: object | None) -> str:
    """

    Normalise a YouTube id when present in ``value``.



    :param value: Value provided for ``value``.

    :type value: object | None

    :returns: Result produced by ``canon_video_id``.

    :rtype: str

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

    Parse a colon/comma/space separated list from an environment string.



    :param value: Value provided for ``value``.

    :type value: str | None

    :returns: Result produced by ``split_env_list``.

    :rtype: list[str]

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

    Collect file or directory paths from a set of environment variables.



    :param variable_names: Value provided for ``variable_names``.

    :type variable_names: Iterable[str]

    :returns: Result produced by ``resolve_paths_from_env``.

    :rtype: list[str]

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
