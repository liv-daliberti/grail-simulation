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

from common.title_index import TitleResolver as _BaseTitleResolver

from .config import DEFAULT_TITLE_DIRS


class TitleResolver(_BaseTitleResolver):
    """Resolve YouTube ids to titles using shared common helpers."""

    def __init__(self, **kwargs) -> None:
        """Initialise the resolver with shared default title directories.

        :param kwargs: Optional keyword overrides forwarded to the base class.
        """
        super().__init__(**self._merge_defaults(kwargs))

    @staticmethod
    def _merge_defaults(overrides: dict[str, object]) -> dict[str, object]:
        """Return keyword arguments merged with the shared defaults."""

        defaults: dict[str, object] = {"default_dirs": DEFAULT_TITLE_DIRS}
        defaults.update(overrides)
        return defaults

    @classmethod
    def with_directories(cls, *directories: str, **kwargs) -> "TitleResolver":
        """Return a resolver overriding the default search directories."""

        params = cls._merge_defaults(kwargs)
        if directories:
            params["default_dirs"] = list(directories)
        return cls(**params)
