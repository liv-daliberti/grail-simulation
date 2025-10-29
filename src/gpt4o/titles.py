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

"""GPT-4o helpers for resolving video titles used in evaluations."""

from __future__ import annotations

from common.text.title_index import TitleResolver as _BaseTitleResolver

from .config import DEFAULT_TITLE_DIRS


class TitleResolver(_BaseTitleResolver):
    """Resolve YouTube ids to titles using shared common helpers."""

    def __init__(self, **kwargs) -> None:
        """Initialise the resolver with shared default title directories.

        :param kwargs: Optional keyword overrides forwarded to the base class.
        :returns: ``None``.
        """
        super().__init__(**self._merge_defaults(kwargs))

    @staticmethod
    def _merge_defaults(overrides: dict[str, object]) -> dict[str, object]:
        """Return keyword arguments merged with the shared defaults.

        :param overrides: User-provided keyword overrides.
        :returns: Combined keyword argument dictionary.
        """

        defaults: dict[str, object] = {"default_dirs": DEFAULT_TITLE_DIRS}
        defaults.update(overrides)
        return defaults

    @classmethod
    def with_directories(cls, *directories: str, **kwargs) -> "TitleResolver":
        """Return a resolver overriding the default search directories.

        :param directories: One or more directories to prioritise.
        :param kwargs: Additional keyword arguments forwarded to ``TitleResolver``.
        :returns: Configured ``TitleResolver`` instance.
        """

        params = cls._merge_defaults(kwargs)
        if directories:
            params["default_dirs"] = list(directories)
        return cls(**params)
