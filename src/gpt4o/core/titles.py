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

from importlib import import_module
from .config import DEFAULT_TITLE_DIRS
_title_index = import_module("common.text.title_index")
_BaseTitleResolver = _title_index.TitleResolver


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

    def title_for(self, video_id: str | None):
        """
        Convenience wrapper around :meth:`~common.text.title_index.TitleResolver.resolve`
        matching other APIs.
        """
        return self.resolve(video_id)

    @property
    def default_dirs(self) -> list[str]:
        """Return a copy of the default title directories used by the resolver."""
        return list(DEFAULT_TITLE_DIRS)
