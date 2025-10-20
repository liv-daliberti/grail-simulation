"""Utilities for resolving slate video ids to human-readable titles."""

from __future__ import annotations

from common.title_index import TitleResolver as _BaseTitleResolver

from .config import DEFAULT_TITLE_DIRS


class TitleResolver(_BaseTitleResolver):
    """Resolve YouTube ids to titles using shared common helpers."""

    def __init__(self, **kwargs) -> None:
        defaults = {"default_dirs": DEFAULT_TITLE_DIRS}
        defaults.update(kwargs)
        super().__init__(**defaults)
