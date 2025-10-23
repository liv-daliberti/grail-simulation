"""Utilities for resolving slate video ids to human-readable titles."""

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
