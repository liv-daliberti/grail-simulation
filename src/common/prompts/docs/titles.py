"""Title resolution utilities for cleaned GRAIL datasets.

This module centralises logic for discovering on-disk title indexes and
instantiating :class:`common.text.TitleResolver` with sensible defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from common.text import TitleResolver

TitleLookup = Callable[[Optional[str]], Optional[str]]

_TITLE_INDEX_ROOT = (
    "/n/fs/similarity/trees/data/results/"
    "capsule-5416997-data/recommendation trees"
)


def _default_title_dirs() -> list[str]:
    """
    Return plausible default directories for title CSVs.

    :returns: Ordered list of candidate directories containing title indexes.
    :rtype: list[str]
    """
    # Absolute (cluster) locations used in shared environments.
    dirs: list[str] = [
        f"{_TITLE_INDEX_ROOT}/trees_gun",
        f"{_TITLE_INDEX_ROOT}/trees_wage",
    ]
    # Repo-local fallback: capsule-5416997/data/recommendation trees/trees_*/
    try:
        repo_root = Path(__file__).resolve().parents[3]
        local_base = repo_root / "capsule-5416997" / "data" / "recommendation trees"
        dirs.append(str(local_base / "trees_gun"))
        dirs.append(str(local_base / "trees_wage"))
    except (OSError, RuntimeError, IndexError):  # pragma: no cover - defensive fallback
        # Resolution/parent traversal may fail on unusual filesystems or layouts.
        pass
    # De-duplicate while preserving order.
    ordered: list[str] = []
    seen: set[str] = set()
    for path in dirs:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


# Materialize at import time so other modules can re-export a concrete list.
DEFAULT_TITLE_DIRS = _default_title_dirs()

_TITLE_RESOLVER_CACHE: TitleResolver | None = None


def default_title_resolver() -> TitleResolver:
    """
    Return a lazily constructed :class:`~common.text.title_index.TitleResolver`
    for GRAIL datasets.

    :returns: Shared :class:`~common.text.title_index.TitleResolver` instance
        backed by default paths.
    :rtype: ~common.text.title_index.TitleResolver
    """
    global _TITLE_RESOLVER_CACHE  # pylint: disable=global-statement
    if _TITLE_RESOLVER_CACHE is None:
        _TITLE_RESOLVER_CACHE = TitleResolver(default_dirs=DEFAULT_TITLE_DIRS)
    return _TITLE_RESOLVER_CACHE


__all__ = ["DEFAULT_TITLE_DIRS", "TitleLookup", "default_title_resolver"]
