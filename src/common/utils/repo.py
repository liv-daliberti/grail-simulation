#!/usr/bin/env python
"""Utilities for resolving repository paths with optional monkeypatch support."""

from __future__ import annotations

import os
from pathlib import Path
import sys as _sys


def resolve_repo_root_from_monkeypatch(module_name: str, default: Path) -> Path:
    """Resolve a repo root honoring a ``_repo_root`` monkeypatch on ``module_name``.

    Test suites sometimes set ``<module>._repo_root`` to control where outputs
    are written. This helper checks the loaded module for either a callable
    or path-like override and falls back to ``default`` when absent.

    Args:
        module_name: Fully-qualified module name to inspect (e.g. "grpo.pipeline").
        default: Default path to return when no monkeypatch is present.

    Returns:
        Path: The patched repo root if available, otherwise ``default``.
    """

    mod = _sys.modules.get(module_name)
    if mod is not None:
        patched = getattr(mod, "_repo_root", None)
        if callable(patched):
            return Path(patched())
        if isinstance(patched, (str, os.PathLike)):
            return Path(patched)
    return default


__all__ = ["resolve_repo_root_from_monkeypatch"]
