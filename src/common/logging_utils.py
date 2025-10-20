"""Logging helpers shared across baseline implementations."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """
    Return a memoised logger configured with a simple stream handler.

    The helper mirrors the historical behaviour from the individual
    baselines: create a :class:`logging.Logger`, attach a stream
    handler when none are registered yet, and set the level to ``INFO``
    (or a caller-provided override).
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def ensure_directory(path: Path | str) -> None:
    """
    Create ``path`` (including parents) when it does not already exist.

    Parameters
    ----------
    path:
        Target directory to ensure.
    """

    Path(path).mkdir(parents=True, exist_ok=True)


__all__ = ["ensure_directory", "get_logger"]
