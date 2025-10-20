"""Utility helpers shared across the XGBoost baseline modules."""

from __future__ import annotations

import logging
import re
from pathlib import Path


_YTID_RE = re.compile(r"([A-Za-z0-9_-]{11})")


def get_logger(name: str = "xgb") -> logging.Logger:
    """
    Return a memoised logger configured with a simple stream handler.

    :param name: Logger name to retrieve or create.
    :type name: str
    :returns: Logger instance configured for console output.
    :rtype: logging.Logger
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_directory(path: Path) -> None:
    """
    Create ``path`` (including parents) when it does not already exist.

    :param path: Target directory to ensure.
    :type path: Path
    """

    path.mkdir(parents=True, exist_ok=True)


def canon_video_id(value: object) -> str:
    """Return a canonicalised YouTube id when present in ``value``.

    :param value: Candidate text containing a YouTube identifier.
    :type value: object
    :returns: Canonical 11-character id or the stripped text when no match is found.
    :rtype: str
    """

    if not isinstance(value, str):
        return ""
    match = _YTID_RE.search(value)
    return match.group(1) if match else value.strip()


__all__ = ["canon_video_id", "ensure_directory", "get_logger"]
