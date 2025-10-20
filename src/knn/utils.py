"""Miscellaneous utilities shared across KNN modules."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "knn") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


__all__ = ["ensure_directory", "get_logger"]
