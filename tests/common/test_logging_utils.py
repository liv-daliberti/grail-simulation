"""Unit tests for :mod:`common.logging.utils`."""

from __future__ import annotations

import logging
from pathlib import Path

from common.logging.utils import ensure_directory, get_logger


def _reset_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    return logger


def test_get_logger_configures_single_stream_handler() -> None:
    name = "test.common.logging"
    logger = _reset_logger(name)

    configured = get_logger(name)
    assert configured is logger
    assert len(configured.handlers) == 1
    handler = configured.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert configured.level == logging.INFO

    configured_again = get_logger(name)
    assert configured_again is configured
    assert len(configured_again.handlers) == 1  # no duplicate handlers added


def test_ensure_directory_creates_nested_paths(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    assert not target.exists()

    ensure_directory(target)
    assert target.exists() and target.is_dir()

    # Idempotent call should not raise or remove contents
    ensure_directory(target)
    assert target.exists() and target.is_dir()
