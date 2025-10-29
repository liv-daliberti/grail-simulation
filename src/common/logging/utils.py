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

"""Convenience wrappers for pipeline logging and filesystem setup."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """
    Return a memoised logger configured with a simple stream handler.

    :param name: Name registered on the logger.
    :param level: Log level applied when initialising the logger.
    :returns: Configured logger instance.
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

    :param path: Target directory to ensure.
    :returns: ``None``.
    """

    Path(path).mkdir(parents=True, exist_ok=True)


__all__ = ["ensure_directory", "get_logger"]
