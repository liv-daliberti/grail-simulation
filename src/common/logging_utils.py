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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """



        Return a memoised logger configured with a simple stream handler.



        The helper mirrors the historical behaviour from the individual

        baselines: create a ``logging.Logger``, attach a stream handler

        when none are registered yet, and set the level to ``INFO`` (or a

        caller-provided override).



    :param name: Value provided for ``name``.

    :type name: str

    :param level: Value provided for ``level``.

    :type level: int

    :returns: Result produced by ``get_logger``.

    :rtype: logging.Logger

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



    :param path: Value provided for ``path``.

    :type path: Path | str

    :returns: ``None``.

    :rtype: None

    """


    Path(path).mkdir(parents=True, exist_ok=True)


__all__ = ["ensure_directory", "get_logger"]
