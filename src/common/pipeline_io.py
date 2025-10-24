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

import json
from pathlib import Path
from typing import Mapping, Sequence


def load_metrics_json(path: Path) -> Mapping[str, object]:
    """

    Load a JSON metrics file and return its contents.



    :param path: Value provided for ``path``.

    :type path: Path

    :returns: Result produced by ``load_metrics_json``.

    :rtype: Mapping[str, object]

    """


    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_markdown_lines(path: Path, lines: Sequence[str]) -> None:
    """

    Persist ``lines`` as a Markdown document with a trailing newline.



    :param path: Value provided for ``path``.

    :type path: Path

    :param lines: Value provided for ``lines``.

    :type lines: Sequence[str]

    :returns: ``None``.

    :rtype: None

    """


    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


__all__ = ["load_metrics_json", "write_markdown_lines"]
