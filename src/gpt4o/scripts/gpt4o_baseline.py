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

"""CLI entry point powering the GPT-4o baseline executable.

This thin wrapper preserves the legacy invocation pattern
``python src/gpt4o/scripts/gpt4o_baseline.py`` while deferring to the
canonical CLI under ``gpt4o.cli``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from importlib import import_module


def _ensure_repo_on_path() -> None:
    """Inject the repository root into ``sys.path`` when executed as a script.

    :returns: ``None``. The repository root is inserted when missing.
    """

    if __package__ not in {None, ""}:
        return
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _forward_to_cli() -> None:
    """Ensure path and forward to ``gpt4o.cli:main``.

    Uses :mod:`importlib` to avoid non-top-level imports while preserving
    backwards-compatible script execution.
    """

    _ensure_repo_on_path()
    import_module("gpt4o.cli").main()


if __name__ == "__main__":
    _forward_to_cli()
