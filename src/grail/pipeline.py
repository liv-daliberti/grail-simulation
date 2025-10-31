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

"""CLI shim that reuses the GRPO evaluation pipeline for GRAIL runs."""

from __future__ import annotations

import sys
from typing import Sequence

from grpo.config import repo_root as _repo_root
from grpo.pipeline import main as _grpo_main

from .reports import DEFAULT_REGENERATE_HINT

__all__ = ["main"]


def _flag_present(argv: Sequence[str], flag: str) -> bool:
    """Return ``True`` when the CLI already specifies ``flag``."""

    flag_eq = f"{flag}="
    return any(token == flag or token.startswith(flag_eq) for token in argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint mirroring :mod:`grpo.pipeline` with GRAIL defaults."""

    user_args = list(argv) if argv is not None else sys.argv[1:]
    extra_args: list[str] = []

    repo_root = _repo_root()
    default_out_dir = str(repo_root / "models" / "grail")

    if not _flag_present(user_args, "--out-dir"):
        extra_args.extend(["--out-dir", default_out_dir])
    if not _flag_present(user_args, "--reports-subdir"):
        extra_args.extend(["--reports-subdir", "grail"])
    if not _flag_present(user_args, "--baseline-label"):
        extra_args.extend(["--baseline-label", "GRAIL"])
    if not _flag_present(user_args, "--regenerate-hint"):
        extra_args.extend(["--regenerate-hint", DEFAULT_REGENERATE_HINT])

    _grpo_main(extra_args + user_args)
