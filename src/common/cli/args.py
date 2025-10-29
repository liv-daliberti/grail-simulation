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

"""Reusable argument helpers for pipeline command-line interfaces."""

from __future__ import annotations

from typing import Sequence

import argparse


def _normalise_flags(flags: Sequence[str] | str) -> Sequence[str]:
    """Return ``flags`` as a tuple while accepting single-string inputs."""
    if isinstance(flags, str):
        return (flags,)
    return tuple(flags)


def add_comma_separated_argument(
    parser: argparse.ArgumentParser,
    *,
    flags: Sequence[str] | str,
    dest: str,
    help_text: str,
    default: str = "",
) -> None:
    """Register a comma-separated argument with consistent flag aliases."""
    parser.add_argument(
        *_normalise_flags(flags),
        default=default,
        dest=dest,
        help=help_text,
    )


def add_sentence_transformer_normalise_flags(  # pylint: disable=too-many-arguments
    parser: argparse.ArgumentParser,
    *,
    dest: str = "sentence_transformer_normalize",
    default: bool = True,
    enable_flags: Sequence[str] | str = ("--sentence-transformer-normalize",),
    disable_flags: Sequence[str] | str = ("--sentence-transformer-no-normalize",),
    enable_help: str = "Enable L2-normalisation for sentence-transformer embeddings (default).",
    disable_help: str = "Disable L2-normalisation for sentence-transformer embeddings.",
) -> None:
    """Add paired boolean flags controlling sentence-transformer normalisation."""
    parser.set_defaults(**{dest: default})
    parser.add_argument(
        *_normalise_flags(enable_flags),
        dest=dest,
        action="store_true",
        help=enable_help,
    )
    parser.add_argument(
        *_normalise_flags(disable_flags),
        dest=dest,
        action="store_false",
        help=disable_help,
    )
__all__ = [
    "add_comma_separated_argument",
    "add_sentence_transformer_normalise_flags",
]
