"""Helpers for constructing consistent argparse interfaces."""

from __future__ import annotations

from typing import Iterable, Sequence

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
    """Register an argument that captures comma-separated values.

    The helper keeps flag combinations consistent across CLIs that surface the
    same semantics (e.g., study filters).
    """

    parser.add_argument(
        *_normalise_flags(flags),
        default=default,
        dest=dest,
        help=help_text,
    )


def add_sentence_transformer_normalise_flags(
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
    parser.set_defaults(**{dest: default})


__all__ = [
    "add_comma_separated_argument",
    "add_sentence_transformer_normalise_flags",
]
