"""Helper utilities shared across KNN CLI entry points."""
from __future__ import annotations

from argparse import ArgumentParser
from typing import Iterable

from common.cli_args import add_sentence_transformer_normalise_flags

def add_sentence_transformer_normalize_flags(
    parser: ArgumentParser,
    *,
    legacy_aliases: bool = False,
    help_prefix: str = "",
) -> None:
    """
    Attach normalisation toggle flags for sentence-transformer embeddings to a parser.

    :param parser: Argument parser receiving the flags that control embedding normalisation.
    :type parser: argparse.ArgumentParser
    :param legacy_aliases: Register underscore-separated aliases to maintain compatibility with
        pre-refactor invocations.
    :type legacy_aliases: bool
    :param help_prefix: Optional text prepended to the generated help strings to provide context.
    :type help_prefix: str
    :returns: ``None``. Flags are added to ``parser`` in-place.
    :rtype: None
    """
    normalize_flags: Iterable[str] = ["--sentence-transformer-normalize"]
    no_normalize_flags: Iterable[str] = ["--sentence-transformer-no-normalize"]
    if legacy_aliases:
        normalize_flags = [*normalize_flags, "--sentence_transformer_normalize"]
        no_normalize_flags = [*no_normalize_flags, "--sentence_transformer_no_normalize"]

    help_prefix = f"{help_prefix.strip()} " if help_prefix else ""
    add_sentence_transformer_normalise_flags(
        parser,
        dest="sentence_transformer_normalize",
        enable_flags=tuple(normalize_flags),
        disable_flags=tuple(no_normalize_flags),
        enable_help=(
            f"{help_prefix}Enable L2-normalisation for sentence-transformer "
            "embeddings (default)."
        ),
        disable_help=f"{help_prefix}Disable L2-normalisation for sentence-transformer embeddings.",
    )

__all__ = ["add_sentence_transformer_normalize_flags"]
