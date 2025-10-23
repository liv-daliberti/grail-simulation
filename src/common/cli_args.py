"""Helpers for constructing consistent argparse interfaces."""

from __future__ import annotations

from typing import Sequence

import argparse


def _normalise_flags(flags: Sequence[str] | str) -> Sequence[str]:
    """

    Return ``flags`` as a tuple while accepting single-string inputs.



    :param flags: Value provided for ``flags``.

    :type flags: Sequence[str] | str

    :returns: Result produced by ``_normalise_flags``.

    :rtype: Sequence[str]

    """


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
    """

    Register an argument that captures comma-separated values.



        The helper keeps flag combinations consistent across CLIs that surface the

        same semantics (e.g., study filters).



    :param parser: Value provided for ``parser``.

    :type parser: argparse.ArgumentParser

    :param flags: Value provided for ``flags``.

    :type flags: Sequence[str] | str

    :param dest: Value provided for ``dest``.

    :type dest: str

    :param help_text: Value provided for ``help_text``.

    :type help_text: str

    :param default: Value provided for ``default``.

    :type default: str

    :returns: ``None``.

    :rtype: None

    """


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
    """

    Add paired boolean flags controlling sentence-transformer normalisation.



    :param parser: Value provided for ``parser``.

    :type parser: argparse.ArgumentParser

    :param dest: Value provided for ``dest``.

    :type dest: str

    :param default: Value provided for ``default``.

    :type default: bool

    :param enable_flags: Value provided for ``enable_flags``.

    :type enable_flags: Sequence[str] | str

    :param disable_flags: Value provided for ``disable_flags``.

    :type disable_flags: Sequence[str] | str

    :param enable_help: Value provided for ``enable_help``.

    :type enable_help: str

    :param disable_help: Value provided for ``disable_help``.

    :type disable_help: str

    :returns: ``None``.

    :rtype: None

    """


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
