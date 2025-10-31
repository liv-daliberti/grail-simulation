#!/usr/bin/env python
"""Shared CLI argument helpers for the XGBoost baseline.

This module centralises common argument definitions that were previously
duplicated across ``xgb.cli.main`` and ``xgb.pipeline.cli``.
"""

from __future__ import annotations

import argparse

from common.cli.args import add_sentence_transformer_normalise_flags


def add_word2vec_args(parser: argparse.ArgumentParser, *, style: str = "hyphen") -> None:
    """Add Word2Vec-related CLI arguments to ``parser``.

    :param parser: Argument parser to extend.
    :param style: Flag style to use: ``"hyphen"`` (``--word2vec-size``) or
        ``"underscore"`` (``--word2vec_size``).
    :returns: ``None``.
    """

    if style not in {"hyphen", "underscore"}:
        raise ValueError("style must be 'hyphen' or 'underscore'")

    def flags(name: str) -> tuple[str, str] | tuple[str]:
        hyphen = name.replace("_", "-")
        underscore = name
        # Always expose both aliases to remain backward compatible across CLIs.
        return (hyphen, underscore)

    # Keep dest names stable regardless of flag style for downstream consumers.
    parser.add_argument(
        *flags("--word2vec_size"),
        type=int,
        default=256,
        dest="word2vec_size",
        help=(
            "Word2Vec vector dimensionality when using the "
            "word2vec feature space."
        ),
    )
    parser.add_argument(
        *flags("--word2vec_window"),
        type=int,
        default=5,
        dest="word2vec_window",
        help="Window size used during Word2Vec training.",
    )
    parser.add_argument(
        *flags("--word2vec_min_count"),
        type=int,
        default=2,
        dest="word2vec_min_count",
        help=(
            "Minimum token frequency retained in the Word2Vec vocabulary."
        ),
    )
    parser.add_argument(
        *flags("--word2vec_epochs"),
        type=int,
        default=10,
        dest="word2vec_epochs",
        help="Number of Word2Vec training epochs.",
    )
    parser.add_argument(
        *flags("--word2vec_workers"),
        type=int,
        default=1,
        dest="word2vec_workers",
        help="Worker threads allocated to Word2Vec training.",
    )
    parser.add_argument(
        *flags("--word2vec_model_dir"),
        default="",
        dest="word2vec_model_dir",
        help="Directory to persist or reuse trained Word2Vec models (optional).",
    )


def add_sentence_transformer_args(
    parser: argparse.ArgumentParser,
    *,
    style: str = "hyphen",
    normalize_default: bool = True,
) -> None:
    """Add SentenceTransformer-related CLI arguments to ``parser``.

    :param parser: Argument parser to extend.
    :param style: Flag style to use: ``"hyphen"`` (``--sentence-transformer-…``) or
        ``"underscore"`` (``--sentence_transformer_…``).
    :param normalize_default: Default for the L2-normalisation toggle.
    :returns: ``None``.
    """

    if style not in {"hyphen", "underscore"}:
        raise ValueError("style must be 'hyphen' or 'underscore'")

    def flags(name: str) -> tuple[str, str] | tuple[str]:
        hyphen = name.replace("_", "-")
        underscore = name
        return (hyphen, underscore)

    parser.add_argument(
        *flags("--sentence_transformer_model"),
        default="sentence-transformers/all-mpnet-base-v2",
        dest="sentence_transformer_model",
        help=(
            "SentenceTransformer model identifier used when evaluating the "
            "sentence_transformer feature space."
        ),
    )
    parser.add_argument(
        *flags("--sentence_transformer_device"),
        default=None if style == "underscore" else "",
        dest="sentence_transformer_device",
        help="PyTorch device string forwarded to SentenceTransformer (e.g. cpu, cuda).",
    )
    parser.add_argument(
        *flags("--sentence_transformer_batch_size"),
        type=int,
        default=32,
        dest="sentence_transformer_batch_size",
        help="Batch size applied during sentence-transformer encoding.",
    )

    # Normalisation flags need alias flexibility to match existing CLIs.
    # Expose both underscore and hyphenated variants of the normalisation flags.
    add_sentence_transformer_normalise_flags(
        parser,
        dest="sentence_transformer_normalize",
        default=normalize_default,
        enable_flags=("--sentence-transformer-normalize", "--sentence_transformer_normalize"),
        disable_flags=(
            "--sentence-transformer-no-normalize",
            "--sentence_transformer_no_normalize",
        ),
        enable_help="L2-normalise sentence-transformer embeddings (default).",
        disable_help="Disable L2-normalisation for sentence-transformer embeddings.",
    )


__all__ = [
    "add_word2vec_args",
    "add_sentence_transformer_args",
]
