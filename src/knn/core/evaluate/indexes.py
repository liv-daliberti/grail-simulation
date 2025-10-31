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

"""Index construction and loading helpers for the KNN evaluator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.text.embeddings import (
    SentenceTransformerConfig,
    sentence_transformer_config_from_args,
)

from ..features import Word2VecConfig
from ..index import (
    SlateQueryConfig,
    build_sentence_transformer_index,
    build_tfidf_index,
    build_word2vec_index,
    load_sentence_transformer_index,
    load_tfidf_index,
    load_word2vec_index,
    save_sentence_transformer_index,
    save_tfidf_index,
    save_word2vec_index,
)


def normalise_feature_space(feature_space: str | None) -> str:
    """
    Return the validated feature space identifier.

    :param feature_space: Raw feature-space string supplied via CLI.
    :returns: Lowercase feature-space token (``tfidf``, ``word2vec``, or ``sentence_transformer``).
    :raises ValueError: If an unsupported feature space is supplied.
    """

    value = (feature_space or "tfidf").lower()
    if value not in {"tfidf", "word2vec", "sentence_transformer"}:
        raise ValueError(f"Unsupported feature space '{feature_space}'")
    return value


def word2vec_config_from_args(args, issue_slug: str) -> Word2VecConfig:
    """
    Return the Word2Vec configuration derived from CLI arguments.

    :param args: Parsed CLI namespace containing Word2Vec options.
    :param issue_slug: Current issue being processed (used to namespace models).
    :returns: Populated :class:`~knn.core.features.Word2VecConfig` instance.
    """

    default_cfg = Word2VecConfig()
    model_root = Path(args.word2vec_model_dir) if args.word2vec_model_dir else default_cfg.model_dir
    return Word2VecConfig(
        vector_size=int(args.word2vec_size),
        window=int(getattr(args, "word2vec_window", default_cfg.window)),
        min_count=int(getattr(args, "word2vec_min_count", default_cfg.min_count)),
        epochs=int(getattr(args, "word2vec_epochs", default_cfg.epochs)),
        model_dir=Path(model_root) / issue_slug,
        seed=int(getattr(args, "knn_seed", default_cfg.seed)),
        workers=int(getattr(args, "word2vec_workers", default_cfg.workers)),
    )


def fit_index_for_issue(
    *,
    feature_space: str,
    train_ds,
    issue_slug: str,
    extra_fields: Sequence[str],
    args,
) -> Mapping[str, Any]:
    """
    Build an index for the requested feature space and handle persistence.

    :param feature_space: ``tfidf``, ``word2vec``, or ``sentence_transformer``.
    :param train_ds: Training split (Hugging Face dataset slice).
    :param issue_slug: Normalised issue identifier.
    :param extra_fields: Optional text fields to concatenate into documents.
    :param args: CLI namespace for additional parameters.
    :returns: Dictionary describing the fitted index artefacts.
    :raises ValueError: If the requested feature space is unsupported.
    """

    if feature_space == "tfidf":
        logging.info("[KNN] Building TF-IDF index for issue=%s", issue_slug)
        index = build_tfidf_index(
            train_ds,
            max_train=args.knn_max_train,
            seed=args.knn_seed,
            max_features=None,
            extra_fields=extra_fields,
        )
        if args.save_index:
            save_tfidf_index(index, Path(args.save_index) / issue_slug)
        return index

    if feature_space == "word2vec":
        logging.info("[KNN] Building Word2Vec index for issue=%s", issue_slug)
        config = word2vec_config_from_args(args, issue_slug)
        index = build_word2vec_index(
            train_ds,
            max_train=args.knn_max_train,
            seed=args.knn_seed,
            extra_fields=extra_fields,
            config=config,
        )
        if args.save_index:
            save_word2vec_index(index, Path(args.save_index) / issue_slug)
        return index

    if feature_space == "sentence_transformer":
        logging.info("[KNN] Building SentenceTransformer index for issue=%s", issue_slug)
        config: SentenceTransformerConfig = sentence_transformer_config_from_args(args)
        index = build_sentence_transformer_index(
            train_ds,
            max_train=args.knn_max_train,
            seed=args.knn_seed,
            extra_fields=extra_fields,
            config=config,
        )
        if args.save_index:
            save_sentence_transformer_index(index, Path(args.save_index) / issue_slug)
        return index

    raise ValueError(f"Unsupported feature space '{feature_space}'")


def load_index_for_issue(
    *,
    feature_space: str,
    issue_slug: str,
    args,
) -> Mapping[str, Any]:
    """
    Load a persisted index for the requested feature space.

    :param feature_space: ``tfidf``, ``word2vec``, or ``sentence_transformer``.
    :param issue_slug: Normalised issue identifier.
    :param args: CLI namespace providing the ``--load-index`` directory.
    :returns: Dictionary with the loaded index artefacts.
    :raises ValueError: If the feature space is not recognised.
    """

    load_path = Path(args.load_index) / issue_slug
    if feature_space == "tfidf":
        logging.info("[KNN] Loading TF-IDF index for issue=%s", issue_slug)
        return load_tfidf_index(load_path)
    if feature_space == "word2vec":
        logging.info("[KNN] Loading Word2Vec index for issue=%s", issue_slug)
        return load_word2vec_index(load_path)
    if feature_space == "sentence_transformer":
        logging.info("[KNN] Loading SentenceTransformer index for issue=%s", issue_slug)
        return load_sentence_transformer_index(load_path)
    raise ValueError(f"Unsupported feature space '{feature_space}'")


def build_or_load_index(
    *,
    train_ds,
    issue_slug: str,
    extra_fields: Sequence[str],
    args,
) -> Mapping[str, Any]:
    """
    Return the KNN index for ``issue_slug`` based on CLI arguments.

    :param train_ds: Training split dataset.
    :param issue_slug: Normalised issue identifier.
    :param extra_fields: Optional extra text fields.
    :param args: CLI namespace containing ``--fit-index`` or ``--load-index``.
    :returns: Dictionary describing the fitted or loaded KNN index.
    :raises ValueError: When neither ``--fit-index`` nor ``--load-index`` is used.
    """

    feature_space = normalise_feature_space(getattr(args, "feature_space", None))
    if args.fit_index:
        return fit_index_for_issue(
            feature_space=feature_space,
            train_ds=train_ds,
            issue_slug=issue_slug,
            extra_fields=extra_fields,
            args=args,
        )
    if args.load_index:
        return load_index_for_issue(
            feature_space=feature_space,
            issue_slug=issue_slug,
            args=args,
        )
    raise ValueError("Set either --fit_index or --load_index to obtain a KNN index")


__all__ = [
    "SlateQueryConfig",
    "build_or_load_index",
    "fit_index_for_issue",
    "load_index_for_issue",
    "normalise_feature_space",
    "sentence_transformer_config_from_args",
    "word2vec_config_from_args",
]
