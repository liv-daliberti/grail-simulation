"""Feature extraction helpers for the XGBoost baseline."""

from __future__ import annotations

from typing import Sequence, Tuple

from knn.features import (
    DEFAULT_TITLE_DIRS,
    Word2VecConfig,
    Word2VecFeatureBuilder,
    assemble_document,
    extract_now_watching,
    extract_slate_items,
    prepare_training_documents,
    prompt_from_builder,
    title_for,
    viewer_profile_sentence,
)


def prepare_prompt_documents(
    train_ds,
    max_train: int,
    seed: int,
    extra_fields: Sequence[str] | None = None,
) -> Tuple[list[str], list[str], list[str]]:
    """Convenience wrapper around :func:`knn.features.prepare_training_documents`."""

    return prepare_training_documents(
        train_ds,
        max_train=max_train,
        seed=seed,
        extra_fields=extra_fields,
    )


__all__ = [
    "DEFAULT_TITLE_DIRS",
    "Word2VecConfig",
    "Word2VecFeatureBuilder",
    "assemble_document",
    "extract_now_watching",
    "extract_slate_items",
    "prepare_prompt_documents",
    "prepare_training_documents",
    "prompt_from_builder",
    "title_for",
    "viewer_profile_sentence",
]
