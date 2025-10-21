"""Feature extraction helpers for the XGBoost baseline."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from common import get_logger
from common.prompt_docs import (
    DEFAULT_TITLE_DIRS as _DEFAULT_TITLE_DIRS,
    PromptDocumentBuilder,
    default_title_resolver,
)

from .data import PROMPT_COLUMN, PROMPT_MAX_HISTORY, SOLUTION_COLUMN

DEFAULT_TITLE_DIRS = _DEFAULT_TITLE_DIRS

_PROMPT_DOC_BUILDER = PromptDocumentBuilder(
    prompt_column=PROMPT_COLUMN,
    solution_column=SOLUTION_COLUMN,
    max_history=PROMPT_MAX_HISTORY,
    title_lookup=default_title_resolver(),
    log_prefix="[prompts]",
    logger=get_logger("xgb.features"),
)


def title_for(video_id: str) -> Optional[str]:
    """Return a human-readable title for a YouTube id if available."""

    return _PROMPT_DOC_BUILDER.title_for(video_id)


def viewer_profile_sentence(example: dict) -> str:
    """Return the viewer profile sentence for ``example``."""

    return _PROMPT_DOC_BUILDER.viewer_profile_sentence(example)


def prompt_from_builder(example: dict) -> str:
    """Return the prompt text for ``example`` using the shared builder."""

    return _PROMPT_DOC_BUILDER.prompt_from_builder(example)


def extract_now_watching(example: dict) -> Optional[Tuple[str, str]]:
    """Return the currently watched title/id, if known."""

    return _PROMPT_DOC_BUILDER.extract_now_watching(example)


def extract_slate_items(example: dict) -> List[Tuple[str, str]]:
    """Return the slate as ``(title, video_id)`` pairs."""

    return _PROMPT_DOC_BUILDER.extract_slate_items(example)


def assemble_document(example: dict, extra_fields: Sequence[str] | None = None) -> str:
    """Return concatenated text used to featurise ``example``."""

    return _PROMPT_DOC_BUILDER.assemble_document(example, extra_fields)


def prepare_training_documents(
    train_ds,
    max_train: int,
    seed: int,
    extra_fields: Sequence[str] | None = None,
) -> Tuple[list[str], list[str], list[str]]:
    """Return TF-IDF training documents and associated labels."""

    return _PROMPT_DOC_BUILDER.prepare_training_documents(
        train_ds,
        max_train,
        seed,
        extra_fields,
    )


def prepare_prompt_documents(
    train_ds,
    max_train: int,
    seed: int,
    extra_fields: Sequence[str] | None = None,
) -> Tuple[list[str], list[str], list[str]]:
    """
    Convenience wrapper around :func:`prepare_training_documents`.
    """

    return prepare_training_documents(
        train_ds,
        max_train=max_train,
        seed=seed,
        extra_fields=extra_fields,
    )


__all__ = [
    "DEFAULT_TITLE_DIRS",
    "assemble_document",
    "extract_now_watching",
    "extract_slate_items",
    "prepare_prompt_documents",
    "prepare_training_documents",
    "prompt_from_builder",
    "title_for",
    "viewer_profile_sentence",
]
