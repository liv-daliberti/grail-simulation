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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from common.prompt_docs import (
    DEFAULT_TITLE_DIRS as _DEFAULT_TITLE_DIRS,
    create_prompt_document_builder,
)

from .data import PROMPT_COLUMN, PROMPT_MAX_HISTORY, SOLUTION_COLUMN

DEFAULT_TITLE_DIRS = _DEFAULT_TITLE_DIRS

_PROMPT_DOC_BUILDER = create_prompt_document_builder(
    prompt_column=PROMPT_COLUMN,
    solution_column=SOLUTION_COLUMN,
    max_history=PROMPT_MAX_HISTORY,
    log_prefix="[prompts]",
    logger_name="xgb.features",
)


def title_for(video_id: str) -> Optional[str]:
    """
    Look up a human-readable title for a given YouTube identifier.

    :param video_id: Candidate YouTube video identifier to resolve.
    :type video_id: str
    :returns: Title associated with ``video_id`` when present in the cache.
    :rtype: Optional[str]
    """

    return _PROMPT_DOC_BUILDER.title_for(video_id)


def viewer_profile_sentence(example: dict) -> str:
    """
    Compose the viewer profile sentence associated with a dataset row.

    :param example: Interaction example containing viewer profile fields.
    :type example: dict
    :returns: Natural-language sentence describing the participant profile.
    :rtype: str
    """

    return _PROMPT_DOC_BUILDER.viewer_profile_sentence(example)


def prompt_from_builder(example: dict) -> str:
    """
    Assemble the full prompt text for a dataset example.

    :param example: Interaction example containing prompt components.
    :type example: dict
    :returns: Prompt text used for feature extraction.
    :rtype: str
    """

    return _PROMPT_DOC_BUILDER.prompt_from_builder(example)


def extract_now_watching(example: dict) -> Optional[Tuple[str, str]]:
    """
    Retrieve the currently watched item for an interaction.

    :param example: Interaction example describing viewer activity.
    :type example: dict
    :returns: Tuple of ``(title, video_id)`` when the "now watching" context exists.
    :rtype: Optional[Tuple[str, str]]
    """

    return _PROMPT_DOC_BUILDER.extract_now_watching(example)


def extract_slate_items(example: dict) -> List[Tuple[str, str]]:
    """
    Extract the slate of candidate items from an interaction example.

    :param example: Dataset row containing slate metadata.
    :type example: dict
    :returns: Ordered list of ``(title, video_id)`` tuples.
    :rtype: List[Tuple[str, str]]
    """

    return _PROMPT_DOC_BUILDER.extract_slate_items(example)


def assemble_document(example: dict, extra_fields: Sequence[str] | None = None) -> str:
    """
    Concatenate prompt components into a single document for featurisation.

    :param example: Dataset row describing the prompt and slate context.
    :type example: dict
    :param extra_fields: Optional additional column names appended to the document.
    :type extra_fields: Sequence[str] | None
    :returns: Combined prompt text passed to vectorisers.
    :rtype: str
    """

    return _PROMPT_DOC_BUILDER.assemble_document(example, extra_fields)


def prepare_training_documents(
    train_ds,
    max_train: int,
    seed: int,
    extra_fields: Sequence[str] | None = None,
) -> Tuple[list[str], list[str], list[str]]:
    """
    Prepare prompt documents, labels, and metadata for TF-IDF training.

    :param train_ds: Dataset split providing training rows.
    :type train_ds: datasets.Dataset | Sequence[dict]
    :param max_train: Optional cap on the number of training rows retained (0 keeps all).
    :type max_train: int
    :param seed: Random seed used when subsampling ``train_ds``.
    :type seed: int
    :param extra_fields: Additional columns appended to the prompt document.
    :type extra_fields: Sequence[str] | None
    :returns: Tuple of ``(documents, label_ids, participant_ids)``.
    :rtype: Tuple[list[str], list[str], list[str]]
    """

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

    :param train_ds: Dataset split providing training rows.
    :type train_ds: datasets.Dataset | Sequence[dict]
    :param max_train: Optional cap on the number of training rows retained (0 keeps all).
    :type max_train: int
    :param seed: Random seed used when subsampling ``train_ds``.
    :type seed: int
    :param extra_fields: Additional columns appended to the prompt document.
    :type extra_fields: Sequence[str] | None
    :returns: Tuple of ``(documents, label_ids, participant_ids)`` mirroring
        :func:`prepare_training_documents`.
    :rtype: Tuple[list[str], list[str], list[str]]
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
