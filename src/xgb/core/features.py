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

"""Prompt feature construction utilities shared by the XGBoost baseline."""

from __future__ import annotations

from common.prompts.docs import (
    DEFAULT_TITLE_DIRS as _DEFAULT_TITLE_DIRS,
    create_prompt_document_builder,
)
from common.prompts.selection import PromptSelectionHelper as _PromptSelectionHelper
from common.prompts.selection import (
    PROMPT_SELECTION_EXPORT_ATTRS as _PROMPT_EXPORTS,
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

_PROMPT_SELECTOR = _PromptSelectionHelper(_PROMPT_DOC_BUILDER)

# Re-export builder-backed helpers that do not inject selection tokens.
title_for = _PROMPT_DOC_BUILDER.title_for
viewer_profile_sentence = _PROMPT_DOC_BUILDER.viewer_profile_sentence
prompt_from_builder = _PROMPT_DOC_BUILDER.prompt_from_builder
extract_now_watching = _PROMPT_DOC_BUILDER.extract_now_watching
extract_slate_items = _PROMPT_DOC_BUILDER.extract_slate_items


def assemble_document(example: dict, extra_fields):  # type: ignore[override]
    """Assemble a prompt document from ``example`` using shared defaults."""

    return _PROMPT_DOC_BUILDER.assemble_document(example, extra_fields)


def prepare_training_documents(
    train_ds,
    max_train: int,
    seed: int,
    extra_fields=None,
):  # type: ignore[override]
    """Prepare training documents and labels from ``train_ds``."""

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
    extra_fields=None,
):  # type: ignore[override]
    """Prepare selection-aware documents using the prompt-selection helper."""

    # Use selection-aware document assembly to mirror KNN behaviour.
    return _PROMPT_SELECTOR.prepare_training_documents(
        train_ds,
        max_train,
        seed,
        extra_fields,
    )

# Build exports from the shared selection export list to avoid duplication.
_CORE_EXPORTS = [name for name in _PROMPT_EXPORTS if name in globals()]
__all__ = ["DEFAULT_TITLE_DIRS", *_CORE_EXPORTS]
