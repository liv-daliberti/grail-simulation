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

from common.prompt_docs import (
    DEFAULT_TITLE_DIRS as _DEFAULT_TITLE_DIRS,
    create_prompt_document_builder,
)
from common.prompt_selection import (
    CandidateMetadata,
    PromptSelectionHelper,
    candidate_feature_tokens,
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

_PROMPT_FEATURES = PromptSelectionHelper(_PROMPT_DOC_BUILDER)

title_for = _PROMPT_FEATURES.title_for
viewer_profile_sentence = _PROMPT_FEATURES.viewer_profile_sentence
prompt_from_builder = _PROMPT_FEATURES.prompt_from_builder
extract_now_watching = _PROMPT_FEATURES.extract_now_watching
extract_slate_items = _PROMPT_FEATURES.extract_slate_items
collect_candidate_metadata = _PROMPT_FEATURES.collect_candidate_metadata
selection_feature_tokens = _PROMPT_FEATURES.selection_feature_tokens
assemble_document = _PROMPT_FEATURES.assemble_document
prepare_training_documents = _PROMPT_FEATURES.prepare_training_documents
prepare_prompt_documents = _PROMPT_FEATURES.prepare_prompt_documents

__all__ = [
    "DEFAULT_TITLE_DIRS",
    "CandidateMetadata",
    "candidate_feature_tokens",
    "collect_candidate_metadata",
    "assemble_document",
    "extract_now_watching",
    "extract_slate_items",
    "prepare_prompt_documents",
    "prepare_training_documents",
    "selection_feature_tokens",
    "prompt_from_builder",
    "title_for",
    "viewer_profile_sentence",
]
