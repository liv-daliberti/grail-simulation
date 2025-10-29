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

"""Shared data structures used by the KNN opinion baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from common.opinion import OpinionExample as BaseOpinionExample
from common.text.embeddings import SentenceTransformerConfig

from .features import Word2VecConfig


@dataclass
class OpinionExample(BaseOpinionExample):  # pylint: disable=too-many-instance-attributes
    """
    Collapsed participant-level prompt and opinion values with session context.

    :ivar step_index: Interaction step index retained from the raw dataset.
    :vartype step_index: int
    :ivar session_id: Session identifier associated with the participant example.
    :vartype session_id: Optional[str]
    """

    step_index: int
    session_id: Optional[str]


@dataclass
class OpinionIndex:  # pylint: disable=too-many-instance-attributes
    """
    Vectorised training corpus, cached targets, and fitted neighbour index.

    :ivar feature_space: Feature space identifier (tfidf, word2vec, sentence_transformer).
    :vartype feature_space: str
    :ivar metric: Distance metric used by the KNN index.
    :vartype metric: str
    :ivar matrix: Document-term matrix or embedding matrix backing the index.
    :vartype matrix: Any
    :ivar vectorizer: Trained vectoriser or embedding builder.
    :vartype vectorizer: Any
    :ivar embeds: Optional Word2Vec feature builder used for inference.
    :vartype embeds: Optional[Any]
    :ivar targets_after: Post-study opinion targets aligned with the corpus.
    :vartype targets_after: numpy.ndarray
    :ivar targets_before: Pre-study opinion targets aligned with the corpus.
    :vartype targets_before: numpy.ndarray
    :ivar participant_keys: Participant/study identifiers aligned with targets.
    :vartype participant_keys: List[Tuple[str, str]]
    :ivar neighbors: Fitted :class:`sklearn.neighbors.NearestNeighbors` index.
    :vartype neighbors: NearestNeighbors
    """

    feature_space: str
    metric: str
    matrix: Any
    vectorizer: Any
    embeds: Optional[Any]
    targets_after: np.ndarray
    targets_before: np.ndarray
    participant_keys: List[Tuple[str, str]]
    neighbors: NearestNeighbors


@dataclass(frozen=True)
class OpinionEmbeddingConfigs:
    """Optional embedding builders available during evaluation."""

    word2vec: Optional[Word2VecConfig]
    sentence_transformer: Optional[SentenceTransformerConfig]


@dataclass(frozen=True)
class OpinionEvaluationContext:
    """Shared configuration for evaluating a single opinion study."""

    args: Any
    dataset: Mapping[str, Sequence[Any]]
    extra_fields: Sequence[str]
    k_values: Sequence[int]
    feature_space: str
    embedding_configs: OpinionEmbeddingConfigs
    outputs_root: Path


__all__ = [
    "OpinionEmbeddingConfigs",
    "OpinionEvaluationContext",
    "OpinionExample",
    "OpinionIndex",
]
