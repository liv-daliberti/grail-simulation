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


@dataclass(frozen=True)
class SessionInfo:
    """Optional session context attached to an opinion example."""

    step_index: int
    session_id: Optional[str]


@dataclass
class OpinionExample(BaseOpinionExample):
    """
    Collapsed participant-level prompt and opinion values with session context.

    :ivar session: Optional tuple of step index and session identifier.
    :vartype session: Optional[~knn.core.opinion_models.SessionInfo]
    """
    session: Optional[SessionInfo] = None

    @property
    def step_index(self) -> int:
        """Return the recorded step index or ``-1`` when unavailable."""
        return self.session.step_index if self.session is not None else -1

    @property
    def session_id(self) -> Optional[str]:
        """Return the recorded session identifier when available."""
        return self.session.session_id if self.session is not None else None


@dataclass
class OpinionTargets:
    """Aligned opinion targets and participant keys used during evaluation."""

    after: np.ndarray
    before: np.ndarray
    participant_keys: List[Tuple[str, str]]


@dataclass
class OpinionIndex:
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
    :ivar targets: Grouped post/pre opinion targets and participant keys aligned with the corpus.
    :vartype targets: ~knn.core.opinion_models.OpinionTargets
    :ivar neighbors: Fitted :class:`sklearn.neighbors.NearestNeighbors` index.
    :vartype neighbors: NearestNeighbors
    """

    feature_space: str
    metric: str
    matrix: Any
    vectorizer: Any
    embeds: Optional[Any]
    targets: OpinionTargets
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


_EXPORTS = (
    "OpinionEmbeddingConfigs",
    "OpinionEvaluationContext",
    "OpinionExample",
    "OpinionIndex",
    "SessionInfo",
    "OpinionTargets",
)

# Use a dynamic filter so static analyzers don't flag names during partial builds.
__all__ = [name for name in _EXPORTS if name in globals()]
