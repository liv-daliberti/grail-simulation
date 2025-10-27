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

"""Prompt feature construction utilities shared by the KNN baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from gensim.models import Word2Vec  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Word2Vec = None

from common.prompt_docs import (
    DEFAULT_TITLE_DIRS as _DEFAULT_TITLE_DIRS,
    EXTRA_FIELD_LABELS as _COMMON_EXTRA_FIELD_LABELS,
    create_prompt_document_builder,
)
from common.prompt_selection import (
    CandidateMetadata,
    PROMPT_SELECTION_EXPORT_ATTRS,
    PromptSelectionHelper,
)

from .data import PROMPT_COLUMN, PROMPT_MAX_HISTORY, SOLUTION_COLUMN

DEFAULT_TITLE_DIRS = _DEFAULT_TITLE_DIRS
EXTRA_FIELD_LABELS = _COMMON_EXTRA_FIELD_LABELS

_PROMPT_DOC_BUILDER = create_prompt_document_builder(
    prompt_column=PROMPT_COLUMN,
    solution_column=SOLUTION_COLUMN,
    max_history=PROMPT_MAX_HISTORY,
    log_prefix="[KNN]",
    logger_name="knn.features",
)

_PROMPT_FEATURES = PromptSelectionHelper(_PROMPT_DOC_BUILDER)

# Explicit re-exports so static analysers see the helper functions.
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
candidate_feature_tokens: Callable[..., Sequence[str]] = _PROMPT_FEATURES.candidate_feature_tokens


# ---------------------------------------------------------------------------
# TF-IDF + Word2Vec interfaces
# ---------------------------------------------------------------------------

@dataclass
class Word2VecConfig:
    """
    Configuration options for Word2Vec embeddings.

    :ivar vector_size: Dimensionality of the learned embedding vectors.
    :vartype vector_size: int
    :ivar window: Context window size used during training.
    :vartype window: int
    :ivar min_count: Minimum token frequency retained in the vocabulary.
    :vartype min_count: int
    :ivar epochs: Number of Word2Vec training epochs.
    :vartype epochs: int
    :ivar model_dir: Directory where trained embeddings are persisted.
    :vartype model_dir: Path
    :ivar seed: Random seed for Word2Vec initialisation.
    :vartype seed: int
    :ivar workers: Number of worker threads allocated to training.
    :vartype workers: int
    """
    vector_size: int = 256
    window: int = 5
    min_count: int = 2
    epochs: int = 10
    model_dir: Path = Path("models/knn/next_video/word2vec_models")
    seed: int = 42
    workers: int = 1

class Word2VecFeatureBuilder:
    """
    Create Word2Vec embeddings from viewer prompts.

    :ivar config: Configuration bundle controlling Word2Vec training/inference.
    :vartype config: Word2VecConfig
    :ivar _model: Loaded gensim :class:`~gensim.models.Word2Vec` instance.
    :vartype _model: gensim.models.Word2Vec | None
    """
    def __init__(self, config: Optional[Word2VecConfig] = None) -> None:
        """
        Initialise the builder with optional configuration overrides.

        :param config: Optional configuration bundle to override defaults.
        :type config: Word2VecConfig | None
        """
        self.config = config or Word2VecConfig()
        self._model = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenise input text into whitespace-delimited lower-case tokens.

        :param text: Text to split into tokens.
        :type text: str
        :returns: List of lower-cased tokens extracted from ``text``.
        :rtype: List[str]
        """
        return text.lower().split()

    def train(self, corpus: Iterable[str]) -> None:
        """
        Train a Word2Vec model using the provided corpus.

        :param corpus: Iterable of documents used to fit the embeddings.
        :type corpus: Iterable[str]
        :raises ImportError: If :mod:`gensim` is unavailable.
        """
        if Word2Vec is None:  # pragma: no cover - optional dependency
            raise ImportError("Install gensim to enable Word2Vec embeddings")
        sentences = [self._tokenize(text) for text in corpus]
        self._model = Word2Vec(
            sentences=sentences,
            vector_size=self.config.vector_size,
            window=self.config.window,
            min_count=self.config.min_count,
            sg=1,
            epochs=self.config.epochs,
            seed=self.config.seed,
            workers=self.config.workers,
        )
        self.save(self.config.model_dir)

    def load(self, directory: Path) -> None:
        """
        Load a previously trained Word2Vec model from disk.

        :param directory: Directory containing the saved Word2Vec model.
        :type directory: Path
        :raises ImportError: If :mod:`gensim` is unavailable.
        """
        if Word2Vec is None:  # pragma: no cover - optional dependency
            raise ImportError("Install gensim to enable Word2Vec embeddings")
        self._model = Word2Vec.load(str(directory / "word2vec.model"))

    def save(self, directory: Path) -> None:
        """
        Persist the trained model to ``directory``.

        :param directory: Destination directory receiving the model artefact.
        :type directory: Path
        :raises RuntimeError: If the model has not been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError("Word2Vec model must be trained before saving")
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save(str(directory / "word2vec.model"))

    def is_trained(self) -> bool:
        """
        Indicate whether the underlying Word2Vec model is ready for inference.

        :returns: ``True`` when a trained model is loaded.
        :rtype: bool
        """
        return self._model is not None

    def encode(self, text: str) -> np.ndarray:
        """
        Return the averaged embedding vector for ``text``.

        :param text: Document to encode using the fitted embeddings.
        :type text: str
        :returns: Averaged embedding vector representing ``text``.
        :rtype: numpy.ndarray
        :raises RuntimeError: If the model has not been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError("Word2Vec model has not been trained/loaded")
        tokens = [token for token in self._tokenize(text) if token in self._model.wv]
        if not tokens:
            return np.zeros(self._model.vector_size, dtype=np.float32)
        return np.asarray(self._model.wv[tokens].mean(axis=0), dtype=np.float32)

    def transform(self, corpus: Sequence[str]) -> np.ndarray:
        """
        Return stacked embeddings for ``corpus``.

        :param corpus: Sequence of documents to transform.
        :type corpus: Sequence[str]
        :returns: Matrix of shape ``(len(corpus), vector_size)``.
        :rtype: numpy.ndarray
        :raises RuntimeError: If the model has not been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError("Word2Vec model has not been trained/loaded")
        if not corpus:
            return np.zeros((0, self._model.vector_size), dtype=np.float32)
        vectors = [self.encode(text) for text in corpus]
        return np.vstack(vectors).astype(np.float32)

__all__ = [
    "DEFAULT_TITLE_DIRS",
    "EXTRA_FIELD_LABELS",
    "CandidateMetadata",
    "Word2VecConfig",
    "Word2VecFeatureBuilder",
    *PROMPT_SELECTION_EXPORT_ATTRS,
    "candidate_feature_tokens",
]
