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

"""Feature-space transformations and neighbour indices for opinion regression."""

from __future__ import annotations

import logging
from typing import Any, Sequence, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from common.opinion import OpinionSpec
from common.text.embeddings import SentenceTransformerConfig, SentenceTransformerEncoder
from common.text.vectorizers import create_tfidf_vectorizer

from .features import Word2VecConfig, Word2VecFeatureBuilder
from .opinion_models import OpinionExample, OpinionIndex, OpinionTargets

LOGGER = logging.getLogger("knn.opinion")


def _build_tfidf_matrix(documents: Sequence[str]) -> Tuple[TfidfVectorizer, Any]:
    """
    Fit a TF-IDF vectoriser using the supplied documents.

    :param documents: Iterable of vectorisable documents consumed by the index.
    :type documents: Sequence[str]
    :returns: Fitted TF-IDF vectoriser and sparse matrix.
    :rtype: Tuple[TfidfVectorizer, Any]
    """
    vectorizer = create_tfidf_vectorizer(max_features=None)
    matrix = vectorizer.fit_transform(documents).astype(np.float32)
    return vectorizer, matrix


@dataclass(frozen=True)
class OpinionIndexConfig:
    """
    Configuration for building an :class:`~knn.core.opinion_models.OpinionIndex`.

    :param feature_space: Feature space identifier (``tfidf``, ``word2vec``,
        or ``sentence_transformer``).
    :param metric: Distance metric used for neighbour search (``cosine`` or ``l2``).
    :param seed: Pseudo-random seed used by components that rely on randomness.
    :param word2vec: Optional Word2Vec configuration for the ``word2vec`` feature space.
    :param sentence_transformer: Optional sentence-transformer configuration for the
        ``sentence_transformer`` feature space.
    """

    feature_space: str
    metric: str
    seed: int
    word2vec: Optional[Word2VecConfig] = None
    sentence_transformer: Optional[SentenceTransformerConfig] = None

    def feature_space_norm(self) -> str:
        """Return the normalised feature-space token."""
        return (self.feature_space or "tfidf").lower()

    def metric_norm(self) -> str:
        """Return the validated distance metric token (cosine or l2)."""
        value = (self.metric or "cosine").lower()
        return value if value in {"cosine", "l2", "euclidean"} else "cosine"


def _create_features_and_matrix(
    *,
    documents: Sequence[str],
    feature_space: str,
    config: OpinionIndexConfig,
) -> Tuple[Any, Optional[TfidfVectorizer], Any]:
    """Materialise embeddings/matrix for the requested feature space."""
    if feature_space == "tfidf":
        vectorizer, matrix = _build_tfidf_matrix(documents)
        embeds = None
    elif feature_space == "word2vec":
        embeds = Word2VecFeatureBuilder(config.word2vec)
        embeds.train(documents)
        matrix = embeds.transform(documents)
        vectorizer = None
    elif feature_space == "sentence_transformer":
        encoder = SentenceTransformerEncoder(
            config.sentence_transformer or SentenceTransformerConfig()
        )
        matrix = encoder.encode(documents).astype(np.float32, copy=False)
        if not hasattr(encoder, "transform"):
            setattr(encoder, "transform", encoder.encode)  # type: ignore[attr-defined]
        embeds = encoder
        vectorizer = None
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported feature space '{feature_space}'.")
    return matrix, vectorizer, embeds


def _log_embedding_preview(
    *,
    feature_space: str,
    documents: Sequence[str],
    vectorizer: Optional[TfidfVectorizer],
    embeds: Any,
) -> None:
    """Emit a concise embedding preview for the first document (best effort)."""
    try:
        if not documents:
            return
        sample_doc = str(documents[0])
        if feature_space == "tfidf" and vectorizer is not None:
            vec = vectorizer.transform([sample_doc])
            row = vec[0]
            nnz = int(row.nnz)
            dim = int(row.shape[1])
            indices = getattr(row, "indices", [])
            data = getattr(row, "data", [])
            preview = [f"{int(i)}:{float(v):.4f}" for i, v in zip(indices[:8], data[:8])]
            LOGGER.info("[OPINION][Embed][TFIDF] dim=%s nnz=%s preview=%s", dim, nnz, preview)
        elif feature_space in {"word2vec", "sentence_transformer"} and embeds is not None:
            arr = np.asarray(embeds.transform([sample_doc])).ravel()  # type: ignore[attr-defined]
            dim = int(arr.shape[0])
            nnz = int(np.count_nonzero(arr))
            preview = [f"{float(x):.4f}" for x in arr[:8]]
            tag = "W2V" if feature_space == "word2vec" else "ST"
            LOGGER.info("[OPINION][Embed][%s] dim=%s nnz=%s preview=%s", tag, dim, nnz, preview)
    except (ValueError, TypeError, AttributeError, IndexError):  # pragma: no cover - best effort
        pass


def build_index(
    *,
    examples: Sequence[OpinionExample],
    spec: OpinionSpec,
    config: OpinionIndexConfig,
) -> OpinionIndex:
    """
    Vectorise ``examples`` and construct a neighbour index.

    :param examples: Collection of dataset rows used in the evaluation.
    :type examples: Sequence[~knn.core.opinion_models.OpinionExample]
    :param config: Configuration controlling feature space, metric, and encoders.
    :type config: ~knn.core.opinion_index.OpinionIndexConfig
    :param spec: Opinion study specification containing issue metadata.
    :type spec: ~common.opinion.OpinionSpec
    :returns: Configured KNN index ready for inference.
    :rtype: OpinionIndex
    """
    np.random.seed(int(config.seed))

    documents = [example.document for example in examples]
    feature_space = config.feature_space_norm()

    matrix, vectorizer, embeds = _create_features_and_matrix(
        documents=documents, feature_space=feature_space, config=config
    )

    # Emit a concise embedding summary for the first training document
    _log_embedding_preview(
        feature_space=feature_space,
        documents=documents,
        vectorizer=vectorizer,
        embeds=embeds,
    )

    targets_after, targets_before, participant_keys = _prepare_targets(examples)

    neighbors = _build_neighbors(
        matrix=matrix,
        metric=config.metric_norm(),
        n_examples=len(examples),
    )

    LOGGER.info(
        "[OPINION] Built %s index for study=%s participants=%d",
        feature_space.upper(),
        spec.key,
        len(examples),
    )

    return OpinionIndex(
        feature_space=feature_space,
        metric=config.metric_norm(),
        matrix=matrix,
        vectorizer=vectorizer,
        embeds=embeds,
        targets=OpinionTargets(
            after=targets_after,
            before=targets_before,
            participant_keys=participant_keys,
        ),
        neighbors=neighbors,
    )


def _prepare_targets(
    examples: Sequence[OpinionExample],
) -> Tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Prepare target arrays and participant keys from examples."""
    targets_after = np.asarray([example.after for example in examples], dtype=np.float32)
    targets_before = np.asarray([example.before for example in examples], dtype=np.float32)
    participant_keys = [
        (example.participant_id, example.participant_study) for example in examples
    ]
    return targets_after, targets_before, participant_keys


def _build_neighbors(*, matrix: Any, metric: str, n_examples: int) -> NearestNeighbors:
    """Fit a brute-force neighbour index using the requested metric."""
    metric_norm = metric if metric in {"cosine", "l2", "euclidean"} else "cosine"
    neighbor_metric = "cosine" if metric_norm == "cosine" else "euclidean"
    max_neighbors = max(25, n_examples)
    neighbors = NearestNeighbors(
        n_neighbors=min(max_neighbors, n_examples),
        metric=neighbor_metric,
        algorithm="brute",
    )
    neighbors.fit(matrix)
    return neighbors


def _transform_documents(
    *,
    index: OpinionIndex,
    documents: Sequence[str],
) -> Any:
    """
    Transform ``documents`` into the feature space of ``index``.

    :param index: KNN index object or registry being manipulated.
    :type index: OpinionIndex
    :param documents: Iterable of vectorisable documents consumed by the index.
    :type documents: Sequence[str]
    :returns: Matrix of transformed document vectors ready for nearest-neighbour search.
    :rtype: Any
    """
    if index.feature_space == "tfidf":
        if index.vectorizer is None:
            raise RuntimeError("TF-IDF vectoriser missing from index.")
        return index.vectorizer.transform(documents).astype(np.float32)
    if index.feature_space in {"word2vec", "sentence_transformer"}:
        if index.embeds is None or not hasattr(index.embeds, "transform"):
            raise RuntimeError("Embedding encoder missing from index.")
        transformed = index.embeds.transform(list(documents))  # type: ignore[attr-defined]
        return np.asarray(transformed, dtype=np.float32)
    raise ValueError(f"Unsupported feature space '{index.feature_space}'.")


def _similarity_from_distances(distances: np.ndarray, *, metric: str) -> np.ndarray:
    """
    Convert neighbour distances into similarity weights.

    :param distances: Array of neighbour distances returned by the index.
    :type distances: np.ndarray
    :param metric: Name of the evaluation metric being inspected.
    :type metric: str
    :returns: Similarity scores derived from the provided distance array.
    :rtype: np.ndarray
    """
    metric_norm = (metric or "cosine").lower()
    distances = np.asarray(distances, dtype=np.float32)
    if metric_norm == "cosine":
        similarities = 1.0 - distances
        return np.clip(similarities, 0.0, None)
    weights = 1.0 / (distances + 1e-6)
    return np.clip(weights, 0.0, None)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Return the weighted mean of ``values`` with fallback for zero weights.

    :param values: Sequence of numeric values contributing to an aggregate statistic.
    :type values: np.ndarray
    :param weights: Optional weight values aligned with the provided observations.
    :type weights: np.ndarray
    :returns: Weighted mean with a fallback for zero weight scenarios.
    :rtype: float
    """
    total = float(weights.sum())
    if total <= 1e-8:
        return float(values.mean()) if values.size else float("nan")
    return float(np.dot(values, weights) / total)


__all__ = [
    "OpinionIndexConfig",
    "_build_tfidf_matrix",
    "_similarity_from_distances",
    "_transform_documents",
    "_weighted_mean",
    "build_index",
]
