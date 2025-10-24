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

# pylint: disable=too-many-lines
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.random import default_rng
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

from common.embeddings import SentenceTransformerConfig, SentenceTransformerEncoder
from common.opinion import (
    DEFAULT_SPECS,
    OpinionExample as BaseOpinionExample,
    OpinionSpec,
    float_or_none,
    opinion_example_kwargs,
)
from common.vectorizers import create_tfidf_vectorizer

from .data import (
    DEFAULT_DATASET_SOURCE,
    EVAL_SPLIT,
    TRAIN_SPLIT,
    load_dataset_source,
)
from .evaluate import parse_k_values, resolve_reports_dir
from .features import Word2VecConfig, Word2VecFeatureBuilder, assemble_document

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

LOGGER = logging.getLogger("knn.opinion")

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

def find_spec(key: str) -> OpinionSpec:
    """
    Return the :class:`OpinionSpec` matching ``key``.

    :param key: Dictionary key identifying the current record.

    :type key: str

    :returns: the :class:`OpinionSpec` matching ``key``

    :rtype: OpinionSpec

    """
    normalised = key.strip().lower()
    for spec in DEFAULT_SPECS:
        if spec.key.lower() == normalised:
            return spec
    expected_keys = [spec.key for spec in DEFAULT_SPECS]
    raise KeyError(
        f"Unknown opinion study '{key}'. Expected one of {expected_keys!r}"
    )

def collect_examples(
    dataset,
    *,
    spec: OpinionSpec,
    extra_fields: Sequence[str],
    max_examples: int | None,
    seed: int,
) -> List[OpinionExample]:
    """
    Collapse the dataset split into participant-level opinion rows.

    :param dataset: Dataset split providing raw participant interactions.
    :type dataset: datasets.Dataset | Sequence[Mapping[str, Any]]
    :param spec: Opinion study specification describing the target columns.
    :type spec: OpinionSpec
    :param extra_fields: Additional prompt columns appended to each document.
    :type extra_fields: Sequence[str]
    :param max_examples: Optional cap on participants retained from the split.
    :type max_examples: int | None
    :param seed: Random seed applied when subsampling participants.
    :type seed: int
    :returns: Participant-level examples combining prompts and opinion values.
    :rtype: List[OpinionExample]
    """
    # pylint: disable=too-many-locals

    LOGGER.info(
        "[OPINION] Collapsing dataset split for study=%s issue=%s rows=%d",
        spec.key,
        spec.issue,
        len(dataset),
    )
    per_participant: Dict[Tuple[str, str], OpinionExample] = {}

    for idx in range(len(dataset)):
        example = dataset[int(idx)]
        if example.get("issue") != spec.issue or example.get("participant_study") != spec.key:
            continue
        before = float_or_none(example.get(spec.before_column))
        after = float_or_none(example.get(spec.after_column))
        if before is None or after is None:
            continue
        document = assemble_document(example, extra_fields)
        if not document:
            continue
        try:
            step_index = int(example.get("step_index") or -1)
        except (TypeError, ValueError):
            step_index = -1
        participant_id = str(example.get("participant_id") or "")
        key = (participant_id, spec.key)
        existing = per_participant.get(key)
        session_id = example.get("session_id")
        base_kwargs = opinion_example_kwargs(
            participant_id=participant_id,
            participant_study=spec.key,
            issue=spec.issue,
            document=document,
            before=before,
            after=after,
        )
        candidate = OpinionExample(
            **base_kwargs,
            step_index=step_index,
            session_id=str(session_id) if session_id is not None else None,
        )
        if existing is None or step_index >= existing.step_index:
            per_participant[key] = candidate

    collapsed = list(per_participant.values())
    LOGGER.info(
        "[OPINION] Collapsed to %d unique participants (from %d rows).",
        len(collapsed),
        len(dataset),
    )

    if max_examples and 0 < max_examples < len(collapsed):
        rng = default_rng(seed)
        order = rng.permutation(len(collapsed))[:max_examples]
        collapsed = [collapsed[i] for i in order]
        LOGGER.info("[OPINION] Sampled %d participants (max=%d).", len(collapsed), max_examples)

    return collapsed

def _build_tfidf_matrix(documents: Sequence[str]) -> Tuple[TfidfVectorizer, Any]:
    """
    Fit a TF-IDF vectoriser using the supplied documents.

    :param documents: Iterable of vectorisable documents consumed by the index.

    :type documents: Sequence[str]

    :returns: Fitted TF-IDF vectoriser and sparse matrix for the provided documents.

    :rtype: Tuple[TfidfVectorizer, Any]

    """
    vectorizer = create_tfidf_vectorizer(max_features=None)
    matrix = vectorizer.fit_transform(documents).astype(np.float32)
    return vectorizer, matrix

def build_index(
    *,
    examples: Sequence[OpinionExample],
    feature_space: str,
    spec: OpinionSpec,
    seed: int,
    metric: str,
    word2vec_config: Optional[Word2VecConfig] = None,
    sentence_config: Optional[SentenceTransformerConfig] = None,
) -> OpinionIndex:
    """
    Vectorise ``examples`` and construct a neighbour index.

    :param examples: Collection of dataset rows used in the evaluation.

    :type examples: Sequence[OpinionExample]

    :param feature_space: Feature space identifier such as ``tfidf`` or ``word2vec``.

    :type feature_space: str

    :param spec: Opinion study specification containing issue metadata.

    :type spec: OpinionSpec

    :param seed: Seed used to initialise pseudo-random operations.

    :type seed: int

    :param metric: Name of the evaluation metric being inspected.

    :type metric: str

    :param word2vec_config: Configuration object carrying Word2Vec hyper-parameters.

    :type word2vec_config: Optional[Word2VecConfig]

    :param sentence_config: Sentence-transformer configuration describing encoding details.

    :type sentence_config: Optional[SentenceTransformerConfig]

    :returns: Tuple containing the fitted index and any feature-space-specific metadata.

    :rtype: OpinionIndex

    """
    # pylint: disable=too-many-arguments,too-many-locals,unused-argument
    documents = [example.document for example in examples]
    feature_space = (feature_space or "tfidf").lower()

    if feature_space == "tfidf":
        vectorizer, matrix = _build_tfidf_matrix(documents)
        embeds = None
    elif feature_space == "word2vec":
        embeds = Word2VecFeatureBuilder(word2vec_config)
        embeds.train(documents)
        matrix = embeds.transform(documents)
        vectorizer = None
    elif feature_space == "sentence_transformer":
        encoder = SentenceTransformerEncoder(sentence_config or SentenceTransformerConfig())
        matrix = encoder.encode(documents).astype(np.float32, copy=False)
        if not hasattr(encoder, "transform"):
            setattr(encoder, "transform", encoder.encode)  # type: ignore[attr-defined]
        embeds = encoder
        vectorizer = None
    else:
        raise ValueError(f"Unsupported feature space '{feature_space}'.")

    targets_after = np.asarray([example.after for example in examples], dtype=np.float32)
    targets_before = np.asarray([example.before for example in examples], dtype=np.float32)
    participant_keys = [(example.participant_id, example.participant_study) for example in examples]

    max_neighbors = max(25, len(examples))
    metric_norm = (metric or "cosine").lower()
    if metric_norm not in {"cosine", "l2", "euclidean"}:
        LOGGER.warning("[OPINION] Unsupported metric '%s'; falling back to cosine.", metric)
        metric_norm = "cosine"
    neighbor_metric = "cosine" if metric_norm == "cosine" else "euclidean"
    neighbors = NearestNeighbors(
        n_neighbors=min(max_neighbors, len(examples)),
        metric=neighbor_metric,
        algorithm="brute",
    )
    neighbors.fit(matrix)

    LOGGER.info(
        "[OPINION] Built %s index for study=%s participants=%d",
        feature_space.upper(),
        spec.key,
        len(examples),
    )

    return OpinionIndex(
        feature_space=feature_space,
        metric=metric_norm,
        matrix=matrix,
        vectorizer=vectorizer,
        embeds=embeds,
        targets_after=targets_after,
        targets_before=targets_before,
        participant_keys=participant_keys,
        neighbors=neighbors,
    )

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

    :returns: the weighted mean of ``values`` with fallback for zero weights

    :rtype: float

    """
    total = float(weights.sum())
    if total <= 1e-8:
        return float(values.mean()) if values.size else float("nan")
    return float(np.dot(values, weights) / total)

def predict_post_indices(
    *,
    index: OpinionIndex,
    eval_examples: Sequence[OpinionExample],
    k_values: Sequence[int],
    exclude_self: bool = False,
) -> Dict[str, Any]:
    """
    Return predictions and aggregate metrics for ``eval_examples``.

    :param index: KNN index object or registry being manipulated.

    :type index: OpinionIndex

    :param eval_examples: Iterable of evaluation examples to score with the index.

    :type eval_examples: Sequence[OpinionExample]

    :param k_values: Iterable of ``k`` values to evaluate or report.

    :type k_values: Sequence[int]

    :param exclude_self: Whether to drop the query point when collecting nearest neighbours.

    :type exclude_self: bool

    :returns: predictions and aggregate metrics for ``eval_examples``

    :rtype: Dict[str, Any]

    """
    # pylint: disable=too-many-locals
    if not eval_examples:
        return {
            "rows": [],
            "per_k_predictions": {int(k): [] for k in k_values},
        }

    requested_k = [int(k) for k in k_values if int(k) > 0]
    if not requested_k:
        return {
            "rows": [],
            "per_k_predictions": {},
            "per_k_change_predictions": {},
        }
    max_available = len(index.participant_keys) - (1 if exclude_self else 0)
    max_available = max(1, max_available)
    unique_k = sorted({k for k in requested_k if k <= max_available})
    if not unique_k:
        unique_k = [min(max_available, max(requested_k))]
    max_k = max(unique_k)

    documents = [example.document for example in eval_examples]
    matrix_eval = _transform_documents(index=index, documents=documents)

    neighbour_distances, neighbour_indices = index.neighbors.kneighbors(
        matrix_eval,
        n_neighbors=max_k,
    )

    per_k_predictions: Dict[int, List[float]] = {k: [] for k in unique_k}
    per_k_change_predictions: Dict[int, List[float]] = {k: [] for k in unique_k}
    rows: List[Dict[str, Any]] = []

    for row_idx, example in enumerate(eval_examples):
        distances = neighbour_distances[row_idx]
        indices = neighbour_indices[row_idx]
        similarities = _similarity_from_distances(distances, metric=index.metric)

        filtered_indices: List[int] = []
        filtered_weights: List[float] = []
        for candidate_idx, weight in zip(indices, similarities):
            if exclude_self:
                participant_key = index.participant_keys[candidate_idx]
                if (
                    participant_key[0] == example.participant_id
                    and participant_key[1] == example.participant_study
                ):
                    continue
            filtered_indices.append(int(candidate_idx))
            filtered_weights.append(float(weight))
            if len(filtered_indices) >= max_k:
                break

        if not filtered_indices:
            continue

        filtered_indices_arr = np.asarray(filtered_indices, dtype=np.int32)
        filtered_weights_arr = np.asarray(filtered_weights, dtype=np.float32)

        record_after: Dict[int, float] = {}
        record_change: Dict[int, float] = {}
        for k in unique_k:
            if len(filtered_indices_arr) < k:
                continue

            top_indices = filtered_indices_arr[:k]
            top_weights = filtered_weights_arr[:k]
            top_targets_after = index.targets_after[top_indices]
            top_targets_before = index.targets_before[top_indices]
            top_changes = top_targets_after - top_targets_before

            predicted_change = _weighted_mean(top_changes, top_weights)
            anchored_prediction = float(example.before) + predicted_change

            per_k_predictions[k].append(anchored_prediction)
            per_k_change_predictions[k].append(predicted_change)
            record_after[k] = float(anchored_prediction)
            record_change[k] = float(predicted_change)

        rows.append(
            {
                "participant_id": example.participant_id,
                "participant_study": example.participant_study,
                "issue": example.issue,
                "session_id": example.session_id,
                "before_index": example.before,
                "after_index": example.after,
                "predictions_by_k": record_after,
                "predicted_change_by_k": record_change,
            }
        )

    return {
        "rows": rows,
        "per_k_predictions": per_k_predictions,
        "per_k_change_predictions": per_k_change_predictions,
    }

def _direction_labels(delta: np.ndarray, *, tolerance: float = 1e-6) -> np.ndarray:
    """Return categorical labels capturing the direction of opinion change."""

    labels = np.zeros(delta.shape, dtype=np.int8)
    labels[delta > tolerance] = 1
    labels[delta < -tolerance] = -1
    return labels


def _metric_bundle(
    *,
    truth_after: np.ndarray,
    truth_before: np.ndarray,
    preds_arr: np.ndarray,
) -> Dict[str, float]:
    """Compute evaluation metrics for the supplied predictions."""

    change_truth = truth_after - truth_before
    change_pred = preds_arr - truth_before
    mae = float(mean_absolute_error(truth_after, preds_arr))
    rmse = float(math.sqrt(mean_squared_error(truth_after, preds_arr)))
    r_squared = float(r2_score(truth_after, preds_arr))
    change_mae = float(mean_absolute_error(change_truth, change_pred))
    direction_truth = _direction_labels(change_truth)
    direction_pred = _direction_labels(change_pred)
    accuracy = (
        float(np.mean(direction_truth == direction_pred))
        if direction_truth.size
        else float("nan")
    )
    eligible = int(direction_truth.size)
    return {
        "mae_after": mae,
        "rmse_after": rmse,
        "r2_after": r_squared,
        "mae_change": change_mae,
        "direction_accuracy": accuracy,
        "eligible": eligible,
    }


def _row_prediction_values(
    rows: Sequence[Dict[str, Any]],
    k: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Extract per-row prediction/ground-truth triples for a given ``k``."""

    actual_after: List[float] = []
    actual_before: List[float] = []
    pred_values: List[float] = []
    key_str = str(k)
    for row in rows:
        predictions_by_k = row.get("predictions_by_k") or {}
        pred_val = predictions_by_k.get(k)
        if pred_val is None:
            pred_val = predictions_by_k.get(key_str)
        if pred_val is None:
            continue
        after_raw = row.get("after_index", row.get("after"))
        before_raw = row.get("before_index", row.get("before"))
        if after_raw is None or before_raw is None:
            continue
        after_val = float(after_raw)
        before_val = float(before_raw)
        if not (math.isfinite(after_val) and math.isfinite(before_val)):
            continue
        pred_values.append(float(pred_val))
        actual_after.append(after_val)
        actual_before.append(before_val)
    return actual_after, actual_before, pred_values


def _metrics_from_rows(
    predictions: Dict[int, List[float]],
    rows: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, float]]:
    """Return metrics computed from row-level prediction payloads."""

    metrics: Dict[int, Dict[str, float]] = {}
    for key in sorted({int(candidate) for candidate in predictions.keys()}):
        actual_after, actual_before, pred_values = _row_prediction_values(rows, key)
        if not pred_values:
            continue
        bundle = _metric_bundle(
            truth_after=np.asarray(actual_after, dtype=np.float32),
            truth_before=np.asarray(actual_before, dtype=np.float32),
            preds_arr=np.asarray(pred_values, dtype=np.float32),
        )
        metrics[key] = bundle
    return metrics


def _metrics_from_eval_examples(
    predictions: Dict[int, List[float]],
    eval_examples: Sequence[OpinionExample],
) -> Dict[int, Dict[str, float]]:
    """Return metrics computed over the full evaluation dataset."""

    truth_after = np.asarray([example.after for example in eval_examples], dtype=np.float32)
    truth_before = np.asarray([example.before for example in eval_examples], dtype=np.float32)
    metrics: Dict[int, Dict[str, float]] = {}
    for k, preds in predictions.items():
        preds_arr = np.asarray(preds, dtype=np.float32)
        if preds_arr.size == 0 or preds_arr.size != truth_after.size:
            continue
        metrics[int(k)] = _metric_bundle(
            truth_after=truth_after,
            truth_before=truth_before,
            preds_arr=preds_arr,
        )
    return metrics


def _summary_metrics(
    *,
    predictions: Dict[int, List[float]],
    eval_examples: Sequence[OpinionExample],
    rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Return MAE, RMSE, and R^2 metrics for each ``k``.

    :param predictions: Sequence of KNN prediction records emitted during evaluation.

    :type predictions: Dict[int, List[float]]

    :param eval_examples: Iterable of evaluation examples to score with the index.

    :type eval_examples: Sequence[OpinionExample]

    :param rows: Optional iterable of per-example prediction rows. When provided,
        metrics are computed only over examples that produced predictions for the
        given ``k``.

    :type rows: Optional[Sequence[Dict[str, Any]]]

    :returns: MAE, RMSE, and R^2 metrics for each ``k``

    :rtype: Dict[int, Dict[str, float]]

    """
    # pylint: disable=too-many-locals
    if rows:
        metrics = _metrics_from_rows(predictions, rows)
        if metrics:
            return metrics
    return _metrics_from_eval_examples(predictions, eval_examples)

@dataclass
class _CurveAccumulator:
    """Accumulate per-k metrics and track the best-performing configuration."""

    mae_by_k: Dict[str, Optional[float]] = field(default_factory=dict)
    r2_by_k: Dict[str, Optional[float]] = field(default_factory=dict)
    _best: Optional[Tuple[int, float, float]] = None

    def add(self, k: int, values: Mapping[str, float]) -> None:
        """Record metrics for a specific ``k`` and update best trackers."""
        raw_mae = float(values.get("mae_after", float("nan")))
        raw_r2 = float(values.get("r2_after", float("nan")))
        mae_value = raw_mae if math.isfinite(raw_mae) else float("inf")
        r2_value = raw_r2 if math.isfinite(raw_r2) else float("-inf")
        self.mae_by_k[str(int(k))] = raw_mae if math.isfinite(raw_mae) else None
        self.r2_by_k[str(int(k))] = raw_r2 if math.isfinite(raw_r2) else None
        if self._is_preferred(mae_value, r2_value):
            self._best = (int(k), mae_value, r2_value)

    def best_summary(self, fallback_k: int) -> Tuple[int, Optional[float], Optional[float]]:
        """Return the best-performing ``k`` and associated metrics (or fallback)."""
        if self._best is None:
            return fallback_k, None, None
        best_k, mae_value, r2_value = self._best
        best_mae = mae_value if math.isfinite(mae_value) else None
        best_r2 = r2_value if math.isfinite(r2_value) else None
        return best_k, best_mae, best_r2

    def _is_preferred(self, mae_value: float, r2_value: float) -> bool:
        if self._best is None:
            return True
        _, best_mae, best_r2 = self._best
        if mae_value < best_mae - 1e-9:
            return True
        if mae_value <= best_mae + 1e-9 and r2_value > best_r2:
            return True
        return False


def _curve_payload(
    metrics_by_k: Dict[int, Dict[str, float]],
    *,
    n_examples: int,
) -> Optional[Dict[str, Any]]:
    """
    Convert ``metrics_by_k`` into a serialisable curve bundle.

    :param metrics_by_k: Mapping from each ``k`` to its associated opinion metrics.

    :type metrics_by_k: Dict[int, Dict[str, float]]

    :param n_examples: Total number of evaluation examples summarised in the bundle.

    :type n_examples: int

    :returns: Dictionary summarising the evaluation curve, including AUC and per-k metrics.

    :rtype: Optional[Dict[str, Any]]

    """
    if not metrics_by_k:
        return None

    accumulator = _CurveAccumulator()
    sorted_items = sorted((int(k), values) for k, values in metrics_by_k.items())
    for k, values in sorted_items:
        accumulator.add(k, values)

    fallback_k = sorted_items[0][0]
    best_k, best_mae, best_r2 = accumulator.best_summary(fallback_k)

    return {
        "metric": "mae_after",
        "mae_by_k": accumulator.mae_by_k,
        "r2_by_k": accumulator.r2_by_k,
        "best_k": int(best_k),
        "best_mae": best_mae,
        "best_r2": best_r2,
        "n_examples": int(n_examples),
    }

def _baseline_metrics(eval_examples: Sequence[OpinionExample]) -> Dict[str, float]:
    """
    Return baseline error metrics for opinion prediction.

    :param eval_examples: Iterable of evaluation examples to score with the index.

    :type eval_examples: Sequence[OpinionExample]

    :returns: baseline error metrics for opinion prediction

    :rtype: Dict[str, float]

    """
    truth_after = np.asarray([example.after for example in eval_examples], dtype=np.float32)
    truth_before = np.asarray([example.before for example in eval_examples], dtype=np.float32)
    change_truth = truth_after - truth_before

    baseline_mean = float(truth_after.mean()) if truth_after.size else float("nan")
    baseline_predictions = np.full_like(truth_after, baseline_mean)
    mae_mean = float(mean_absolute_error(truth_after, baseline_predictions))
    rmse_mean = float(math.sqrt(mean_squared_error(truth_after, baseline_predictions)))

    mae_no_change = float(mean_absolute_error(truth_after, truth_before))
    change_zero = np.zeros_like(change_truth)
    mae_change_zero = float(mean_absolute_error(change_truth, change_zero))
    direction_truth = _direction_labels(change_truth)
    baseline_direction_accuracy = (
        float(np.mean(direction_truth == 0))
        if direction_truth.size
        else float("nan")
    )

    return {
        "mae_global_mean_after": mae_mean,
        "rmse_global_mean_after": rmse_mean,
        "mae_using_before": mae_no_change,
        "mae_change_zero": mae_change_zero,
        "global_mean_after": baseline_mean,
        "direction_accuracy": baseline_direction_accuracy,
    }

def _plot_metric(
    *,
    metrics_by_k: Dict[int, Dict[str, float]],
    metric_key: str,
    output_path: Path,
) -> None:
    """
    Save a line plot of ``metric_key`` vs. ``k`` if matplotlib is available.

    :param metrics_by_k: Mapping from each ``k`` to its associated opinion metrics.

    :type metrics_by_k: Dict[int, Dict[str, float]]

    :param metric_key: Dictionary key pointing to a metric within the payload.

    :type metric_key: str

    :param output_path: Filesystem path for the generated report or figure.

    :type output_path: Path

    :returns: None.

    :rtype: None

    """
    # pylint: disable=too-many-arguments,too-many-locals
    if plt is None:  # pragma: no cover - optional dependency
        LOGGER.warning("[OPINION] Skipping %s plot (matplotlib not installed).", metric_key)
        return

    if not metrics_by_k:
        LOGGER.warning("[OPINION] Skipping %s plot (no metrics).", metric_key)
        return

    sorted_items = sorted(metrics_by_k.items())
    k_values = [item[0] for item in sorted_items]
    values = [item[1][metric_key] for item in sorted_items]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, values, marker="o")
    plt.title(f"{metric_key} vs k")
    plt.xlabel("k")
    plt.ylabel(metric_key)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _plot_change_heatmap(
    *,
    actual_changes: Sequence[float],
    predicted_changes: Sequence[float],
    output_path: Path,
) -> None:
    """
    Render a 2D histogram comparing actual vs. predicted opinion shifts.

    :param actual_changes: Sequence of observed opinion deltas for participants.

    :type actual_changes: Sequence[float]

    :param predicted_changes: Predicted opinion deltas returned by the model.

    :type predicted_changes: Sequence[float]

    :param output_path: Filesystem path for the generated report or figure.

    :type output_path: Path

    :returns: None.

    :rtype: None

    """
    if plt is None:  # pragma: no cover - optional dependency
        LOGGER.warning("[OPINION] Skipping opinion-change heatmap (matplotlib not installed).")
        return

    if not actual_changes or not predicted_changes:
        LOGGER.warning("[OPINION] Skipping opinion-change heatmap (no valid predictions).")
        return

    actual = np.asarray(actual_changes, dtype=np.float32)
    predicted = np.asarray(predicted_changes, dtype=np.float32)
    if actual.size == 0 or predicted.size == 0:
        LOGGER.warning("[OPINION] Skipping opinion-change heatmap (empty arrays).")
        return

    min_val = float(min(actual.min(), predicted.min()))
    max_val = float(max(actual.max(), predicted.max()))
    if math.isclose(min_val, max_val):
        span = 0.1 if math.isfinite(min_val) else 1.0
        min_val -= span
        max_val += span
    else:
        extent = max(abs(min_val), abs(max_val))
        if not math.isfinite(extent) or extent <= 1e-6:
            extent = 1.0
        min_val, max_val = -extent, extent

    bins = 40
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4.5))
    hist = plt.hist2d(
        actual,
        predicted,
        bins=bins,
        range=[[min_val, max_val], [min_val, max_val]],
        cmap="magma",
        cmin=1,
    )
    plt.colorbar(hist[3], label="Participants")
    plt.plot([min_val, max_val], [min_val, max_val], color="cyan", linestyle="--", linewidth=1.0)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
    plt.axvline(0.0, color="grey", linestyle=":", linewidth=0.8)
    plt.xlabel("Actual opinion change (post - pre)")
    plt.ylabel("Predicted opinion change")
    plt.title("Predicted vs. actual opinion change")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

@dataclass(frozen=True)
class _OutputContext:
    """Lightweight wrapper for output-related runtime state."""

    args: Any
    spec: OpinionSpec
    index: OpinionIndex
    outputs_root: Path


@dataclass(frozen=True)
class _OutputPayload:
    """Container bundling metrics and per-example predictions for export."""

    rows: Sequence[Dict[str, Any]]
    metrics_by_k: Dict[int, Dict[str, float]]
    baseline: Dict[str, float]
    best_k: int
    curve_metrics: Optional[Dict[str, Any]] = None


def _opinion_change_series(
    rows: Sequence[Dict[str, Any]],
    best_k: int,
) -> Tuple[List[float], List[float]]:
    """Return paired lists of actual and predicted opinion changes."""

    actual_changes: List[float] = []
    predicted_changes: List[float] = []
    for row in rows:
        prediction = row.get("predictions_by_k", {}).get(best_k)
        if prediction is None:
            continue
        before = float(row["before_index"])
        actual_changes.append(float(row["after_index"]) - before)
        change_lookup = row.get("predicted_change_by_k", {})
        predicted_change = change_lookup.get(best_k)
        if predicted_change is None:
            predicted_change = float(prediction) - before
        predicted_changes.append(float(predicted_change))
    return actual_changes, predicted_changes


def _write_prediction_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Serialise per-row predictions to ``path`` in JSONL format."""

    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            change_dict = {
                str(k): float(v) for k, v in row.get("predicted_change_by_k", {}).items()
            }
            serializable = {
                **row,
                "predictions_by_k": {str(k): float(v) for k, v in row["predictions_by_k"].items()},
                "predicted_change_by_k": change_dict,
            }
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def _compose_metrics_record(
    context: _OutputContext,
    payload: _OutputPayload,
    plots: Mapping[str, Path],
) -> Dict[str, Any]:
    """Build the metrics payload persisted alongside predictions."""

    metrics_by_k = {}
    for k, values in payload.metrics_by_k.items():
        bundle = {
            "mae_after": float(values["mae_after"]),
            "rmse_after": float(values["rmse_after"]),
            "r2_after": float(values["r2_after"]),
            "mae_change": float(values["mae_change"]),
        }
        direction_accuracy = values.get("direction_accuracy")
        if direction_accuracy is not None:
            bundle["direction_accuracy"] = float(direction_accuracy)
        eligible_value = values.get("eligible")
        if eligible_value is not None:
            bundle["eligible"] = int(eligible_value)
        metrics_by_k[str(k)] = bundle
    record: Dict[str, Any] = {
        "model": "knn_opinion",
        "feature_space": context.index.feature_space,
        "dataset": context.args.dataset or DEFAULT_DATASET_SOURCE,
        "study": context.spec.key,
        "issue": context.spec.issue,
        "label": context.spec.label,
        "split": EVAL_SPLIT,
        "n_participants": len(payload.rows),
        "metrics_by_k": metrics_by_k,
        "baseline": payload.baseline,
        "best_k": int(payload.best_k),
        "best_metrics": payload.metrics_by_k.get(int(payload.best_k), {}),
        "plots": {
            "mae_vs_k": str(plots["mae"]),
            "r2_vs_k": str(plots["r2"]),
            "change_heatmap": str(plots["heatmap"]),
        },
    }
    if payload.curve_metrics:
        record["curve_metrics"] = payload.curve_metrics
    best_metrics = record["best_metrics"]
    best_direction_accuracy = best_metrics.get("direction_accuracy")
    if best_direction_accuracy is not None:
        record["best_direction_accuracy"] = float(best_direction_accuracy)
    eligible_best = best_metrics.get("eligible")
    if eligible_best is not None:
        record["eligible"] = int(eligible_best)
    return record


def _write_outputs(
    *,
    context: _OutputContext,
    payload: _OutputPayload,
) -> None:
    """
    Persist per-example predictions, metrics, and plots.

    :param context: Runtime configuration referencing CLI args and output roots.

    :type context: _OutputContext

    :param payload: Metrics, predictions, and derived artefacts to persist.

    :type payload: _OutputPayload

    :returns: None.

    :rtype: None

    """
    feature_space = context.index.feature_space
    study_dir = context.outputs_root / context.spec.key
    study_dir.mkdir(parents=True, exist_ok=True)

    actual_changes, predicted_changes = _opinion_change_series(payload.rows, payload.best_k)

    predictions_path = study_dir / f"opinion_knn_{context.spec.key}_{EVAL_SPLIT}.jsonl"
    _write_prediction_rows(predictions_path, payload.rows)

    reports_dir = (
        resolve_reports_dir(Path(context.args.out_dir)) / "knn" / feature_space / "opinion"
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    mae_plot = reports_dir / f"mae_{context.spec.key}.png"
    r2_plot = reports_dir / f"r2_{context.spec.key}.png"
    _plot_metric(metrics_by_k=payload.metrics_by_k, metric_key="mae_after", output_path=mae_plot)
    _plot_metric(metrics_by_k=payload.metrics_by_k, metric_key="r2_after", output_path=r2_plot)
    heatmap_path = reports_dir / f"change_heatmap_{context.spec.key}.png"
    _plot_change_heatmap(
        actual_changes=actual_changes,
        predicted_changes=predicted_changes,
        output_path=heatmap_path,
    )

    metrics_path = study_dir / f"opinion_knn_{context.spec.key}_{EVAL_SPLIT}_metrics.json"
    serializable_metrics = _compose_metrics_record(
        context,
        payload,
        {"mae": mae_plot, "r2": r2_plot, "heatmap": heatmap_path},
    )
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(serializable_metrics, handle, ensure_ascii=False, indent=2)

    LOGGER.info(
        "[OPINION] Wrote predictions=%s metrics=%s",
        predictions_path,
        metrics_path,
    )

# pylint: disable=too-many-branches,too-many-locals,too-many-statements
def run_opinion_eval(args) -> None:
    """
    Execute the post-study opinion index evaluation.

    :param args: Namespace object containing parsed command-line arguments.

    :type args: Any

    :returns: None.

    :rtype: None

    """
    os_env = os.environ
    os_env.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os_env.setdefault("HF_HOME", args.cache_dir)

    dataset_source = args.dataset or DEFAULT_DATASET_SOURCE
    dataset = load_dataset_source(dataset_source, args.cache_dir)

    if args.opinion_studies:
        requested = [
            token.strip() for token in args.opinion_studies.split(",") if token.strip()
        ]
    else:
        requested = [spec.key for spec in DEFAULT_SPECS]

    specs = [find_spec(token) for token in requested]

    extra_fields = [
        token.strip()
        for token in (args.knn_text_fields or "").split(",")
        if token.strip()
    ]

    k_values = parse_k_values(args.knn_k, args.knn_k_sweep)
    LOGGER.info("[OPINION] Evaluating k values: %s", k_values)

    feature_space = str(getattr(args, "feature_space", "tfidf")).lower()
    word2vec_cfg = None
    if feature_space == "word2vec":
        word2vec_cfg = Word2VecConfig(
            vector_size=int(args.word2vec_size),
            window=int(getattr(args, "word2vec_window", Word2VecConfig().window)),
            min_count=int(getattr(args, "word2vec_min_count", Word2VecConfig().min_count)),
            epochs=int(getattr(args, "word2vec_epochs", Word2VecConfig().epochs)),
            model_dir=Path(args.word2vec_model_dir or Word2VecConfig().model_dir) / "opinion",
            seed=int(getattr(args, "knn_seed", Word2VecConfig().seed)),
            workers=int(getattr(args, "word2vec_workers", Word2VecConfig().workers)),
        )
    sentence_cfg = None
    if feature_space == "sentence_transformer":
        device_raw = getattr(args, "sentence_transformer_device", "")
        sentence_cfg = SentenceTransformerConfig(
            model_name=getattr(
                args,
                "sentence_transformer_model",
                SentenceTransformerConfig().model_name,
            ),
            device=device_raw or None,
            batch_size=int(getattr(args, "sentence_transformer_batch_size", 32)),
            normalize=bool(getattr(args, "sentence_transformer_normalize", True)),
        )

    outputs_root = Path(args.out_dir) / "opinion" / feature_space
    outputs_root.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        LOGGER.info("[OPINION] Study=%s (%s)", spec.key, spec.label)
        train_examples = collect_examples(
            dataset[TRAIN_SPLIT],
            spec=spec,
            extra_fields=extra_fields,
            max_examples=int(getattr(args, "knn_max_train", 0) or 0),
            seed=int(getattr(args, "knn_seed", 42)),
        )
        if not train_examples:
            LOGGER.warning("[OPINION] No training examples found for study=%s", spec.key)
            continue

        index = build_index(
            examples=train_examples,
            feature_space=feature_space,
            spec=spec,
            seed=int(getattr(args, "knn_seed", 42)),
            metric=str(getattr(args, "knn_metric", "cosine")),
            word2vec_config=word2vec_cfg,
            sentence_config=sentence_cfg,
        )

        eval_examples = collect_examples(
            dataset[EVAL_SPLIT],
            spec=spec,
            extra_fields=extra_fields,
            max_examples=int(getattr(args, "eval_max", 0) or 0),
            seed=int(getattr(args, "knn_seed", 42)),
        )
        if not eval_examples:
            LOGGER.warning("[OPINION] No evaluation examples found for study=%s", spec.key)
            continue

        train_curve_metrics: Optional[Dict[str, Any]] = None
        if train_examples:
            train_predictions_bundle = predict_post_indices(
                index=index,
                eval_examples=train_examples,
                k_values=k_values,
                exclude_self=True,
            )
            train_rows = train_predictions_bundle["rows"]
            train_metrics_by_k = _summary_metrics(
                predictions=train_predictions_bundle["per_k_predictions"],
                eval_examples=train_examples,
                rows=train_rows,
            )
            train_curve = _curve_payload(
                train_metrics_by_k,
                n_examples=len(train_examples),
            )
            if train_curve:
                train_curve_metrics = train_curve

        predictions_bundle = predict_post_indices(
            index=index,
            eval_examples=eval_examples,
            k_values=k_values,
        )
        rows = predictions_bundle["rows"]
        metrics_by_k = _summary_metrics(
            predictions=predictions_bundle["per_k_predictions"],
            eval_examples=eval_examples,
            rows=rows,
        )

        if metrics_by_k:
            best_k, _ = max(metrics_by_k.items(), key=lambda item: item[1]["r2_after"])
        else:
            best_k = k_values[0]

        baseline = _baseline_metrics(eval_examples)
        eval_curve_metrics = _curve_payload(
            metrics_by_k,
            n_examples=len(eval_examples),
        )
        curve_bundle: Optional[Dict[str, Any]] = None
        if eval_curve_metrics:
            curve_bundle = {"eval": eval_curve_metrics}
            if train_curve_metrics:
                curve_bundle["train"] = train_curve_metrics
        elif train_curve_metrics:
            curve_bundle = {"train": train_curve_metrics}
        if curve_bundle is not None:
            curve_bundle["metric"] = "mae_after"
        output_context = _OutputContext(
            args=args,
            spec=spec,
            index=index,
            outputs_root=outputs_root,
        )
        output_payload = _OutputPayload(
            rows=rows,
            metrics_by_k=metrics_by_k,
            baseline=baseline,
            best_k=best_k,
            curve_metrics=curve_bundle,
        )
        _write_outputs(context=output_context, payload=output_payload)

        LOGGER.info(
            "[OPINION][DONE] study=%s participants=%d best_k=%d r2=%.4f mae=%.4f",
            spec.key,
            len(rows),
            best_k,
            float(metrics_by_k.get(best_k, {}).get("r2_after", float("nan"))),
            float(metrics_by_k.get(best_k, {}).get("mae_after", float("nan"))),
        )

__all__ = ["run_opinion_eval", "OpinionSpec", "DEFAULT_SPECS"]
