"""TF-IDF index construction and slate-level prediction helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .features import (
    assemble_document,
    extract_slate_items,
    prepare_training_documents,
    title_for,
)


@dataclass(frozen=True)
class SlateQueryConfig:
    """Configuration controlling slate scoring behaviour."""

    text_fields: Sequence[str] = ()
    lowercase: bool = True
    metric: str | None = None


@dataclass(frozen=True)
class SlateIndexData:
    """Pre-computed index artefacts used during slate scoring."""

    vectorizer: Any
    matrix: Any
    label_id_canon: np.ndarray
    label_title_canon: np.ndarray


@dataclass(frozen=True)
class CandidateScorer:
    """Callable helper for scoring slate candidates."""

    index_data: SlateIndexData
    base_parts: Sequence[str]
    config: SlateQueryConfig
    unique_k: Sequence[int]

    def score(self, title: str, video_id: str) -> Dict[int, float]:
        """Return per-``k`` scores for the given slate candidate."""
        query, sims = _candidate_query(
            index_data=self.index_data,
            base_parts=self.base_parts,
            title=title,
            video_id=video_id,
            config=self.config,
        )
        mask = _candidate_mask(title, video_id, self.index_data)
        if not mask.any():
            fallback = float(sims.max() * 0.01) if sims.size else 0.0
            return {k: fallback for k in self.unique_k}

        sims_masked = sims[mask]
        if sims_masked.size == 0:
            return {k: 0.0 for k in self.unique_k}

        score_vector = _score_vector_from_similarity(
            sims_masked=sims_masked,
            index_data=self.index_data,
            mask=mask,
            query=query,
            metric=self.config.metric,
        )
        return _aggregate_scores(score_vector, self.unique_k)


def build_tfidf_index(
    train_ds,
    *,
    max_train: int = 100_000,
    seed: int = 42,
    max_features: int | None = 200_000,
    extra_fields: Sequence[str] | None = None,
):
    """Create a TF-IDF index from the training split.

    Returns a dictionary containing the fitted vectoriser, sparse matrix, and
    labels for candidate filtering.
    """

    docs, labels_id, labels_title = prepare_training_documents(
        train_ds,
        max_train=max_train,
        seed=seed,
        extra_fields=extra_fields,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None,
        token_pattern=r"(?u)\b[\w\-]{2,}\b",
        max_features=max_features,
    )
    matrix = vectorizer.fit_transform(docs).astype(np.float32)

    return {
        "vectorizer": vectorizer,
        "X": matrix,
        "labels_id": labels_id,
        "labels_title": labels_title,
    }


def save_tfidf_index(index: Dict[str, Any], out_dir: str) -> None:
    """Persist the TF-IDF index to ``out_dir`` for later reuse."""

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(index["vectorizer"], directory / "vectorizer.joblib")
    sparse.save_npz(directory / "X.npz", index["X"])
    np.save(
        directory / "labels_id.npy",
        np.asarray(index["labels_id"], dtype=object),
        allow_pickle=True,
    )
    np.save(
        directory / "labels_title.npy",
        np.asarray(index["labels_title"], dtype=object),
        allow_pickle=True,
    )
    meta = {
        "n_docs": int(index["X"].shape[0]),
        "n_features": int(index["X"].shape[1]),
    }
    with open(directory / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def load_tfidf_index(in_dir: str) -> Dict[str, Any]:
    """Load a TF-IDF index previously saved via :func:`save_tfidf_index`."""

    directory = Path(in_dir)
    vectorizer = joblib.load(directory / "vectorizer.joblib")
    matrix = sparse.load_npz(directory / "X.npz")
    labels_id = np.load(directory / "labels_id.npy", allow_pickle=True).tolist()
    labels_title = np.load(directory / "labels_title.npy", allow_pickle=True).tolist()
    return {
        "vectorizer": vectorizer,
        "X": matrix,
        "labels_id": labels_id,
        "labels_title": labels_title,
    }


def _safe_str(value: Any, *, lowercase: bool = True) -> str:
    """Return a normalised string representation of ``value``."""

    try:
        text = "" if value is None else str(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        text = ""
    text = text.strip()
    return text.lower() if lowercase else text


def _build_base_parts(
    example: dict,
    slate_pairs: Sequence[tuple[str, str]],
    config: SlateQueryConfig,
) -> List[str]:
    parts: List[str] = []
    base_text = assemble_document(example, config.text_fields)
    if base_text:
        parts.append(_safe_str(base_text, lowercase=config.lowercase))

    surfaces = []
    for title, video_id in slate_pairs:
        surface = _surface_text(title, video_id)
        if surface:
            surfaces.append(_safe_str(surface, lowercase=config.lowercase))
    if surfaces:
        parts.append(" ".join(surfaces))
    return parts


def _surface_text(title: str, video_id: str) -> str:
    """Return the most human-friendly representation for a candidate surface."""
    if title and title.strip() and title != "(untitled)":
        return title
    return title_for(video_id) or video_id or ""


def _score_candidates(
    index_data: SlateIndexData,
    slate_pairs: Sequence[tuple[str, str]],
    base_parts: Sequence[str],
    unique_k: Sequence[int],
    config: SlateQueryConfig,
) -> Dict[int, List[float]]:
    scorer = CandidateScorer(index_data, base_parts, config, unique_k)
    scores_by_k = {k: [] for k in unique_k}
    for title, video_id in slate_pairs:
        candidate_scores = scorer.score(title, video_id)
        for k, score in candidate_scores.items():
            scores_by_k[k].append(score)
    return scores_by_k


def _candidate_query(
    index_data: SlateIndexData,
    base_parts: Sequence[str],
    title: str,
    video_id: str,
    config: SlateQueryConfig,
):
    """Return the transformed query vector and raw similarities for a candidate."""
    surface = _surface_text(title, video_id)
    parts = list(base_parts)
    if surface:
        parts.append(_safe_str(surface, lowercase=config.lowercase))
    query_text = "\n".join(part for part in parts if part)
    query = index_data.vectorizer.transform([query_text])
    sims = (query @ index_data.matrix.T).toarray().ravel()
    return query, sims


def _score_vector_from_similarity(
    *,
    sims_masked,
    index_data: SlateIndexData,
    mask: np.ndarray,
    query,
    metric: Optional[str],
):
    """Return a scoring vector based on the configured distance metric."""
    if metric == "l2":
        subset = index_data.matrix[mask].astype(np.float32)
        diff = subset - query
        dists = np.sqrt(diff.power(2).sum(axis=1)).A.ravel()
        return -dists
    return sims_masked


def _aggregate_scores(score_vector: np.ndarray, unique_k: Sequence[int]) -> Dict[int, float]:
    """Aggregate neighbour scores for each ``k`` value."""
    sorted_scores = np.sort(score_vector)[::-1]
    cumulative = np.cumsum(sorted_scores)
    return {
        k: float(cumulative[int(min(max(1, k), sorted_scores.size)) - 1])
        for k in unique_k
    }


def _candidate_mask(
    title: str,
    video_id: str,
    index_data: SlateIndexData,
) -> np.ndarray:
    """Return a boolean mask selecting rows that match the candidate."""
    mask = np.zeros_like(index_data.label_id_canon, dtype=bool)
    vid_canon = _canon_vid(video_id or "")
    if vid_canon:
        mask |= index_data.label_id_canon == vid_canon
    title_canon = _canon(title or "")
    if title_canon:
        mask |= index_data.label_title_canon == title_canon
    return mask


def knn_predict_among_slate_multi(
    *,
    knn_index: Dict[str, Any],
    example: dict,
    k_values: Sequence[int],
    config: Optional[SlateQueryConfig] = None,
) -> Dict[int, Optional[int]]:  # pylint: disable=too-many-branches
    """Score each slate option using candidate-aware TF-IDF kNN.

    Returns a mapping from ``k`` to the predicted option index (1-based). ``None``
    indicates that a prediction could not be produced for that ``k``.
    """

    config = config or SlateQueryConfig()
    if config.metric not in (None, "l2", "cosine"):
        raise ValueError(f"Unsupported metric '{config.metric}'")

    unique_k = sorted({int(k) for k in k_values if int(k) > 0})
    if not unique_k:
        return {}

    slate_pairs = extract_slate_items(example)
    if not slate_pairs:
        return {k: 1 for k in unique_k}

    base_parts = _build_base_parts(example, slate_pairs, config)

    index_data = _build_index_data(knn_index)
    if index_data is None:
        return {k: None for k in unique_k}

    scores_by_k = _score_candidates(
        index_data,
        slate_pairs,
        base_parts,
        unique_k,
        config,
    )

    predictions: Dict[int, Optional[int]] = {}
    for k, scores in scores_by_k.items():
        if not scores or not any(np.isfinite(scores)):
            predictions[k] = None
        else:
            predictions[k] = int(np.argmax(scores)) + 1
    return predictions


def _build_index_data(knn_index: Dict[str, Any]) -> Optional[SlateIndexData]:
    if not knn_index or "vectorizer" not in knn_index:
        return None
    labels_id = np.asarray(knn_index["labels_id"], dtype=object)
    labels_title = np.asarray(knn_index["labels_title"], dtype=object)
    return SlateIndexData(
        vectorizer=knn_index["vectorizer"],
        matrix=knn_index["X"],
        label_id_canon=np.asarray([_canon_vid(label) for label in labels_id], dtype=object),
        label_title_canon=np.asarray([_canon(label or "") for label in labels_title], dtype=object),
    )


def _canon(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower().strip())


def _canon_vid(value: str) -> str:
    if not isinstance(value, str):
        return ""
    match = re.search(r"([A-Za-z0-9_-]{11})", value)
    return match.group(1) if match else value.strip()


__all__ = [
    "build_tfidf_index",
    "knn_predict_among_slate_multi",
    "load_tfidf_index",
    "save_tfidf_index",
]
