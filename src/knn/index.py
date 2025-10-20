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

from prompt_builder.formatters import clean_text

from .data import PROMPT_COLUMN
from .features import (
    extract_now_watching,
    extract_slate_items,
    prepare_training_documents,
    prompt_from_builder,
    title_for,
    viewer_profile_sentence,
)


@dataclass(frozen=True)
class SlateQueryConfig:
    """Configuration controlling slate scoring behaviour."""

    text_fields: Sequence[str] = ()
    lowercase: bool = True
    metric: str | None = None


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

    base_parts: List[str] = []
    prompt_text = prompt_from_builder(example)
    prompt_added = False
    if prompt_text:
        base_parts.append(_safe_str(prompt_text, lowercase=lowercase))
        prompt_added = True
    if not prompt_added:
        profile = viewer_profile_sentence(example)
        if profile:
            base_parts.append(_safe_str(profile, lowercase=lowercase))
    if not prompt_added and PROMPT_COLUMN in example:
        state_text = clean_text(example.get(PROMPT_COLUMN))
        if state_text:
            base_parts.append(_safe_str(state_text, lowercase=lowercase))
    for field in text_fields or []:
        if field in example and example[field] is not None:
            base_parts.append(_safe_str(example[field], lowercase=lowercase))
    current = extract_now_watching(example)
    if current:
        now_title, now_id = current
        if now_title:
            base_parts.append(_safe_str(now_title, lowercase=lowercase))
        if now_id:
            base_parts.append(_safe_str(now_id, lowercase=lowercase))
    slate_surfaces: List[str] = []
    for title, video_id in slate_pairs:
        surface = (
            title
            if title and title.strip() and title != "(untitled)"
            else (title_for(video_id) or video_id or "")
        )
        if surface:
            slate_surfaces.append(_safe_str(surface, lowercase=lowercase))
    if slate_surfaces:
        base_parts.append(" ".join(slate_surfaces))

    if not knn_index or "vectorizer" not in knn_index:
        return {k: None for k in unique_k}

    vectorizer = knn_index["vectorizer"]
    matrix = knn_index["X"]
    labels_id = np.asarray(knn_index["labels_id"], dtype=object)
    labels_title = np.asarray(knn_index["labels_title"], dtype=object)

    label_id_canon = np.asarray([_canon_vid(label) for label in labels_id], dtype=object)
    label_title_canon = np.asarray(
        [_canon(label or "") for label in labels_title],
        dtype=object,
    )

    scores_by_k = {k: [] for k in unique_k}

    for title, video_id in slate_pairs:
        surface = (
            title
            if title and title.strip() and title != "(untitled)"
            else (title_for(video_id) or video_id or "")
        )
        parts = list(base_parts)
        if surface:
            parts.append(_safe_str(surface, lowercase=lowercase))
        query_text = "\n".join(part for part in parts if part)
        query = vectorizer.transform([query_text])
        sims = (query @ matrix.T).toarray().ravel()

        vid_canon = _canon_vid(video_id or "")
        title_canon = _canon(title or "")
        mask = np.zeros_like(sims, dtype=bool)
        if vid_canon:
            mask |= label_id_canon == vid_canon
        if title_canon:
            mask |= label_title_canon == title_canon
        if not mask.any():
            fallback = float(sims.max() * 0.01) if sims.size else 0.0
            for k in unique_k:
                scores_by_k[k].append(fallback)
            continue
        sims_masked = sims[mask]
        if sims_masked.size == 0:
            for k in unique_k:
                scores_by_k[k].append(0.0)
            continue
        if metric == "cosine" or metric is None:
            score_vector = sims_masked
        else:
            subset = matrix[mask].astype(np.float32)
            diff = subset - query
            dists = np.sqrt(diff.power(2).sum(axis=1)).A.ravel()
            score_vector = -dists
        sorted_scores = np.sort(score_vector)[::-1]
        cumulative = np.cumsum(sorted_scores)
        for k in unique_k:
            kk = int(min(max(1, k), sorted_scores.size))
            scores_by_k[k].append(float(cumulative[kk - 1]))

    predictions: Dict[int, Optional[int]] = {}
    for k, scores in scores_by_k.items():
        if not scores or not any(np.isfinite(scores)):
            predictions[k] = None
        else:
            predictions[k] = int(np.argmax(scores)) + 1
    return predictions


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
