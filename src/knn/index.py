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

from common.embeddings import SentenceTransformerConfig, SentenceTransformerEncoder

from .features import (
    assemble_document,
    extract_slate_items,
    prepare_training_documents,
    Word2VecConfig,
    Word2VecFeatureBuilder,
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
    """Pre-computed index artifacts used during slate scoring."""

    feature_space: str
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

    :param train_ds: Training dataset split (Hugging Face dataset or list-like).
    :param max_train: Maximum number of rows to sample for fitting.
    :param seed: Random seed used for subsampling.
    :param max_features: Optional vocabulary cap for the vectoriser.
    :param extra_fields: Extra textual columns concatenated into each document.
    :returns: Dictionary containing the fitted vectoriser, sparse matrix, and
        label metadata.
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
        "feature_space": "tfidf",
        "vectorizer": vectorizer,
        "X": matrix,
        "labels_id": labels_id,
        "labels_title": labels_title,
    }


def build_word2vec_index(
    train_ds,
    *,
    max_train: int = 100_000,
    seed: int = 42,
    extra_fields: Sequence[str] | None = None,
    config: Word2VecConfig | None = None,
):
    """Create a Word2Vec index from the training split.

    :param train_ds: Training dataset split.
    :param max_train: Maximum number of rows to sample for fitting.
    :param seed: Random seed used for subsampling.
    :param extra_fields: Extra textual columns concatenated into each document.
    :param config: Optional :class:`~knn.features.Word2VecConfig` override.
    :returns: Dictionary containing the trained Word2Vec model and embeddings.
    :raises ImportError: If gensim is unavailable.
    """

    docs, labels_id, labels_title = prepare_training_documents(
        train_ds,
        max_train=max_train,
        seed=seed,
        extra_fields=extra_fields,
    )

    builder = Word2VecFeatureBuilder(config)
    try:
        builder.train(docs)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Install gensim to enable Word2Vec embeddings (pip install gensim)"
        ) from exc

    matrix = builder.transform(docs)

    return {
        "feature_space": "word2vec",
        "vectorizer": builder,
        "X": matrix,
        "labels_id": labels_id,
        "labels_title": labels_title,
        "word2vec_config": builder.config,
    }


def build_sentence_transformer_index(
    train_ds,
    *,
    max_train: int = 100_000,
    seed: int = 42,
    extra_fields: Sequence[str] | None = None,
    config: SentenceTransformerConfig | None = None,
):
    """Create a sentence-transformer index from the training split.

    :param train_ds: Training dataset split.
    :param max_train: Maximum number of rows to sample.
    :param seed: Random seed used for subsampling.
    :param extra_fields: Extra textual columns concatenated into each document.
    :param config: Optional :class:`~common.embeddings.SentenceTransformerConfig` override.
    :returns: Dictionary containing the encoder, dense matrix, and label metadata.
    :raises ImportError: If ``sentence_transformers`` is unavailable.
    """

    docs, labels_id, labels_title = prepare_training_documents(
        train_ds,
        max_train=max_train,
        seed=seed,
        extra_fields=extra_fields,
    )
    st_config = config or SentenceTransformerConfig()
    encoder = SentenceTransformerEncoder(st_config)
    if not hasattr(encoder, "transform"):
        setattr(encoder, "transform", encoder.encode)  # type: ignore[attr-defined]
    matrix = encoder.encode(docs).astype(np.float32, copy=False)
    return {
        "feature_space": "sentence_transformer",
        "vectorizer": encoder,
        "X": matrix,
        "labels_id": labels_id,
        "labels_title": labels_title,
        "sentence_transformer_config": st_config,
    }


def save_tfidf_index(index: Dict[str, Any], out_dir: str) -> None:
    """Persist the TF-IDF index to ``out_dir`` for later reuse.

    :param index: Dictionary returned by :func:`build_tfidf_index`.
    :param out_dir: Target directory where artifacts will be written.
    """

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
        "feature_space": "tfidf",
    }
    with open(directory / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def load_tfidf_index(in_dir: str) -> Dict[str, Any]:
    """Load a TF-IDF index previously saved via :func:`save_tfidf_index`.

    :param in_dir: Directory containing the persisted TF-IDF artifacts.
    :returns: Dictionary with the loaded vectoriser, matrix, and labels.
    """

    directory = Path(in_dir)
    vectorizer = joblib.load(directory / "vectorizer.joblib")
    matrix = sparse.load_npz(directory / "X.npz")
    labels_id = np.load(directory / "labels_id.npy", allow_pickle=True).tolist()
    labels_title = np.load(directory / "labels_title.npy", allow_pickle=True).tolist()
    return {
        "feature_space": "tfidf",
        "vectorizer": vectorizer,
        "X": matrix,
        "labels_id": labels_id,
        "labels_title": labels_title,
    }


def save_word2vec_index(index: Dict[str, Any], out_dir: str) -> None:
    """Persist the Word2Vec index to ``out_dir`` for later reuse.

    :param index: Dictionary returned by :func:`build_word2vec_index`.
    :param out_dir: Target directory where embeddings and metadata are stored.
    """

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    builder: Word2VecFeatureBuilder = index["vectorizer"]
    model_dir = directory / "word2vec_model"
    builder.save(model_dir)

    np.save(
        directory / "X.npy",
        np.asarray(index["X"], dtype=np.float32),
        allow_pickle=False,
    )
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

    config = getattr(index.get("word2vec_config"), "__dict__", {}) or {}
    meta = {
        "n_docs": int(index["X"].shape[0]),
        "vector_size": int(index["X"].shape[1]) if index["X"].size else 0,
        "feature_space": "word2vec",
        "config": {
            "vector_size": int(config.get("vector_size", 0)),
            "window": int(config.get("window", 5)),
            "min_count": int(config.get("min_count", 2)),
            "epochs": int(config.get("epochs", 10)),
            "seed": int(config.get("seed", 42)),
            "workers": int(config.get("workers", 1)),
        },
    }
    with open(directory / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def save_sentence_transformer_index(index: Dict[str, Any], out_dir: str) -> None:
    """Persist the sentence-transformer index to ``out_dir`` for later reuse."""

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(index["X"], dtype=np.float32)
    np.save(directory / "X.npy", matrix, allow_pickle=False)
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

    config_obj: SentenceTransformerConfig = index.get("sentence_transformer_config") or SentenceTransformerConfig()
    meta = {
        "feature_space": "sentence_transformer",
        "n_docs": int(matrix.shape[0]),
        "vector_size": int(matrix.shape[1]) if matrix.size else 0,
        "config": {
            "model_name": config_obj.model_name,
            "device": config_obj.device,
            "batch_size": int(config_obj.batch_size),
            "normalize": bool(config_obj.normalize),
        },
    }
    with open(directory / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def load_word2vec_index(in_dir: str) -> Dict[str, Any]:
    """Load a Word2Vec index previously saved via :func:`save_word2vec_index`.

    :param in_dir: Directory containing the persisted Word2Vec artifacts.
    :returns: Dictionary with the restored Word2Vec model, embeddings, and labels.
    """

    directory = Path(in_dir)

    meta_path = directory / "meta.json"
    config_kwargs: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
            config_kwargs = meta.get("config", {})

    config = Word2VecConfig(
        vector_size=int(config_kwargs.get("vector_size", 256)),
        window=int(config_kwargs.get("window", 5)),
        min_count=int(config_kwargs.get("min_count", 2)),
        epochs=int(config_kwargs.get("epochs", 10)),
        model_dir=directory / "word2vec_model",
        seed=int(config_kwargs.get("seed", 42)),
        workers=int(config_kwargs.get("workers", 1)),
    )
    builder = Word2VecFeatureBuilder(config)
    builder.load(config.model_dir)

    matrix = np.load(directory / "X.npy")
    labels_id = np.load(directory / "labels_id.npy", allow_pickle=True).tolist()
    labels_title = np.load(directory / "labels_title.npy", allow_pickle=True).tolist()

    return {
        "feature_space": "word2vec",
        "vectorizer": builder,
        "X": matrix,
        "labels_id": labels_id,
        "labels_title": labels_title,
        "word2vec_config": config,
    }


def load_sentence_transformer_index(in_dir: str) -> Dict[str, Any]:
    """Load a sentence-transformer index saved via :func:`save_sentence_transformer_index`."""

    directory = Path(in_dir)
    matrix = np.load(directory / "X.npy").astype(np.float32, copy=False)
    labels_id = np.load(directory / "labels_id.npy", allow_pickle=True).tolist()
    labels_title = np.load(directory / "labels_title.npy", allow_pickle=True).tolist()
    meta_path = directory / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        config = SentenceTransformerConfig(**meta.get("config", {}))
    else:
        config = SentenceTransformerConfig()
    encoder = SentenceTransformerEncoder(config)
    if not hasattr(encoder, "transform"):
        setattr(encoder, "transform", encoder.encode)  # type: ignore[attr-defined]
    return {
        "feature_space": "sentence_transformer",
        "vectorizer": encoder,
        "X": matrix,
        "labels_id": labels_id,
        "labels_title": labels_title,
        "sentence_transformer_config": config,
    }


def _safe_str(value: Any, *, lowercase: bool = True) -> str:
    """Return a normalised string representation of ``value``.

    :param value: Arbitrary object to convert into a string.
    :param lowercase: Whether the result should be lowercased.
    :returns: Stripped string representation safe for tokenisation.
    """

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
    """Assemble the base text parts used to score slate candidates.

    :param example: Dataset row containing viewer prompt information.
    :param slate_pairs: Sequence of ``(title, video_id)`` tuples.
    :param config: Query configuration controlling casing and extra fields.
    :returns: List of textual fragments concatenated during scoring.
    """
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
    """Return the most human-friendly representation for a candidate surface.

    :param title: Title associated with the candidate (may be empty).
    :param video_id: Canonical video identifier.
    :returns: Preferential title to use when building query context.
    """
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
    """Score every slate candidate for each ``k`` value.

    :param index_data: Loaded index data required for similarity lookups.
    :param slate_pairs: Sequence of slate candidates as ``(title, video_id)``.
    :param base_parts: Common text fragments shared across candidates.
    :param unique_k: Sorted ``k`` values evaluated for the slate.
    :param config: Query configuration instance.
    :returns: Mapping from ``k`` to the list of scores per candidate.
    """
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
    """Return the transformed query vector and raw similarities for a candidate.

    :param index_data: Prepared index data (vectoriser + matrix + labels).
    :param base_parts: Text fragments shared across candidates for the slate.
    :param title: Candidate title.
    :param video_id: Candidate video identifier.
    :param config: Query configuration controlling casing and metric.
    :returns: Tuple of ``(query_vector, similarity_array)``.
    """
    surface = _surface_text(title, video_id)
    parts = list(base_parts)
    if surface:
        parts.append(_safe_str(surface, lowercase=config.lowercase))
    query_text = "\n".join(part for part in parts if part)

    if index_data.feature_space in {"word2vec", "sentence_transformer"}:
        query_matrix = index_data.vectorizer.transform([query_text])
        query_vec = np.asarray(query_matrix[0], dtype=np.float32)
        sims = index_data.matrix @ query_vec
        return query_vec, sims

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
    """Return a scoring vector based on the configured distance metric.

    :param sims_masked: Similarity values restricted to matching rows.
    :param index_data: Index data providing matrices for distance computation.
    :param mask: Boolean mask selecting rows matching the candidate.
    :param query: Query vector for the candidate.
    :param metric: Distance metric (``l2`` or ``cosine``) or ``None`` for dot product.
    :returns: Vector of scores aligned with ``sims_masked`` ordering.
    """
    if index_data.feature_space in {"word2vec", "sentence_transformer"}:
        subset = index_data.matrix[mask]
        if subset.ndim == 1 and subset.size > 0:
            subset = subset.reshape(1, -1)
        result = sims_masked
        if subset.size > 0:
            if metric == "l2":
                diff = subset - query
                if diff.ndim == 1:
                    diff = diff.reshape(1, -1)
                dists = np.linalg.norm(diff, axis=1)
                result = -dists
            elif metric == "cosine":
                query_norm = float(np.linalg.norm(query)) or 1.0
                subset_norms = np.linalg.norm(subset, axis=1)
                denom = np.maximum(subset_norms * query_norm, 1e-8)
                result = (subset @ query) / denom
        return result

    if metric == "l2":
        subset_sparse = index_data.matrix[mask]
        if subset_sparse.shape[0] == 0:
            return np.asarray([], dtype=np.float32)
        subset = subset_sparse.toarray().astype(np.float32, copy=False)
        query_dense = query.toarray().astype(np.float32, copy=False).ravel()
        diff = subset - query_dense
        dists = np.linalg.norm(diff, axis=1)
        return -dists
    return sims_masked


def _aggregate_scores(score_vector: np.ndarray, unique_k: Sequence[int]) -> Dict[int, float]:
    """Aggregate neighbour scores for each ``k`` value.

    :param score_vector: Sorted neighbour score vector for a candidate.
    :param unique_k: Sequence of ``k`` values to aggregate over.
    :returns: Mapping of ``k`` to the cumulative score of the top-``k`` neighbours.
    """
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
    """Return a boolean mask selecting rows that match the candidate.

    :param title: Candidate title text.
    :param video_id: Candidate video identifier.
    :param index_data: Index data including canonical label arrays.
    :returns: Boolean mask over training rows matching title or video id.
    """
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

    :param knn_index: Dictionary describing the fitted index artifacts.
    :param example: Dataset row containing the slate to score.
    :param k_values: Sequence of ``k`` values to evaluate.
    :param config: Optional :class:`~knn.index.SlateQueryConfig` overriding defaults.
    :returns: Mapping from ``k`` to the predicted 1-based option index. ``None``
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
    """Construct :class:`SlateIndexData` from a raw index dictionary.

    :param knn_index: Dictionary describing the fitted index artifacts.
    :returns: :class:`SlateIndexData` instance or ``None`` when the index is invalid.
    """

    if not knn_index or "vectorizer" not in knn_index:
        return None
    labels_id = np.asarray(knn_index["labels_id"], dtype=object)
    labels_title = np.asarray(knn_index["labels_title"], dtype=object)
    feature_space = str(knn_index.get("feature_space", "tfidf")).lower()
    return SlateIndexData(
        feature_space=feature_space,
        vectorizer=knn_index["vectorizer"],
        matrix=knn_index["X"],
        label_id_canon=np.asarray([_canon_vid(label) for label in labels_id], dtype=object),
        label_title_canon=np.asarray([_canon(label or "") for label in labels_title], dtype=object),
    )


def _canon(text: str) -> str:
    """Return a canonicalised alphanumeric string.

    :param text: Input string to canonicalise.
    :returns: Lowercased string with non-alphanumeric characters removed.
    """

    return re.sub(r"[^a-z0-9]+", "", (text or "").lower().strip())


def _canon_vid(value: str) -> str:
    """Return the canonical 11-character YouTube video identifier.

    :param value: Raw candidate identifier.
    :returns: Canonical video id or the stripped input when no match is found.
    """

    if not isinstance(value, str):
        return ""
    match = re.search(r"([A-Za-z0-9_-]{11})", value)
    return match.group(1) if match else value.strip()


__all__ = [
    "build_tfidf_index",
    "build_word2vec_index",
    "build_sentence_transformer_index",
    "knn_predict_among_slate_multi",
    "load_tfidf_index",
    "load_word2vec_index",
    "load_sentence_transformer_index",
    "save_tfidf_index",
    "save_word2vec_index",
    "save_sentence_transformer_index",
]
