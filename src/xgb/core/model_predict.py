#!/usr/bin/env python
"""Model persistence and slate prediction helpers for the XGBoost baseline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from common.evaluation.matrix_summary import log_single_embedding

from ._optional import LabelEncoder, joblib
from .utils import canon_video_id
from .vectorizers import BaseTextVectorizer, load_vectorizer
from . import features as feature_utils
from .model_types import XGBoostSlateModel


LOGGER = logging.getLogger("xgb.model")
_embed_log_state = {"printed_online": False}


def save_xgboost_model(model: XGBoostSlateModel, out_dir: Path | str) -> None:
    """
    Persist a trained XGBoost model bundle to ``out_dir``.

    :param model: Model bundle produced by :func:`fit_xgboost_model`.
    :type model: XGBoostSlateModel
    :param out_dir: Directory path where the bundle should be written.
    :type out_dir: Path or str
    """

    if joblib is None:
        raise ImportError("Install joblib to save the XGBoost baseline artifacts.")
    if LabelEncoder is None:  # pragma: no cover - optional dependency
        raise ImportError("Install scikit-learn to serialize XGBoost label encoders.")

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    vectorizer_dir = directory / "vectorizer"
    model.vectorizer.save(vectorizer_dir)
    joblib.dump(model.label_encoder, directory / "label_encoder.joblib")
    joblib.dump(model.booster, directory / "xgboost_model.joblib")
    with open(directory / "config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "extra_fields": list(model.extra_fields),
                "vectorizer": model.vectorizer.metadata(),
            },
            handle,
            indent=2,
        )
    if model.training_history:
        with open(directory / "training_history.json", "w", encoding="utf-8") as handle:
            json.dump(model.training_history, handle, indent=2)


def load_xgboost_model(in_dir: Path | str) -> XGBoostSlateModel:
    """
    Load an XGBoost model bundle previously saved to disk.

    :param in_dir: Directory containing the saved bundle.
    :type in_dir: Path or str
    :returns: Restored model bundle.
    :rtype: XGBoostSlateModel
    """

    if joblib is None:
        raise ImportError("Install joblib to load the XGBoost baseline artifacts.")
    if LabelEncoder is None:  # pragma: no cover - optional dependency
        raise ImportError("Install scikit-learn to load the XGBoost baseline artifacts.")

    directory = Path(in_dir)
    vectorizer_dir = directory / "vectorizer"
    if vectorizer_dir.exists():
        vectorizer = load_vectorizer(vectorizer_dir)
    else:  # Backwards compatibility with pre-vectorizer refactor bundles.
        vectorizer = joblib.load(directory / "vectorizer.joblib")
    encoder: LabelEncoder = joblib.load(directory / "label_encoder.joblib")
    booster = joblib.load(directory / "xgboost_model.joblib")
    config_path = directory / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        extra_fields = tuple(payload.get("extra_fields", []))
    else:
        extra_fields = ()
    training_history = None
    history_path = directory / "training_history.json"
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as handle:
                loaded_history = json.load(handle)
            if isinstance(loaded_history, dict):
                training_history = loaded_history
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            LOGGER.warning("Unable to load training history from %s.", history_path)
    return XGBoostSlateModel(
        vectorizer=vectorizer,
        label_encoder=encoder,
        booster=booster,
        extra_fields=extra_fields,
        training_history=training_history,
    )


def predict_among_slate(
    model: XGBoostSlateModel,
    example: dict,
    *,
    extra_fields: Sequence[str] | None = None,
) -> Tuple[Optional[int], Dict[str, float]]:
    """
    Predict which slate option should be selected for ``example``.

    :param model: Trained model bundle used for inference.
    :type model: XGBoostSlateModel
    :param example: Interaction row containing prompt text and candidate slate.
    :type example: dict
    :param extra_fields: Optional override of additional columns appended during inference.
    :type extra_fields: Sequence[str], optional
    :returns: Tuple with a 1-based predicted option index (``None`` when unknown) and
        per-video probabilities.
    :rtype: Tuple[Optional[int], Dict[str, float]]
    """

    extra_fields = tuple(extra_fields) if extra_fields is not None else model.extra_fields
    document = feature_utils.assemble_document(example, extra_fields)
    if not document.strip():
        return None, {}

    row_matrix = model.vectorizer.transform([document])
    # Log an embedding summary at inference time (covers the load_model path).
    if not _embed_log_state["printed_online"]:
        log_single_embedding(row_matrix, logger=LOGGER, tag="[XGB][Embed][Online]")
        _embed_log_state["printed_online"] = True
    proba = model.booster.predict_proba(row_matrix)
    if proba.ndim != 2 or proba.shape[0] == 0:
        return None, {}

    class_probs = proba[0]
    classes = model.label_encoder.classes_
    probability_map = {cls: float(prob) for cls, prob in zip(classes, class_probs)}

    slate_pairs = list(feature_utils.extract_slate_items(example))
    best_index = _select_best_candidate(slate_pairs, probability_map)
    if best_index is None and slate_pairs:
        # Open-set fallback using vector-space similarity between prompt and candidate title.
        best_index = _open_set_best_index(
            model.vectorizer,
            row_matrix,
            document,
            slate_pairs,
        )
    return best_index, probability_map


def _l2_norm(mat) -> float:
    """Return the L2 norm for dense or sparse vectors, safely.

    Falls back to ``0.0`` when inputs are malformed or operations fail.
    """
    try:
        if hasattr(mat, "multiply") and hasattr(mat, "sum"):
            return float(np.sqrt(mat.multiply(mat).sum()))
        arr = np.asarray(mat).ravel()
        return float(np.sqrt(float(np.dot(arr, arr))))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _open_set_best_index(
    vectorizer: BaseTextVectorizer,
    prompt_vector: Any,
    document: str,
    slate_pairs: Sequence[tuple[str, str]],
) -> Optional[int]:
    """Return the index of the most similar candidate by cosine similarity.

    Reuses ``prompt_vector`` when supplied to avoid recomputing embeddings.
    Falls back to encoding ``document`` on demand so call sites remain tolerant
    to ``None`` inputs. Dot products are used when cosine norms cannot be
    established.
    """
    try:
        doc_vec = prompt_vector if prompt_vector is not None else vectorizer.transform([document])
    except (ValueError, TypeError, AttributeError, RuntimeError):
        return None

    doc_norm = _l2_norm(doc_vec)
    best: Optional[tuple[float, int]] = None
    for idx, (title, vid) in enumerate(slate_pairs, start=1):
        text = feature_utils.title_for(vid) or title or ""
        if not text.strip():
            continue
        try:
            cand_vec = vectorizer.transform([text])
        except (ValueError, TypeError, AttributeError, RuntimeError):
            continue
        try:
            if hasattr(doc_vec, "multiply") and hasattr(cand_vec, "sum"):
                score = float(doc_vec.multiply(cand_vec).sum())
            else:
                score = float(
                    np.dot(np.asarray(doc_vec).ravel(), np.asarray(cand_vec).ravel())
                )
            norm = doc_norm * _l2_norm(cand_vec)
            score = (score / norm) if norm > 0 else score
        except (ValueError, TypeError, AttributeError):
            # Treat failures as worst similarity to avoid selecting this candidate.
            score = -float("inf")
        # Avoid subscripting when best is None to satisfy type checkers/pylint.
        if best is None:
            best = (score, idx)
        elif score > best[0]:
            best = (score, idx)
    return None if best is None else best[1]


def _select_best_candidate(
    slate_pairs: Sequence[tuple[str, str]],
    probability_map: Dict[str, float],
) -> Optional[int]:
    """
    Select the highest-scoring slate option using primary and fallback keys.

    :param slate_pairs: Sequence of ``(title, video_id)`` pairs representing the slate.
    :type slate_pairs: Sequence[tuple[str, str]]
    :param probability_map: Probability lookup keyed by canonical video id.
    :type probability_map: Dict[str, float]
    :returns: 1-based index of the preferred candidate, or ``None`` when unavailable.
    :rtype: Optional[int]
    """
    primary = _best_index_by_key(slate_pairs, probability_map, _candidate_id_key)
    if primary is not None:
        return primary
    return _best_index_by_key(slate_pairs, probability_map, _fallback_candidate_key)


def _best_index_by_key(
    slate_pairs: Sequence[tuple[str, str]],
    probability_map: Dict[str, float],
    key_fn: Callable[[str, str], str],
) -> Optional[int]:
    """
    Return the index of the candidate maximising ``probability_map`` lookups.

    :param slate_pairs: Sequence of ``(title, video_id)`` pairs representing the slate.
    :type slate_pairs: Sequence[tuple[str, str]]
    :param probability_map: Probability lookup keyed by canonical identifiers.
    :type probability_map: Dict[str, float]
    :param key_fn: Callable deriving lookup keys from candidate metadata.
    :type key_fn: Callable[[str, str], str]
    :returns: 1-based best index or ``None`` when no candidate receives a score.
    :rtype: Optional[int]
    """
    best_index: Optional[int] = None
    best_score = -np.inf
    for idx, (title, video_id) in enumerate(slate_pairs, start=1):
        key = key_fn(title, video_id)
        score = probability_map.get(key)
        if score is None or score <= best_score:
            continue
        best_score = score
        best_index = idx
    return best_index


def _candidate_id_key(title: str, video_id: str) -> str:
    """
    Derive the canonical lookup key for a candidate video.

    :param title: Candidate title extracted from the slate.
    :type title: str
    :param video_id: Raw video identifier associated with the candidate.
    :type video_id: str
    :returns: Canonical identifier prioritising ``video_id`` and falling back to ``title``.
    :rtype: str
    """
    return canon_video_id(video_id) or canon_video_id(title)


def _fallback_candidate_key(title: str, video_id: str) -> str:
    """
    Derive a fallback lookup key when the primary identifier is missing.

    :param title: Candidate title extracted from the slate.
    :type title: str
    :param video_id: Raw video identifier associated with the candidate.
    :type video_id: str
    :returns: Canonical identifier based on ID, title lookup, or derived title text.
    :rtype: str
    """
    candidate = canon_video_id(video_id)
    if candidate:
        return candidate
    if title:
        title_candidate = canon_video_id(title)
        if title_candidate:
            return title_candidate
    derived = feature_utils.title_for(video_id) or title or ""
    return canon_video_id(derived) or derived.strip()


__all__ = [
    "save_xgboost_model",
    "load_xgboost_model",
    "predict_among_slate",
]
