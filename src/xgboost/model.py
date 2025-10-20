"""XGBoost training and inference utilities for slate prediction."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from .features import assemble_document, extract_slate_items, prepare_prompt_documents, title_for

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore

_YTID_RE = re.compile(r"([A-Za-z0-9_-]{11})")


def _canon_vid(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    match = _YTID_RE.search(value)
    return match.group(1) if match else value.strip()


@dataclass
class XGBoostSlateModel:
    """Container bundling the vectoriser, label encoder, and trained model."""

    vectorizer: TfidfVectorizer
    label_encoder: LabelEncoder
    booster: Any
    extra_fields: Tuple[str, ...]


def fit_xgboost_model(
    train_ds,
    *,
    max_train: int = 200_000,
    seed: int = 42,
    extra_fields: Sequence[str] | None = None,
    max_features: Optional[int] = 200_000,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    n_estimators: int = 300,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    tree_method: str = "hist",
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    **xgb_kwargs: Any,
) -> XGBoostSlateModel:
    """Train an XGBoost multi-class classifier over prompt documents."""

    if XGBClassifier is None:  # pragma: no cover - optional dependency
        raise ImportError("Install xgboost to train the XGBoost baseline.")

    docs, labels_id, _ = prepare_prompt_documents(
        train_ds,
        max_train=max_train,
        seed=seed,
        extra_fields=extra_fields,
    )
    if not docs:
        raise RuntimeError("No training documents were produced for XGBoost fitting.")

    tfidf_max_features = max_features if max_features and max_features > 0 else None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None,
        token_pattern=r"(?u)\b[\w\-]{2,}\b",
        max_features=tfidf_max_features,
    )
    matrix = vectorizer.fit_transform(docs)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_id)
    n_classes = len(encoder.classes_)
    if n_classes < 2:
        raise RuntimeError(
            "Training labels must contain at least two unique gold video ids for XGBoost."
        )

    booster = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method=tree_method,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        random_state=seed,
        nthread=-1,
        **xgb_kwargs,
    )
    booster.fit(matrix, y)

    return XGBoostSlateModel(
        vectorizer=vectorizer,
        label_encoder=encoder,
        booster=booster,
        extra_fields=tuple(extra_fields or ()),
    )


def save_xgboost_model(model: XGBoostSlateModel, out_dir: Path | str) -> None:
    """Persist a trained XGBoost model bundle to ``out_dir``."""

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.vectorizer, directory / "vectorizer.joblib")
    joblib.dump(model.label_encoder, directory / "label_encoder.joblib")
    joblib.dump(model.booster, directory / "xgboost_model.joblib")
    with open(directory / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"extra_fields": list(model.extra_fields)}, handle, indent=2)


def load_xgboost_model(in_dir: Path | str) -> XGBoostSlateModel:
    """Load an XGBoost model bundle previously saved to disk."""

    directory = Path(in_dir)
    vectorizer: TfidfVectorizer = joblib.load(directory / "vectorizer.joblib")
    encoder: LabelEncoder = joblib.load(directory / "label_encoder.joblib")
    booster = joblib.load(directory / "xgboost_model.joblib")
    config_path = directory / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        extra_fields = tuple(payload.get("extra_fields", []))
    else:
        extra_fields = ()
    return XGBoostSlateModel(
        vectorizer=vectorizer,
        label_encoder=encoder,
        booster=booster,
        extra_fields=extra_fields,
    )


def predict_among_slate(
    model: XGBoostSlateModel,
    example: dict,
    *,
    extra_fields: Sequence[str] | None = None,
) -> Tuple[Optional[int], Dict[str, float]]:
    """Predict which slate option should be selected for ``example``.

    Returns a tuple containing the 1-based index of the predicted option (``None`` when
    prediction is not possible) and a probability map keyed by canonical video id.
    """

    extra_fields = tuple(extra_fields) if extra_fields is not None else model.extra_fields
    document = assemble_document(example, extra_fields)
    if not document.strip():
        return None, {}

    row_matrix = model.vectorizer.transform([document])
    proba = model.booster.predict_proba(row_matrix)
    if proba.ndim != 2 or proba.shape[0] == 0:
        return None, {}

    class_probs = proba[0]
    classes = model.label_encoder.classes_
    probability_map = {cls: float(prob) for cls, prob in zip(classes, class_probs)}

    best_index = None
    best_score = -math.inf
    for idx, (_, video_id) in enumerate(extract_slate_items(example), start=1):
        canon = _canon_vid(video_id)
        score = probability_map.get(canon)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_index = idx

    if best_index is None:
        # Fall back to titles when ids were unseen during training.
        for idx, (title, video_id) in enumerate(extract_slate_items(example), start=1):
            candidate = _canon_vid(video_id) or _canon_vid(title_for(video_id) or "")
            score = probability_map.get(candidate)
            if score is not None and score > best_score:
                best_score = score
                best_index = idx

    return best_index, probability_map


__all__ = [
    "XGBoostSlateModel",
    "fit_xgboost_model",
    "load_xgboost_model",
    "predict_among_slate",
    "save_xgboost_model",
]
