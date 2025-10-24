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

"""Model training helpers for the Grail Simulation XGBoost baseline."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ._optional import LabelEncoder, joblib

from .features import assemble_document, extract_slate_items, prepare_prompt_documents, title_for
from .utils import canon_video_id
from .vectorizers import (
    BaseTextVectorizer,
    SentenceTransformerVectorizerConfig,
    TfidfConfig,
    Word2VecVectorizerConfig,
    create_vectorizer,
    load_vectorizer,
)


def _ensure_label_encoder_available(action: str) -> None:
    """
    Guard helper ensuring scikit-learn is installed before continuing.

    :param action: Short description of the attempted action for error messaging.
    :type action: str
    :raises ImportError: If scikit-learn's :class:`~sklearn.preprocessing.LabelEncoder`
        is unavailable.
    """
    if LabelEncoder is None:  # pragma: no cover - optional dependency
        raise ImportError(f"Install scikit-learn to {action}.")

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore

LOGGER = logging.getLogger("xgb.model")
_gpu_training_state = {"enabled": True}


def _should_retry_on_cpu(tree_method: str, exc: Exception) -> bool:
    """
    Determine whether a GPU-specific failure warrants retrying on CPU.

    :param tree_method: Requested tree construction algorithm.
    :type tree_method: str
    :param exc: Exception raised during booster training.
    :type exc: Exception
    :returns: ``True`` when the failure appears related to missing GPU support.
    :rtype: bool
    """

    if not tree_method or not str(tree_method).lower().startswith("gpu"):
        return False
    message = str(exc)
    lowered = message.lower()
    gpu_indicators = (
        "cudaerrorinitializationerror",
        "cudaerrornodevice",
        "cuda driver version",
        "cuda driver initialization failed",
        "gpu is not enabled",
    )
    return any(token in lowered for token in gpu_indicators)

@dataclass
class XGBoostSlateModel:
    """
    Container bundling the vectoriser, label encoder, and trained model.

    :ivar vectorizer: Text vectoriser fitted on training documents.
    :vartype vectorizer: BaseTextVectorizer
    :ivar label_encoder: Encoder mapping video identifiers to numeric labels.
    :vartype label_encoder: sklearn.preprocessing.LabelEncoder
    :ivar booster: Trained XGBoost booster instance.
    :vartype booster: Any
    :ivar extra_fields: Additional prompt fields captured during training.
    :vartype extra_fields: Tuple[str, ...]
    """

    vectorizer: BaseTextVectorizer
    label_encoder: LabelEncoder
    booster: Any
    extra_fields: Tuple[str, ...]


# pylint: disable=too-many-instance-attributes
@dataclass
class XGBoostBoosterParams:
    """
    XGBoost-specific hyper-parameters used when instantiating the booster.

    :ivar learning_rate: Gradient boosting learning rate.
    :vartype learning_rate: float
    :ivar max_depth: Maximum tree depth.
    :vartype max_depth: int
    :ivar n_estimators: Number of boosting rounds.
    :vartype n_estimators: int
    :ivar subsample: Row subsampling ratio applied per boosting round.
    :vartype subsample: float
    :ivar colsample_bytree: Column subsampling ratio applied per tree.
    :vartype colsample_bytree: float
    :ivar tree_method: Tree construction algorithm name.
    :vartype tree_method: str
    :ivar reg_lambda: L2 regularisation weight.
    :vartype reg_lambda: float
    :ivar reg_alpha: L1 regularisation weight.
    :vartype reg_alpha: float
    :ivar extra_kwargs: Additional keyword arguments forwarded to
        :class:`xgboost.XGBClassifier`.
    :vartype extra_kwargs: Dict[str, Any] | None
    """

    learning_rate: float = 0.1
    max_depth: int = 6
    n_estimators: int = 300
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    tree_method: str = "hist"
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    extra_kwargs: Dict[str, Any] | None = None


# pylint: disable=too-many-instance-attributes
@dataclass
class XGBoostTrainConfig:
    """
    Configuration options controlling training behaviour.

    :ivar max_train: Maximum number of rows sampled from the training split (0 keeps all).
    :vartype max_train: int
    :ivar seed: Random seed used for subsampling.
    :vartype seed: int
    :ivar max_features: Maximum number of TF-IDF features (``None`` for unlimited).
    :vartype max_features: int | None
    :ivar vectorizer_kind: Feature extraction strategy (``tfidf``, ``word2vec``,
        or ``sentence_transformer``).
    :vartype vectorizer_kind: str
    :ivar booster: Hyper-parameter bundle applied when initialising the booster.
    :vartype booster: XGBoostBoosterParams
    """

    max_train: int = 200_000
    seed: int = 42
    max_features: Optional[int] = 200_000
    vectorizer_kind: str = "tfidf"
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    word2vec: Word2VecVectorizerConfig = field(default_factory=Word2VecVectorizerConfig)
    sentence_transformer: SentenceTransformerVectorizerConfig = field(
        default_factory=SentenceTransformerVectorizerConfig
    )
    booster: XGBoostBoosterParams = field(default_factory=XGBoostBoosterParams)


def fit_xgboost_model(
    train_ds,
    *,
    config: Optional[XGBoostTrainConfig] = None,
    extra_fields: Sequence[str] | None = None,
) -> XGBoostSlateModel:
    """
    Train an XGBoost multi-class classifier over prompt documents.

    :param train_ds: Dataset split providing training examples.
    :type train_ds: datasets.Dataset or sequence-like
    :param config: Hyper-parameter bundle controlling training behaviour, including
        subsampling limits and booster hyper-parameters.
    :type config: XGBoostTrainConfig, optional
    :param extra_fields: Optional column names appended to the prompt document.
    :type extra_fields: Sequence[str], optional
    :returns: Trained model bundle containing vectoriser, label encoder, and booster.
    :rtype: XGBoostSlateModel
    :raises ImportError: If the upstream :mod:`xgboost` package is unavailable.
    :raises RuntimeError: When the training split is empty or lacks diverse labels.
    """

    if XGBClassifier is None:  # pragma: no cover - optional dependency
        raise ImportError("Install xgboost to train the XGBoost baseline.")
    _ensure_label_encoder_available("train the XGBoost baseline")

    train_config = config or XGBoostTrainConfig()
    train_config.vectorizer_kind = (train_config.vectorizer_kind or "tfidf").lower()
    capped = (
        train_config.max_features
        if train_config.max_features and train_config.max_features > 0
        else None
    )
    train_config.tfidf.max_features = capped
    vectorizer, encoder, booster = _build_model_components(
        train_ds=train_ds,
        train_config=train_config,
        extra_fields=extra_fields,
    )

    return XGBoostSlateModel(
        vectorizer=vectorizer,
        label_encoder=encoder,
        booster=booster,
        extra_fields=tuple(extra_fields or ()),
    )


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
    _ensure_label_encoder_available("serialize XGBoost label encoders")

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
    _ensure_label_encoder_available("load the XGBoost baseline artifacts")

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

    slate_pairs = list(extract_slate_items(example))
    best_index = _select_best_candidate(slate_pairs, probability_map)
    return best_index, probability_map


def _encode_labels(labels_id: Sequence[str]) -> Tuple[LabelEncoder, Any]:
    """
    Fit a label encoder and transform the provided identifiers.

    :param labels_id: Collection of canonical video identifiers.
    :type labels_id: Sequence[str]
    :returns: Tuple of ``(encoder, encoded_labels)``.
    :rtype: Tuple[LabelEncoder, Any]
    :raises RuntimeError: If fewer than two unique labels are supplied.
    """
    _ensure_label_encoder_available("encode prompt labels for XGBoost")
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels_id)
    if len(encoder.classes_) < 2:
        raise RuntimeError(
            "Training labels must contain at least two unique gold video ids for XGBoost."
        )
    return encoder, encoded


def _train_booster(
    *,
    train_config: XGBoostTrainConfig,
    matrix,
    labels,
) -> Any:
    """
    Instantiate and fit the XGBoost booster component.

    :param train_config: Training configuration containing booster hyper-parameters.
    :type train_config: XGBoostTrainConfig
    :param matrix: Feature matrix produced by the text vectoriser.
    :type matrix: Any
    :param labels: Numeric labels aligned with ``matrix`` rows.
    :type labels: Any
    :returns: Fitted XGBoost classifier implementing ``predict_proba``.
    :rtype: Any
    :raises Exception: Propagated when booster training fails on both GPU and CPU.
    """
    params = train_config.booster
    booster_kwargs = dict(params.extra_kwargs or {})
    unique_labels = int(np.unique(labels).size)
    booster_kwargs.setdefault("num_class", unique_labels)

    requested_method = params.tree_method or "hist"
    use_gpu = requested_method.lower().startswith("gpu") and _gpu_training_state["enabled"]
    effective_method = requested_method if use_gpu else (
        "hist" if requested_method.lower().startswith("gpu") else requested_method
    )

    booster = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        tree_method=effective_method,
        reg_lambda=params.reg_lambda,
        reg_alpha=params.reg_alpha,
        random_state=train_config.seed,
        nthread=-1,
        **booster_kwargs,
    )
    try:
        booster.fit(matrix, labels)
        return booster
    except Exception as exc:  # pragma: no cover - runtime guard
        if use_gpu and _should_retry_on_cpu(requested_method, exc):
            LOGGER.warning(
                "XGBoost GPU training failed with '%s'. Retrying current task on CPU.",
                exc,
            )
            _disable_gpu_boosters()
            cpu_kwargs = _scrub_gpu_kwargs(booster_kwargs)
            cpu_booster = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                learning_rate=params.learning_rate,
                subsample=params.subsample,
                colsample_bytree=params.colsample_bytree,
                tree_method="hist",
                reg_lambda=params.reg_lambda,
                reg_alpha=params.reg_alpha,
                random_state=train_config.seed,
                nthread=-1,
                **cpu_kwargs,
            )
            cpu_booster.fit(matrix, labels)
            return cpu_booster
        raise


def _disable_gpu_boosters() -> None:
    """
    Disable GPU-backed boosters for the remainder of the process.

    Invoked after repeated GPU failures to prevent recurring retries.
    """

    if _gpu_training_state["enabled"]:
        LOGGER.warning("Disabling XGBoost GPU boosters for the remainder of this process.")
    _gpu_training_state["enabled"] = False


def _scrub_gpu_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove GPU-specific keyword arguments from a booster configuration.

    :param kwargs: Keyword arguments forwarded to :class:`xgboost.XGBClassifier`.
    :type kwargs: Dict[str, Any]
    :returns: Copy of ``kwargs`` without GPU-specific options.
    :rtype: Dict[str, Any]
    """

    cleaned = dict(kwargs)
    cleaned.pop("gpu_id", None)
    if cleaned.get("predictor", "").startswith("gpu"):
        cleaned.pop("predictor", None)
    if cleaned.get("tree_method", "").startswith("gpu"):
        cleaned.pop("tree_method", None)
    return cleaned


def _build_model_components(
    *,
    train_ds,
    train_config: XGBoostTrainConfig,
    extra_fields: Sequence[str] | None,
) -> Tuple[BaseTextVectorizer, LabelEncoder, Any]:
    """
    Construct the model components required for training and inference.

    :param train_ds: Dataset split providing training examples.
    :type train_ds: datasets.Dataset | Sequence[dict]
    :param train_config: Configuration bundle controlling sampling and hyper-parameters.
    :type train_config: XGBoostTrainConfig
    :param extra_fields: Additional prompt columns appended during featurisation.
    :type extra_fields: Sequence[str] | None
    :returns: Tuple of ``(vectorizer, label_encoder, booster)``.
    :rtype: Tuple[BaseTextVectorizer, LabelEncoder, Any]
    :raises RuntimeError: If no documents are produced for training.
    """

    docs, labels_id, _ = prepare_prompt_documents(
        train_ds,
        max_train=train_config.max_train,
        seed=train_config.seed,
        extra_fields=extra_fields,
    )
    if not docs:
        raise RuntimeError("No training documents were produced for XGBoost fitting.")

    vectorizer = create_vectorizer(
        train_config.vectorizer_kind,
        tfidf=train_config.tfidf,
        word2vec=train_config.word2vec,
        sentence_transformer=train_config.sentence_transformer,
    )
    matrix = vectorizer.fit_transform(docs)
    if hasattr(matrix, "astype"):
        matrix = matrix.astype(np.float32, copy=False)  # type: ignore[assignment]
    encoder, encoded_labels = _encode_labels(labels_id)
    booster = _train_booster(
        train_config=train_config,
        matrix=matrix,
        labels=encoded_labels,
    )
    return vectorizer, encoder, booster


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
    best_score = -math.inf
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
    derived = title_for(video_id) or title or ""
    return canon_video_id(derived) or derived.strip()


__all__ = [
    "XGBoostSlateModel",
    "XGBoostBoosterParams",
    "XGBoostTrainConfig",
    "TfidfConfig",
    "Word2VecVectorizerConfig",
    "SentenceTransformerVectorizerConfig",
    "fit_xgboost_model",
    "load_xgboost_model",
    "predict_among_slate",
    "save_xgboost_model",
]
