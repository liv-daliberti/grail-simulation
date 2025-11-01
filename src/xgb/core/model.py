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

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from common.ml.xgb.callbacks import build_fit_callbacks
from common.ml.xgb.fit_utils import harmonize_fit_kwargs
from common.evaluation.matrix_summary import log_embedding_previews
from ._optional import LabelEncoder

from . import features as feature_utils
from .vectorizers import (
    BaseTextVectorizer,
    SentenceTransformerVectorizerConfig,
    TfidfConfig,
    Word2VecVectorizerConfig,
    create_vectorizer,
)
from .model_config import XGBoostBoosterParams, XGBoostTrainConfig
from .model_predict import (
    load_xgboost_model,
    predict_among_slate,
    save_xgboost_model,
)
from .model_types import (
    EncodedDataset,
    EvaluationArtifactsContext,
    TrainingBatch,
    XGBoostSlateModel,
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


def _ensure_float32(matrix: Any) -> Any:
    """Cast vectoriser outputs to ``float32`` when supported."""

    if hasattr(matrix, "astype"):
        return matrix.astype(
            np.float32,
            copy=False,
        )  # type: ignore[assignment]
    return matrix


# Types moved to model_types (EncodedDataset, TrainingBatch, EvaluationArtifactsContext,
# XGBoostSlateModel)

# Config dataclasses moved to model_config (XGBoostBoosterParams, XGBoostTrainConfig).


def fit_xgboost_model(
    train_ds,
    *,
    config: Optional[XGBoostTrainConfig] = None,
    extra_fields: Sequence[str] | None = None,
    eval_ds=None,
    collect_history: bool = False,
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
    :param eval_ds: Optional evaluation split monitored during fitting.
    :type eval_ds: datasets.Dataset | Sequence[dict] | None
    :param collect_history: When ``True`` record per-round training metrics.
    :type collect_history: bool
    :returns: Trained model bundle containing vectoriser, label encoder, booster, and history.
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
    vectorizer, encoder, booster, history = _build_model_components(
        train_ds=train_ds,
        train_config=train_config,
        extra_fields=extra_fields,
        eval_ds=eval_ds,
        collect_history=collect_history,
    )

    return XGBoostSlateModel(
        vectorizer=vectorizer,
        label_encoder=encoder,
        booster=booster,
        extra_fields=tuple(extra_fields or ()),
        training_history=history,
    )


# Persistence and prediction utilities moved to model_predict
    # Utilities moved to model_predict


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


def _instantiate_classifier(
    train_config: XGBoostTrainConfig,
    params: XGBoostBoosterParams,
    tree_method: str,
    extra_kwargs: Dict[str, Any],
) -> Any:
    """Create an :class:`xgboost.XGBClassifier` with shared defaults."""

    cleaned_kwargs = dict(extra_kwargs)
    cleaned_kwargs.pop("tree_method", None)
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        tree_method=tree_method,
        reg_lambda=params.reg_lambda,
        reg_alpha=params.reg_alpha,
        random_state=train_config.seed,
        nthread=-1,
        **cleaned_kwargs,
    )


def _build_fit_kwargs(collect_history: bool, batch: TrainingBatch) -> Dict[str, Any]:
    """Construct ``fit`` keyword arguments when evaluation tracking is enabled."""

    if not collect_history:
        return {}
    eval_set = [(batch.train.matrix, batch.train.labels)]
    if batch.evaluation is not None:
        eval_set.append((batch.evaluation.matrix, batch.evaluation.labels))
    fit_kwargs: Dict[str, Any] = {"eval_set": eval_set, "verbose": False}

    # Attach callbacks when available. Prefer the official EarlyStopping callback
    # (XGBoost >= 2.0) and fall back to logging-only when unavailable. A separate
    # fallback to the legacy ``early_stopping_rounds`` kwarg is handled in
    # ``_train_booster`` to keep compatibility with multiple XGBoost versions.
    # Attach logging and EarlyStopping via shared helpers when available.
    callbacks = build_fit_callbacks(
        objective="classification",
        logger=LOGGER,
        prefix="[XGB][Train]",
        has_eval=batch.evaluation is not None,
        interval=25,
        early_stopping_metric="mlogloss",
    )
    if callbacks:
        fit_kwargs["callbacks"] = callbacks
    return fit_kwargs


def _ensure_history_metrics(booster: Any, *, collect_history: bool) -> None:
    """
    Ensure the booster tracks accuracy metrics when evaluation history is requested.
    """

    if not collect_history or not hasattr(booster, "get_xgb_params"):
        return
    params = booster.get_xgb_params()
    metrics = params.get("eval_metric")
    # Respect callable metrics by leaving them untouched.
    if callable(metrics):
        return
    if metrics is None:
        booster.set_params(eval_metric=["mlogloss", "merror"])
        return
    if isinstance(metrics, str):
        metrics_list = [metrics]
    elif isinstance(metrics, (list, tuple)):
        metrics_list = list(metrics)
    else:  # Unsupported type; do not modify.
        return
    required = ["mlogloss", "merror"]
    missing = [metric for metric in required if metric not in metrics_list]
    if missing:
        booster.set_params(eval_metric=metrics_list + missing)


def _collect_training_history(booster, collect_history: bool) -> Optional[Dict[str, Any]]:
    """Return evaluation history when requested."""

    if not collect_history:
        return None
    return booster.evals_result()


def _train_booster(
    *,
    train_config: XGBoostTrainConfig,
    batch: TrainingBatch,
    collect_history: bool = False,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Instantiate and fit the XGBoost booster component.

    :param train_config: Training configuration containing booster hyper-parameters.
    :type train_config: XGBoostTrainConfig
    :param batch: Encoded training data and optional evaluation split.
    :type batch: TrainingBatch
    :param collect_history: When ``True`` record per-round metrics for plotting.
    :type collect_history: bool
    :returns: Tuple of fitted classifier and optional evaluation history.
    :rtype: Tuple[Any, Dict[str, Any] | None]
    :raises Exception: Propagated when booster training fails on both GPU and CPU.
    """
    params = train_config.booster
    booster_kwargs = dict(params.extra_kwargs or {})
    booster_kwargs.setdefault("num_class", int(np.unique(batch.train.labels).size))

    requested_method = params.tree_method or "hist"
    requested_lower = requested_method.lower()
    attempt_gpu = requested_lower.startswith("gpu") and _gpu_training_state["enabled"]
    tree_method = requested_method if attempt_gpu else (
        "hist" if requested_lower.startswith("gpu") else requested_method
    )

    booster = _instantiate_classifier(
        train_config,
        params,
        tree_method,
        booster_kwargs,
    )
    fit_kwargs = _build_fit_kwargs(collect_history, batch)
    _ensure_history_metrics(booster, collect_history=collect_history)
    # Ensure compatibility across XGBoost versions: only pass supported kwargs
    # and fall back to legacy early_stopping_rounds when callbacks are not
    # supported on the estimator.
    _harmonize_classifier_fit_kwargs(booster, batch, fit_kwargs)
    try:
        booster.fit(batch.train.matrix, batch.train.labels, **fit_kwargs)
        return booster, _collect_training_history(booster, bool(fit_kwargs))
    except Exception as exc:  # pragma: no cover - runtime guard
        if attempt_gpu and _should_retry_on_cpu(requested_method, exc):
            LOGGER.warning(
                "XGBoost GPU training failed with '%s'. Retrying current task on CPU.",
                exc,
            )
            _disable_gpu_boosters()
            return _retry_training_on_cpu(
                train_config=train_config,
                params=params,
                batch=batch,
                booster_kwargs=booster_kwargs,
                fit_kwargs=fit_kwargs,
            )
        raise


def _harmonize_classifier_fit_kwargs(
    booster: Any,
    batch: TrainingBatch,
    fit_kwargs: Dict[str, Any],
) -> None:
    """Prune unsupported kwargs and attach legacy early stopping when needed."""
    harmonize_fit_kwargs(booster, fit_kwargs, has_eval=batch.evaluation is not None)


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


def _retry_training_on_cpu(
    *,
    train_config: XGBoostTrainConfig,
    params: XGBoostBoosterParams,
    batch: TrainingBatch,
    booster_kwargs: Dict[str, Any],
    fit_kwargs: Dict[str, Any],
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Retry booster fitting on CPU after a GPU-specific failure.

    :returns: Tuple of fitted classifier and optional evaluation history.
    :rtype: Tuple[Any, Dict[str, Any] | None]
    """

    cpu_kwargs = _scrub_gpu_kwargs(booster_kwargs)
    cpu_booster = _instantiate_classifier(
        train_config,
        params,
        "hist",
        cpu_kwargs,
    )
    _ensure_history_metrics(cpu_booster, collect_history=bool(fit_kwargs))
    cpu_booster.fit(batch.train.matrix, batch.train.labels, **fit_kwargs)
    return cpu_booster, _collect_training_history(cpu_booster, bool(fit_kwargs))


def _prepare_eval_artifacts(
    *,
    collect_history: bool,
    context: EvaluationArtifactsContext,
) -> Tuple[Any | None, Any | None]:
    """Prepare evaluation matrices compatible with the fitted encoder."""

    if not collect_history or context.dataset is None:
        return None, None

    eval_docs, eval_label_ids, _ = feature_utils.prepare_prompt_documents(
        context.dataset,
        max_train=0,
        seed=context.train_config.seed,
        extra_fields=context.extra_fields,
    )
    if not eval_docs:
        return None, None

    known_labels = set(context.encoder.classes_)
    filtered_docs: list[str] = []
    filtered_labels: list[str] = []
    for doc, label_id in zip(eval_docs, eval_label_ids):
        if label_id in known_labels:
            filtered_docs.append(doc)
            filtered_labels.append(label_id)
        else:
            LOGGER.debug(
                "Omitting evaluation row with unseen label '%s' during "
                "training history capture.",
                label_id,
            )

    if not filtered_docs:
        return None, None

    eval_matrix = _ensure_float32(context.vectorizer.transform(filtered_docs))
    return eval_matrix, context.encoder.transform(filtered_labels)


def _prepare_training_dataset(
    train_ds,
    train_config: XGBoostTrainConfig,
    extra_fields: Sequence[str] | None,
) -> Tuple[BaseTextVectorizer, LabelEncoder, EncodedDataset]:
    """Vectorise ``train_ds`` and return the encoded dataset alongside the encoder."""
    docs, labels_id = _extract_training_docs(
        train_ds,
        max_train=train_config.max_train,
        seed=train_config.seed,
        extra_fields=extra_fields,
    )
    vectorizer, matrix = _fit_training_vectorizer(docs, train_config)
    # Emit a concise embedding summary for the first training document to aid debugging.
    log_embedding_previews(
        vectorizer, docs, matrix[0], logger=LOGGER, tag="[XGB][Embed]"
    )
    encoder, encoded_labels = _encode_labels(labels_id)
    return vectorizer, encoder, EncodedDataset(matrix=matrix, labels=encoded_labels)


def _extract_training_docs(
    train_ds,
    *,
    max_train: int,
    seed: int,
    extra_fields: Sequence[str] | None,
) -> Tuple[Sequence[str], Sequence[str]]:
    """Extract training documents and aligned label IDs from ``train_ds``."""
    docs, labels_id, _ = feature_utils.prepare_prompt_documents(
        train_ds,
        max_train=max_train,
        seed=seed,
        extra_fields=extra_fields,
    )
    if not docs:
        raise RuntimeError("No training documents were produced for XGBoost fitting.")
    return docs, labels_id


def _fit_training_vectorizer(
    docs: Sequence[str],
    train_config: XGBoostTrainConfig,
) -> Tuple[BaseTextVectorizer, Any]:
    """Initialise and fit the text vectorizer for training documents."""
    vectorizer = create_vectorizer(
        train_config.vectorizer_kind,
        tfidf=train_config.tfidf,
        word2vec=train_config.word2vec,
        sentence_transformer=train_config.sentence_transformer,
    )
    matrix = _ensure_float32(vectorizer.fit_transform(docs))
    return vectorizer, matrix


def _prepare_evaluation_dataset(
    *,
    collect_history: bool,
    context: EvaluationArtifactsContext,
) -> EncodedDataset | None:
    """Return the optional evaluation split when metrics should be collected."""
    eval_matrix, eval_labels = _prepare_eval_artifacts(
        collect_history=collect_history,
        context=context,
    )
    if eval_matrix is None or eval_labels is None:
        return None
    return EncodedDataset(matrix=eval_matrix, labels=eval_labels)


def _build_model_components(
    *,
    train_ds,
    train_config: XGBoostTrainConfig,
    extra_fields: Sequence[str] | None,
    eval_ds=None,
    collect_history: bool = False,
) -> Tuple[BaseTextVectorizer, LabelEncoder, Any, Optional[Dict[str, Any]]]:
    """
    Construct the model components required for training and inference.

    :param train_ds: Dataset split providing training examples.
    :type train_ds: datasets.Dataset | Sequence[dict]
    :param train_config: Configuration bundle controlling sampling and hyper-parameters.
    :type train_config: XGBoostTrainConfig
    :param extra_fields: Additional prompt columns appended during featurisation.
    :type extra_fields: Sequence[str] | None
    :param eval_ds: Optional evaluation dataset monitored during training.
    :type eval_ds: datasets.Dataset | Sequence[dict] | None
    :param collect_history: Record per-round metrics when ``True``.
    :type collect_history: bool
    :returns: Tuple of ``(vectorizer, label_encoder, booster, history)``.
    :rtype: Tuple[BaseTextVectorizer, LabelEncoder, Any, Dict[str, Any] | None]
    :raises RuntimeError: If no documents are produced for training.
    """

    vectorizer, encoder, train_split = _prepare_training_dataset(
        train_ds, train_config, extra_fields
    )
    eval_split = _prepare_evaluation_dataset(
        collect_history=collect_history,
        context=EvaluationArtifactsContext(
            dataset=eval_ds,
            vectorizer=vectorizer,
            encoder=encoder,
            train_config=train_config,
            extra_fields=extra_fields,
        ),
    )

    batch = TrainingBatch(
        train=train_split,
        evaluation=eval_split,
    )
    booster, history = _train_booster(
        train_config=train_config,
        batch=batch,
        collect_history=collect_history,
    )
    return vectorizer, encoder, booster, history


# Prediction helpers moved to model_predict (save/load/predict and selection utilities)


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
