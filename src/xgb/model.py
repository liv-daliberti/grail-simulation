"""XGBoost training and inference utilities for slate prediction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None  # type: ignore[assignment]

import numpy as np

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

try:  # pragma: no cover - optional dependency
    from sklearn.preprocessing import LabelEncoder
except ImportError:  # pragma: no cover - optional dependency
    LabelEncoder = None  # type: ignore[assignment]


def _ensure_label_encoder_available(action: str) -> None:
    """Ensure optional scikit-learn dependency is present before continuing."""
    if LabelEncoder is None:  # pragma: no cover - optional dependency
        raise ImportError(f"Install scikit-learn to {action}.")

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore

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
    :ivar vectorizer_kind: Feature extraction strategy (``tfidf``, ``word2vec``, or ``sentence_transformer``).
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
    capped = train_config.max_features if train_config.max_features and train_config.max_features > 0 else None
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
    """Return an encoder and numeric labels derived from ``labels_id``."""
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
    """Instantiate and fit the XGBoost booster component."""
    params = train_config.booster
    booster_kwargs = dict(params.extra_kwargs or {})
    unique_labels = int(np.unique(labels).size)
    booster_kwargs.setdefault("num_class", unique_labels)
    booster = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        tree_method=params.tree_method,
        reg_lambda=params.reg_lambda,
        reg_alpha=params.reg_alpha,
        random_state=train_config.seed,
        nthread=-1,
        **booster_kwargs,
    )
    booster.fit(matrix, labels)
    return booster


def _build_model_components(
    *,
    train_ds,
    train_config: XGBoostTrainConfig,
    extra_fields: Sequence[str] | None,
) -> Tuple[BaseTextVectorizer, LabelEncoder, Any]:
    """Return fitted vectoriser, label encoder, and trained booster."""

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
    """Return the highest scoring slate index, applying fallback heuristics."""
    primary = _best_index_by_key(slate_pairs, probability_map, _candidate_id_key)
    if primary is not None:
        return primary
    return _best_index_by_key(slate_pairs, probability_map, _fallback_candidate_key)


def _best_index_by_key(
    slate_pairs: Sequence[tuple[str, str]],
    probability_map: Dict[str, float],
    key_fn: Callable[[str, str], str],
) -> Optional[int]:
    """Return the best slate index according to ``key_fn`` lookup."""
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
    """Return the canonical id associated with ``video_id`` (falling back to ``title``)."""
    return canon_video_id(video_id) or canon_video_id(title)


def _fallback_candidate_key(title: str, video_id: str) -> str:
    """Return a fallback key using canonical id or derived title."""
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
    "fit_xgboost_model",
    "load_xgboost_model",
    "predict_among_slate",
    "save_xgboost_model",
]
