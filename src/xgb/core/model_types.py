#!/usr/bin/env python
"""Shared dataclasses and typing helpers for the XGBoost baseline.

This module holds lightweight containers used by both training and
inference code to avoid circular imports and keep modules focused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, TYPE_CHECKING

from ._optional import LabelEncoder
from .vectorizers import BaseTextVectorizer

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .model_config import XGBoostTrainConfig  # noqa: F401


@dataclass(frozen=True)
class EncodedDataset:
    """Pair of encoded documents and their aligned labels."""

    matrix: Any
    labels: Any


@dataclass(frozen=True)
class TrainingBatch:
    """Container for training data and optional evaluation split."""

    train: EncodedDataset
    evaluation: EncodedDataset | None = None


@dataclass(frozen=True)
class EvaluationArtifactsContext:
    """Information required to prepare evaluation matrices."""

    dataset: Any | None
    vectorizer: BaseTextVectorizer
    encoder: LabelEncoder
    train_config: "XGBoostTrainConfig"
    extra_fields: Sequence[str] | None = None


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
    :ivar training_history: Optional history captured during fitting (per-round metrics).
    :vartype training_history: Dict[str, Any] | None
    """

    vectorizer: BaseTextVectorizer
    label_encoder: LabelEncoder
    booster: Any
    extra_fields: Tuple[str, ...]
    training_history: Optional[dict[str, Any]] = None


__all__ = [
    "EncodedDataset",
    "TrainingBatch",
    "EvaluationArtifactsContext",
    "XGBoostSlateModel",
]

