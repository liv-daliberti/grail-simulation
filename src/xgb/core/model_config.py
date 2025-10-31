#!/usr/bin/env python
"""Configuration dataclasses for the XGBoost baseline.

This module groups booster and training configuration structures to keep
``model.py`` focused on the training pipeline logic and reduce file size.
Where possible, constructors accept grouped configs and ``**kwargs`` for
backward compatibility with older flat argument patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .vectorizers import (
    SentenceTransformerVectorizerConfig,
    TfidfConfig,
    Word2VecVectorizerConfig,
)


@dataclass(frozen=True)
class _BoosterCore:
    learning_rate: float = 0.1
    max_depth: int = 6
    n_estimators: int = 300
    tree_method: str = "hist"


@dataclass(frozen=True)
class _BoosterSampling:
    subsample: float = 0.8
    colsample_bytree: float = 0.8


@dataclass(frozen=True)
class _BoosterRegularization:
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0


@dataclass(init=False)
class XGBoostBoosterParams:
    """XGBoost hyper-parameters grouped into sub-configs.

    Backwards-compatible usage via flat keyword arguments is supported using
    ``**kwargs``. Prefer constructing grouped sub-configs for clarity.
    """

    core: _BoosterCore = field(default_factory=_BoosterCore)
    sampling: _BoosterSampling = field(default_factory=_BoosterSampling)
    regularization: _BoosterRegularization = field(default_factory=_BoosterRegularization)
    extra_kwargs: Dict[str, Any] | None = None

    def __init__(
        self,
        *,
        extra_kwargs: Dict[str, Any] | None = None,
        core: _BoosterCore | None = None,
        sampling: _BoosterSampling | None = None,
        regularization: _BoosterRegularization | None = None,
        **flat_kwargs: Any,
    ) -> None:
        # Merge flat overrides into grouped configs when provided.
        core = core or _BoosterCore(
            learning_rate=flat_kwargs.pop("learning_rate", 0.1),
            max_depth=flat_kwargs.pop("max_depth", 6),
            n_estimators=flat_kwargs.pop("n_estimators", 300),
            tree_method=flat_kwargs.pop("tree_method", "hist"),
        )
        sampling = sampling or _BoosterSampling(
            subsample=flat_kwargs.pop("subsample", 0.8),
            colsample_bytree=flat_kwargs.pop("colsample_bytree", 0.8),
        )
        regularization = regularization or _BoosterRegularization(
            reg_lambda=flat_kwargs.pop("reg_lambda", 1.0),
            reg_alpha=flat_kwargs.pop("reg_alpha", 0.0),
        )
        self.core = core
        self.sampling = sampling
        self.regularization = regularization
        # Remaining ``flat_kwargs`` are forwarded to the estimator unchanged.
        self.extra_kwargs = {**(extra_kwargs or {}), **flat_kwargs} if flat_kwargs else extra_kwargs

    # Compatibility accessors
    @property
    def learning_rate(self) -> float:  # pragma: no cover - simple forwarding
        return self.core.learning_rate

    @property
    def max_depth(self) -> int:  # pragma: no cover
        return self.core.max_depth

    @property
    def n_estimators(self) -> int:  # pragma: no cover
        return self.core.n_estimators

    @property
    def tree_method(self) -> str:  # pragma: no cover
        return self.core.tree_method

    @property
    def subsample(self) -> float:  # pragma: no cover
        return self.sampling.subsample

    @property
    def colsample_bytree(self) -> float:  # pragma: no cover
        return self.sampling.colsample_bytree

    @property
    def reg_lambda(self) -> float:  # pragma: no cover
        return self.regularization.reg_lambda

    @property
    def reg_alpha(self) -> float:  # pragma: no cover
        return self.regularization.reg_alpha

    @classmethod
    def create(cls, **kwargs: Any) -> "XGBoostBoosterParams":
        """Convenience constructor using flat keyword arguments.

        Accepts any of the flat parameters like ``learning_rate`` or grouped
        overrides via ``core=``, ``sampling=``, and ``regularization=``. Any
        unknown items are placed into :attr:`extra_kwargs`.
        """
        return cls(**kwargs)


@dataclass
class _TrainVectorizers:
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    word2vec: Word2VecVectorizerConfig = field(default_factory=Word2VecVectorizerConfig)
    sentence_transformer: SentenceTransformerVectorizerConfig = field(
        default_factory=SentenceTransformerVectorizerConfig
    )


@dataclass(init=False)
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
    vectorizers: _TrainVectorizers = field(default_factory=_TrainVectorizers)
    booster: XGBoostBoosterParams = field(default_factory=XGBoostBoosterParams)

    def __init__(
        self,
        *,
        max_train: int = 200_000,
        seed: int = 42,
        max_features: Optional[int] = 200_000,
        vectorizer_kind: str = "tfidf",
        booster: XGBoostBoosterParams | None = None,
        vectorizers: _TrainVectorizers | None = None,
        tfidf: TfidfConfig | None = None,
        word2vec: Word2VecVectorizerConfig | None = None,
        sentence_transformer: SentenceTransformerVectorizerConfig | None = None,
        **_: Any,
    ) -> None:
        self.max_train = max_train
        self.seed = seed
        self.max_features = max_features
        self.vectorizer_kind = vectorizer_kind
        self.vectorizers = vectorizers or _TrainVectorizers(
            tfidf=tfidf or TfidfConfig(),
            word2vec=word2vec or Word2VecVectorizerConfig(),
            sentence_transformer=sentence_transformer or SentenceTransformerVectorizerConfig(),
        )
        self.booster = booster or XGBoostBoosterParams()

    # Backwards-compatible accessors for vectorizer configs
    @property
    def tfidf(self) -> TfidfConfig:  # pragma: no cover - simple forwarding
        return self.vectorizers.tfidf

    @tfidf.setter
    def tfidf(self, value: TfidfConfig) -> None:  # pragma: no cover - simple forwarding
        self.vectorizers.tfidf = value

    @property
    def word2vec(self) -> Word2VecVectorizerConfig:  # pragma: no cover
        return self.vectorizers.word2vec

    @word2vec.setter
    def word2vec(self, value: Word2VecVectorizerConfig) -> None:  # pragma: no cover
        self.vectorizers.word2vec = value

    @property
    def sentence_transformer(self) -> SentenceTransformerVectorizerConfig:  # pragma: no cover
        return self.vectorizers.sentence_transformer

    @sentence_transformer.setter
    def sentence_transformer(
        self, value: SentenceTransformerVectorizerConfig
    ) -> None:  # pragma: no cover
        self.vectorizers.sentence_transformer = value

    @classmethod
    def create(cls, **kwargs: Any) -> "XGBoostTrainConfig":
        """Convenience constructor compatible with historic flat args.

        Accepts vectoriser overrides and booster hyper-parameters via ``kwargs``.
        """
        return cls(**kwargs)


__all__ = [
    "XGBoostBoosterParams",
    "XGBoostTrainConfig",
]

