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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from common.pipeline.types import StudySpec, narrow_opinion_selection
from common.opinion.sweep_types import (
    BaseOpinionSweepOutcome,
    BaseOpinionSweepTask,
)

from ...core.vectorizers import (
    SentenceTransformerVectorizerConfig,
    TfidfConfig,
    Word2VecVectorizerConfig,
)
from .sweep import SweepConfig


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class OpinionStageConfig:
    """Inputs required to launch the opinion regression stage."""

    dataset: str
    cache_dir: str
    base_out_dir: Path
    extra_fields: Sequence[str]
    studies: Sequence[str]
    max_participants: int
    seed: int
    max_features: int | None
    tree_method: str
    overwrite: bool
    tfidf_config: TfidfConfig
    word2vec_config: Word2VecVectorizerConfig
    sentence_transformer_config: SentenceTransformerVectorizerConfig
    word2vec_model_base: Path | None
    reuse_existing: bool


@dataclass(frozen=True, init=False)
class OpinionSweepOutcome:
    """
    Compose the common sweep outcome with the XGBoost-specific R² metric.

    Composition keeps instance attributes low while exposing a familiar
    attribute interface via properties that forward to ``base``.
    """

    base: BaseOpinionSweepOutcome[SweepConfig]
    r_squared: float

    def __init__(
        self,
        base: BaseOpinionSweepOutcome[SweepConfig] | None = None,
        r_squared: float | None = None,
        **kwargs,
    ) -> None:
        """Allow construction from either a composed base or flat kwargs."""
        if base is None:
            base = BaseOpinionSweepOutcome(
                order_index=kwargs.pop("order_index"),
                study=kwargs.pop("study"),
                config=kwargs.pop("config"),
                mae=kwargs.pop("mae"),
                rmse=kwargs.pop("rmse"),
                artifact=kwargs.pop("artifact"),
                accuracy_summary=kwargs.pop("accuracy_summary"),
            )
        if r_squared is None:
            r_squared = kwargs.pop("r_squared", None)
        if r_squared is None:
            raise TypeError("OpinionSweepOutcome requires r_squared")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "r_squared", float(r_squared))

    # Compatibility properties mirroring BaseOpinionSweepOutcome
    @property
    def order_index(self) -> int:  # pragma: no cover - simple forwarding
        return self.base.order_index

    @property
    def study(self) -> StudySpec:  # pragma: no cover - simple forwarding
        return self.base.study

    @property
    def config(self) -> SweepConfig:  # pragma: no cover - simple forwarding
        return self.base.config

    @property
    def mae(self) -> float:  # pragma: no cover - simple forwarding
        return self.base.mae

    @property
    def rmse(self) -> float:  # pragma: no cover - simple forwarding
        return self.base.rmse

    @property
    def artifact(self):  # pragma: no cover - simple forwarding
        return self.base.artifact

    @property
    def accuracy_summary(self):  # pragma: no cover - simple forwarding
        return self.base.accuracy_summary

    @property
    def metrics_path(self):  # pragma: no cover - simple forwarding
        return self.base.metrics_path

    @property
    def metrics(self):  # pragma: no cover - simple forwarding
        return self.base.metrics

    @property
    def accuracy(self):  # pragma: no cover - simple forwarding
        return self.base.accuracy

    @property
    def baseline_accuracy(self):  # pragma: no cover - simple forwarding
        return self.base.baseline_accuracy

    @property
    def accuracy_delta(self):  # pragma: no cover - simple forwarding
        return self.base.accuracy_delta

    @property
    def eligible(self):  # pragma: no cover - simple forwarding
        return self.base.eligible


@dataclass(frozen=True)
class OpinionSweepTask(BaseOpinionSweepTask[SweepConfig]):
    """
    Extend :class:`common.opinion.sweep_types.BaseOpinionSweepTask` with the
    keyword arguments required by the XGBoost implementation.
    """

    request_args: Mapping[str, object]
    feature_space: str = "tfidf"


OpinionStudySelection = narrow_opinion_selection(OpinionSweepOutcome)


@dataclass(frozen=True)
class OpinionDataSettings:
    """Dataset and sampling settings for opinion sweeps."""

    dataset: str
    cache_dir: str
    extra_fields: Sequence[str]
    max_participants: int
    seed: int
    max_features: int | None


@dataclass(frozen=True)
class OpinionVectorizerSettings:
    """Vectoriser configurations used across opinion sweep tasks."""

    tfidf_config: TfidfConfig
    word2vec_config: Word2VecVectorizerConfig
    sentence_transformer_config: SentenceTransformerVectorizerConfig
    word2vec_model_base: Path | None


@dataclass(frozen=True)
class OpinionXgbSettings:
    """XGBoost-related settings shared across sweeps."""

    tree_method: str
    overwrite: bool


@dataclass(frozen=True, init=False)
class OpinionSweepRunContext:
    """Configuration shared across opinion sweep evaluations (grouped)."""

    sweep_dir: Path
    data: OpinionDataSettings
    vectorizers: OpinionVectorizerSettings
    xgb: OpinionXgbSettings

    # Backwards-compat initialiser supporting legacy flat kwargs used in tests
    def __init__(  # pylint: disable=too-many-locals
        self,
        *,
        sweep_dir: Path,
        data: OpinionDataSettings | None = None,
        vectorizers: OpinionVectorizerSettings | None = None,
        xgb: OpinionXgbSettings | None = None,
        # Legacy flat kwargs
        dataset: str | None = None,
        cache_dir: str | None = None,
        extra_fields: Sequence[str] | None = None,
        max_participants: int | None = None,
        seed: int | None = None,
        max_features: int | None = None,
        tree_method: str | None = None,
        overwrite: bool | None = None,
        tfidf_config: TfidfConfig | None = None,
        word2vec_config: Word2VecVectorizerConfig | None = None,
        sentence_transformer_config: SentenceTransformerVectorizerConfig | None = None,
        word2vec_model_base: Path | None = None,
    ) -> None:
        if data is None:
            data = OpinionDataSettings(
                dataset=dataset or "",
                cache_dir=cache_dir or "",
                extra_fields=tuple(extra_fields or ()),
                max_participants=int(max_participants or 0),
                seed=int(seed or 0),
                max_features=max_features,
            )
        if vectorizers is None:
            vectorizers = OpinionVectorizerSettings(
                tfidf_config=tfidf_config or TfidfConfig(),
                word2vec_config=word2vec_config or Word2VecVectorizerConfig(),
                sentence_transformer_config=(
                    sentence_transformer_config or SentenceTransformerVectorizerConfig()
                ),
                word2vec_model_base=word2vec_model_base,
            )
        if xgb is None:
            xgb = OpinionXgbSettings(
                tree_method=tree_method or "hist",
                overwrite=bool(overwrite) if overwrite is not None else False,
            )
        object.__setattr__(self, "sweep_dir", sweep_dir)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "vectorizers", vectorizers)
        object.__setattr__(self, "xgb", xgb)

    # Backwards-compatible attribute accessors
    @property
    def dataset(self) -> str:  # pragma: no cover - simple forwarding
        return self.data.dataset

    @property
    def cache_dir(self) -> str:  # pragma: no cover - simple forwarding
        return self.data.cache_dir

    @property
    def extra_fields(self) -> Sequence[str]:  # pragma: no cover - simple forwarding
        return self.data.extra_fields

    @property
    def max_participants(self) -> int:  # pragma: no cover - simple forwarding
        return self.data.max_participants

    @property
    def seed(self) -> int:  # pragma: no cover - simple forwarding
        return self.data.seed

    @property
    def max_features(self) -> int | None:  # pragma: no cover - simple forwarding
        return self.data.max_features

    @property
    def tfidf_config(self) -> TfidfConfig:  # pragma: no cover - simple forwarding
        return self.vectorizers.tfidf_config

    @property
    def word2vec_config(self) -> Word2VecVectorizerConfig:  # pragma: no cover
        return self.vectorizers.word2vec_config

    @property
    def sentence_transformer_config(
        self,
    ) -> SentenceTransformerVectorizerConfig:  # pragma: no cover
        return self.vectorizers.sentence_transformer_config

    @property
    def word2vec_model_base(self) -> Path | None:  # pragma: no cover
        return self.vectorizers.word2vec_model_base

    @property
    def tree_method(self) -> str:  # pragma: no cover - simple forwarding
        return self.xgb.tree_method

    @property
    def overwrite(self) -> bool:  # pragma: no cover - simple forwarding
        return self.xgb.overwrite
