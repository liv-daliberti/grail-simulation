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

"""
Opinion-stage context and outcome types for the XGBoost pipeline.

This module defines small, focused data classes used to configure and
execute the opinion-regression stage backed by XGBoost, and a thin wrapper
around the common opinion sweep outcome that adds the XGBoost-specific
``r_squared`` metric. It also groups related settings (data, vectorisers, and
XGBoost parameters) into cohesive structures and provides a backwards-
compatible constructor for legacy call sites.

Examples
--------
- Constructing an outcome from a composed base outcome::

    from common.opinion.sweep_types import BaseOpinionSweepOutcome, MetricsArtifact, AccuracySummary
    from xgb.pipeline_context import SweepConfig

    base = BaseOpinionSweepOutcome[SweepConfig](
        order_index=0,
        study=...,            # a StudySpec
        config=...,           # a SweepConfig
        mae=0.12,
        rmse=0.21,
        artifact=MetricsArtifact(path=..., payload={}),
        accuracy_summary=AccuracySummary(value=0.61, baseline=0.55, delta=0.06, eligible=1200),
    )
    outcome = OpinionSweepOutcome(base=base, r_squared=0.78)

- Constructing an outcome from flat keyword arguments (legacy)::

    outcome = OpinionSweepOutcome(
        order_index=0,
        study=..., config=..., mae=0.12, rmse=0.21,
        artifact=MetricsArtifact(path=..., payload={}),
        accuracy_summary=AccuracySummary(...),
        r_squared=0.78,
    )

- Grouping run settings for a sweep::

    ctx = OpinionSweepRunContext(
        sweep_dir=Path("runs/opinion/xgb"),
        dataset="my_dataset",
        cache_dir="~/.cache/opinion",
        extra_fields=("title", "category"),
        max_participants=1000,
        seed=7,
        max_features=50000,
        tree_method="hist",
        overwrite=False,
        tfidf_config=TfidfConfig(),
        word2vec_config=Word2VecVectorizerConfig(),
        sentence_transformer_config=SentenceTransformerVectorizerConfig(),
        word2vec_model_base=None,
    )

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from common.pipeline.types import narrow_opinion_selection
from common.opinion.sweep_types import (
    BaseOpinionSweepOutcome,
    BaseOpinionSweepTask,
)

from ...core.vectorizers import (
    SentenceTransformerVectorizerConfig,
    TfidfConfig,
    Word2VecVectorizerConfig,
)


@dataclass(frozen=True, init=False)
class OpinionStageConfig:
    """Inputs required to launch the opinion regression stage.

    Group settings to keep instance attributes small, and provide
    backward-compatible attribute accessors for legacy call sites.
    """

    __module__ = "xgb.pipeline.context"

    base_out_dir: Path
    studies: Sequence[str]
    reuse_existing: bool
    data: OpinionDataSettings
    vectorizers: OpinionVectorizerSettings
    xgb: OpinionXgbSettings

    def __init__(
        self,
        *,
        base_out_dir: Path,
        studies: Sequence[str],
        reuse_existing: bool,
        data: OpinionDataSettings | None = None,
        vectorizers: OpinionVectorizerSettings | None = None,
        xgb: OpinionXgbSettings | None = None,
        **legacy_kwargs: object,
    ) -> None:
        if data is None:
            data = OpinionDataSettings(
                dataset=str(legacy_kwargs.get("dataset", "")),
                cache_dir=str(legacy_kwargs.get("cache_dir", "")),
                extra_fields=tuple(legacy_kwargs.get("extra_fields", ()) or ()),
                max_participants=int(legacy_kwargs.get("max_participants", 0) or 0),
                seed=int(legacy_kwargs.get("seed", 0) or 0),
                max_features=legacy_kwargs.get("max_features"),  # type: ignore[arg-type]
            )
        if vectorizers is None:
            vectorizers = OpinionVectorizerSettings(
                tfidf_config=(
                    legacy_kwargs.get("tfidf_config") or TfidfConfig()
                ),  # type: ignore[arg-type]
                word2vec_config=(
                    legacy_kwargs.get("word2vec_config")
                    or Word2VecVectorizerConfig()
                ),  # type: ignore[arg-type]
                sentence_transformer_config=(
                    legacy_kwargs.get("sentence_transformer_config")
                    or SentenceTransformerVectorizerConfig()
                ),  # type: ignore[arg-type]
                word2vec_model_base=legacy_kwargs.get(
                    "word2vec_model_base"
                ),  # type: ignore[arg-type]
            )
        if xgb is None:
            xgb = OpinionXgbSettings(
                tree_method=str(legacy_kwargs.get("tree_method", "hist") or "hist"),
                overwrite=(
                    bool(legacy_kwargs.get("overwrite"))
                    if legacy_kwargs.get("overwrite") is not None
                    else False
                ),
            )
        object.__setattr__(self, "base_out_dir", base_out_dir)
        object.__setattr__(self, "studies", tuple(studies or ()))
        object.__setattr__(self, "reuse_existing", bool(reuse_existing))
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "vectorizers", vectorizers)
        object.__setattr__(self, "xgb", xgb)

    # Legacy attribute compatibility
    @property
    def dataset(self) -> str:  # pragma: no cover - simple forwarding
        """Dataset identifier used for the stage.

        :returns: Dataset name or path.
        :rtype: str
        """
        return self.data.dataset

    @property
    def cache_dir(self) -> str:  # pragma: no cover - simple forwarding
        """Cache directory used when loading the dataset.

        :returns: Filesystem path to the cache directory.
        :rtype: str
        """
        return self.data.cache_dir

    @property
    def extra_fields(self) -> Sequence[str]:  # pragma: no cover - simple forwarding
        """Additional prompt/document fields included during evaluation.

        :returns: Ordered field names requested from the dataset.
        :rtype: Sequence[str]
        """
        return self.data.extra_fields

    @property
    def max_participants(self) -> int:  # pragma: no cover - simple forwarding
        """Maximum number of participants to include per study.

        :returns: Upper bound on participants sampled.
        :rtype: int
        """
        return self.data.max_participants

    @property
    def seed(self) -> int:  # pragma: no cover - simple forwarding
        """Random seed used for reproducible sampling.

        :returns: Seed value controlling random operations.
        :rtype: int
        """
        return self.data.seed

    @property
    def max_features(self) -> int | None:  # pragma: no cover - simple forwarding
        """Maximum number of TF–IDF features to include.

        :returns: Feature cap, or ``None`` to include all.
        :rtype: Optional[int]
        """
        return self.data.max_features

    @property
    def tfidf_config(self) -> TfidfConfig:  # pragma: no cover - simple forwarding
        """TF–IDF vectoriser configuration shared across tasks.

        :returns: TF–IDF configuration object.
        :rtype: TfidfConfig
        """
        return self.vectorizers.tfidf_config

    @property
    def word2vec_config(self) -> Word2VecVectorizerConfig:  # pragma: no cover
        """Word2Vec vectoriser configuration shared across tasks.

        :returns: Word2Vec configuration object.
        :rtype: Word2VecVectorizerConfig
        """
        return self.vectorizers.word2vec_config

    @property
    def sentence_transformer_config(
        self,
    ) -> SentenceTransformerVectorizerConfig:  # pragma: no cover
        """Sentence-transformer vectoriser configuration shared across tasks.

        :returns: Sentence-transformer configuration object.
        :rtype: SentenceTransformerVectorizerConfig
        """
        return self.vectorizers.sentence_transformer_config

    @property
    def word2vec_model_base(self) -> Path | None:  # pragma: no cover
        """Base directory for Word2Vec model artefacts, if provided.

        :returns: Path to the model base or ``None`` if unset.
        :rtype: Optional[~pathlib.Path]
        """
        return self.vectorizers.word2vec_model_base

    @property
    def tree_method(self) -> str:  # pragma: no cover - simple forwarding
        """XGBoost tree construction algorithm name.

        :returns: Tree method identifier (e.g. ``hist``).
        :rtype: str
        """
        return self.xgb.tree_method

    @property
    def overwrite(self) -> bool:  # pragma: no cover - simple forwarding
        """Whether to overwrite existing artefacts when present.

        :returns: ``True`` to overwrite, ``False`` to reuse.
        :rtype: bool
        """
        return self.xgb.overwrite


@dataclass(frozen=True, init=False)
class OpinionSweepOutcome:
    """
    Compose the common sweep outcome with the XGBoost-specific R² metric.

    Composition keeps instance attributes low while exposing a familiar
    attribute interface via properties that forward to ``base``.
    """

    __module__ = "xgb.pipeline.context"

    base: "common.opinion.sweep_types.BaseOpinionSweepOutcome[xgb.pipeline.context.SweepConfig]"
    r_squared: float

    def __init__(
        self,
        base: BaseOpinionSweepOutcome["xgb.pipeline.context.SweepConfig"] | None = None,
        r_squared: float | None = None,
        **kwargs,
    ) -> None:
        """
        Allow construction from either a composed base or flat kwargs.

        Flat kwargs expected: ``order_index``, ``study``, ``config``, ``mae``, ``rmse``,
        ``artifact``, ``accuracy_summary``, ``r_squared``.

        :param base: Composed base opinion sweep outcome holding common fields.
        :type base: BaseOpinionSweepOutcome[~xgb.pipeline.context.SweepConfig], optional
        :param r_squared: Coefficient of determination for the regression.
        :type r_squared: float, optional
        :param kwargs: Flat construction arguments used when ``base`` is not supplied.
        :type kwargs: dict
        :returns: None.
        :rtype: None
        """
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
        """Return the deterministic ordering index for this outcome.

        :returns: Zero-based ordinal used to preserve submission order.
        :rtype: int
        """
        return self.base.order_index

    @property
    def study(self) -> "common.pipeline.types.StudySpec":  # pragma: no cover - simple forwarding
        """Study metadata associated with the evaluated configuration.

        :returns: The :class:`~common.pipeline.types.StudySpec` for this outcome.
        :rtype: ~common.pipeline.types.StudySpec
        """
        return self.base.study

    @property
    def config(self) -> "xgb.pipeline.context.SweepConfig":  # pragma: no cover - simple forwarding
        """Pipeline hyper-parameter configuration evaluated by the sweep.

        :returns: The configuration object used to produce this outcome.
        :rtype: ~xgb.pipeline.context.SweepConfig
        """
        return self.base.config

    @property
    def mae(self) -> float:  # pragma: no cover - simple forwarding
        """Mean absolute error achieved by the configuration.

        :returns: MAE metric aggregated over the evaluation split.
        :rtype: float
        """
        return self.base.mae

    @property
    def rmse(self) -> float:  # pragma: no cover - simple forwarding
        """Root mean squared error achieved by the configuration.

        :returns: RMSE metric aggregated over the evaluation split.
        :rtype: float
        """
        return self.base.rmse

    @property
    def artifact(self):  # pragma: no cover - simple forwarding
        """Metrics artefact bundling the payload and its storage path.

        :returns: :class:`~common.opinion.sweep_types.MetricsArtifact` instance.
        :rtype: ~common.opinion.sweep_types.MetricsArtifact
        """
        return self.base.artifact

    @property
    def accuracy_summary(self):  # pragma: no cover - simple forwarding
        """Directional accuracy metrics summary for this outcome.

        :returns: Summary containing accuracy, baseline, delta, and eligibility.
        :rtype: ~common.opinion.sweep_types.AccuracySummary
        """
        return self.base.accuracy_summary

    @property
    def metrics_path(self):  # pragma: no cover - simple forwarding
        """Filesystem path to the persisted metrics artefact.

        :returns: Path where the metrics JSON is stored on disk.
        :rtype: ~pathlib.Path
        """
        return self.base.metrics_path

    @property
    def metrics(self):  # pragma: no cover - simple forwarding
        """Raw metrics payload loaded from the artefact.

        :returns: Deserialised metrics mapping.
        :rtype: Mapping[str, object]
        """
        return self.base.metrics

    @property
    def accuracy(self):  # pragma: no cover - simple forwarding
        """Directional accuracy achieved by the configuration.

        :returns: Accuracy value if available, otherwise ``None``.
        :rtype: Optional[float]
        """
        return self.base.accuracy

    @property
    def baseline_accuracy(self):  # pragma: no cover - simple forwarding
        """Directional accuracy achieved by the baseline configuration.

        :returns: Baseline accuracy value if available, otherwise ``None``.
        :rtype: Optional[float]
        """
        return self.base.baseline_accuracy

    @property
    def accuracy_delta(self):  # pragma: no cover - simple forwarding
        """Improvement in accuracy over the baseline configuration.

        :returns: Difference between accuracy and baseline, if available.
        :rtype: Optional[float]
        """
        return self.base.accuracy_delta

    @property
    def eligible(self):  # pragma: no cover - simple forwarding
        """Number of examples contributing to accuracy metrics.

        :returns: Eligible evaluation count if available, otherwise ``None``.
        :rtype: Optional[int]
        """
        return self.base.eligible


@dataclass(frozen=True)
class OpinionSweepTask(BaseOpinionSweepTask["xgb.pipeline.context.SweepConfig"]):
    """
    Extend :class:`common.opinion.sweep_types.BaseOpinionSweepTask` with the
    keyword arguments required by the XGBoost implementation.

    :param request_args: Keyword arguments passed to the opinion evaluation runner.
    :type request_args: Mapping[str, object]
    :param feature_space: Feature space evaluated by the sweep task.
    :type feature_space: str
    """

    __module__ = "xgb.pipeline.context"

    request_args: Mapping[str, object]
    feature_space: str = "tfidf"


OpinionStudySelection = narrow_opinion_selection(OpinionSweepOutcome)


@dataclass(frozen=True)
class OpinionDataSettings:
    """Dataset and sampling settings for opinion sweeps."""

    __module__ = "xgb.pipeline.context"

    dataset: str
    cache_dir: str
    extra_fields: Sequence[str]
    max_participants: int
    seed: int
    max_features: int | None


@dataclass(frozen=True)
class OpinionVectorizerSettings:
    """Vectoriser configurations used across opinion sweep tasks."""

    __module__ = "xgb.pipeline.context"

    tfidf_config: TfidfConfig
    word2vec_config: Word2VecVectorizerConfig
    sentence_transformer_config: SentenceTransformerVectorizerConfig
    word2vec_model_base: Path | None


@dataclass(frozen=True)
class OpinionXgbSettings:
    """XGBoost-related settings shared across sweeps."""

    __module__ = "xgb.pipeline.context"

    tree_method: str
    overwrite: bool


@dataclass(frozen=True, init=False)
class OpinionSweepRunContext:
    """Configuration shared across opinion sweep evaluations (grouped)."""

    __module__ = "xgb.pipeline.context"

    sweep_dir: Path
    data: OpinionDataSettings
    vectorizers: OpinionVectorizerSettings
    xgb: OpinionXgbSettings

    # Backwards-compat initialiser supporting legacy flat kwargs used in tests
    def __init__(
        self,
        *,
        sweep_dir: Path,
        data: OpinionDataSettings | None = None,
        vectorizers: OpinionVectorizerSettings | None = None,
        xgb: OpinionXgbSettings | None = None,
        **legacy_kwargs: object,
    ) -> None:
        """
        Construct a grouped run context, also accepting legacy flat kwargs.

        Preferred usage passes grouped settings via ``data``, ``vectorizers``, and ``xgb``.
        For backwards compatibility, legacy keyword arguments are also accepted through
        ``**legacy_kwargs``. Supported legacy keys are:
        ``dataset``, ``cache_dir``, ``extra_fields``, ``max_participants``, ``seed``,
        ``max_features``, ``tree_method``, ``overwrite``, ``tfidf_config``,
        ``word2vec_config``, ``sentence_transformer_config``, and ``word2vec_model_base``.

        :param sweep_dir: Root directory where sweep artefacts are written.
        :type sweep_dir: Path
        :param data: Pre-grouped dataset and sampling settings.
        :type data: OpinionDataSettings, optional
        :param vectorizers: Pre-grouped vectoriser configurations.
        :type vectorizers: OpinionVectorizerSettings, optional
        :param xgb: Pre-grouped XGBoost shared settings for sweeps.
        :type xgb: OpinionXgbSettings, optional
        :param legacy_kwargs: Legacy flat kwargs listed above.
        :type legacy_kwargs: Mapping[str, object]
        :returns: None.
        :rtype: None
        """
        if data is None:
            data = OpinionDataSettings(
                dataset=str(legacy_kwargs.get("dataset", "")),
                cache_dir=str(legacy_kwargs.get("cache_dir", "")),
                extra_fields=tuple(legacy_kwargs.get("extra_fields", ()) or ()),
                max_participants=int(legacy_kwargs.get("max_participants", 0) or 0),
                seed=int(legacy_kwargs.get("seed", 0) or 0),
                max_features=legacy_kwargs.get("max_features"),  # type: ignore[arg-type]
            )
        if vectorizers is None:
            vectorizers = OpinionVectorizerSettings(
                tfidf_config=(
                    legacy_kwargs.get("tfidf_config") or TfidfConfig()
                ),  # type: ignore[arg-type]
                word2vec_config=(
                    legacy_kwargs.get("word2vec_config") or Word2VecVectorizerConfig()
                ),  # type: ignore[arg-type]
                sentence_transformer_config=(
                    legacy_kwargs.get("sentence_transformer_config")
                    or SentenceTransformerVectorizerConfig()
                ),  # type: ignore[arg-type]
                word2vec_model_base=legacy_kwargs.get(
                    "word2vec_model_base"
                ),  # type: ignore[arg-type]
            )
        if xgb is None:
            xgb = OpinionXgbSettings(
                tree_method=str(legacy_kwargs.get("tree_method", "hist") or "hist"),
                overwrite=(
                    bool(legacy_kwargs.get("overwrite"))
                    if legacy_kwargs.get("overwrite") is not None
                    else False
                ),
            )
        object.__setattr__(self, "sweep_dir", sweep_dir)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "vectorizers", vectorizers)
        object.__setattr__(self, "xgb", xgb)

    # Backwards-compatible attribute accessors
    @property
    def dataset(self) -> str:  # pragma: no cover - simple forwarding
        """Dataset identifier used for the sweep."""
        return self.data.dataset

    @property
    def cache_dir(self) -> str:  # pragma: no cover - simple forwarding
        """Cache directory used when loading the dataset."""
        return self.data.cache_dir

    @property
    def extra_fields(self) -> Sequence[str]:  # pragma: no cover - simple forwarding
        """Additional prompt/document fields included during evaluation."""
        return self.data.extra_fields

    @property
    def max_participants(self) -> int:  # pragma: no cover - simple forwarding
        """Maximum number of participants to include per study."""
        return self.data.max_participants

    @property
    def seed(self) -> int:  # pragma: no cover - simple forwarding
        """Random seed used for reproducible sampling."""
        return self.data.seed

    @property
    def max_features(self) -> int | None:  # pragma: no cover - simple forwarding
        """Maximum number of TF–IDF features, or ``None`` for all."""
        return self.data.max_features

    @property
    def tfidf_config(self) -> TfidfConfig:  # pragma: no cover - simple forwarding
        """TF–IDF vectoriser configuration shared across sweep tasks."""
        return self.vectorizers.tfidf_config

    @property
    def word2vec_config(self) -> Word2VecVectorizerConfig:  # pragma: no cover
        """Word2Vec vectoriser configuration shared across sweep tasks."""
        return self.vectorizers.word2vec_config

    @property
    def sentence_transformer_config(
        self,
    ) -> SentenceTransformerVectorizerConfig:  # pragma: no cover
        """Sentence-transformer configuration shared across sweep tasks."""
        return self.vectorizers.sentence_transformer_config

    @property
    def word2vec_model_base(self) -> Path | None:  # pragma: no cover
        """Base directory for Word2Vec model artefacts, if provided."""
        return self.vectorizers.word2vec_model_base

    @property
    def tree_method(self) -> str:  # pragma: no cover - simple forwarding
        """XGBoost tree construction algorithm name (e.g. ``hist``)."""
        return self.xgb.tree_method

    @property
    def overwrite(self) -> bool:  # pragma: no cover - simple forwarding
        """Whether to overwrite existing artefacts when present."""
        return self.xgb.overwrite
