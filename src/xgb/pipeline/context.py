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

"""Data structures describing the Grail Simulation XGBoost pipeline.

Defines sweep configuration objects, execution contexts, and selection
results exchanged between the sweep, evaluation, and reporting stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from common.opinion import OpinionCalibrationMetrics
from common.pipeline.types import (
    BasePipelineSweepOutcome,
    StudySelection as BaseStudySelection,
    StudySpec,
    narrow_opinion_selection,
)
from common.opinion.sweep_types import (
    BaseOpinionSweepOutcome,
    BaseOpinionSweepTask,
    BaseSweepTask,
)

from ..core.model import XGBoostBoosterParams
from ..core.vectorizers import (
    SentenceTransformerVectorizerConfig,
    TfidfConfig,
    Word2VecVectorizerConfig,
)


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class SweepConfig:
    """
    Hyper-parameter configuration evaluated during XGBoost sweeps.

    :param text_vectorizer: Vectoriser identifier (e.g. ``tfidf``).
    :type text_vectorizer: str
    :param vectorizer_tag: Short tag used in directory and report labels.
    :type vectorizer_tag: str
    :param learning_rate: Booster learning rate.
    :type learning_rate: float
    :param max_depth: Tree depth explored during sweeps.
    :type max_depth: int
    :param n_estimators: Number of boosting rounds.
    :type n_estimators: int
    :param subsample: Row subsampling ratio.
    :type subsample: float
    :param colsample_bytree: Column subsampling ratio per tree.
    :type colsample_bytree: float
    :param reg_lambda: L2 regularisation weight.
    :type reg_lambda: float
    :param reg_alpha: L1 regularisation weight.
    :type reg_alpha: float
    :param vectorizer_cli: Extra CLI arguments associated with the vectoriser.
    :type vectorizer_cli: Tuple[str, ...]
    """

    text_vectorizer: str
    vectorizer_tag: str
    learning_rate: float
    max_depth: int
    n_estimators: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float
    vectorizer_cli: Tuple[str, ...] = field(default_factory=tuple)

    def label(self) -> str:
        """
        Produce a filesystem- and report-friendly identifier.

        :param self: Sweep configuration instance being labelled.
        :type self: SweepConfig
        :returns: Composite label encoding vectoriser and booster parameters.
        :rtype: str
        """

        tag = self.vectorizer_tag or self.text_vectorizer
        base = (
            f"lr{self.learning_rate:g}_depth{self.max_depth}_"
            f"estim{self.n_estimators}_sub{self.subsample:g}_"
            f"col{self.colsample_bytree:g}_l2{self.reg_lambda:g}_l1{self.reg_alpha:g}"
        ).replace(".", "p")
        return f"{tag}_{base}"

    def booster_params(self, tree_method: str) -> XGBoostBoosterParams:
        """
        Convert the sweep configuration into :class:`~xgb.core.model.XGBoostBoosterParams`.

        :param tree_method: Tree construction algorithm passed to XGBoost.
        :type tree_method: str
        :returns: Booster parameter bundle mirroring this configuration.
        :rtype: XGBoostBoosterParams
        """

        return XGBoostBoosterParams.create(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method=tree_method,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
        )

    def cli_args(self, tree_method: str) -> List[str]:
        """
        Serialise the configuration into CLI arguments for :mod:`xgb.cli`.

        :param tree_method: Tree construction algorithm passed to XGBoost.
        :type tree_method: str
        :returns: List of CLI flags encoding the configuration.
        :rtype: List[str]
        """

        return [
            "--text_vectorizer",
            self.text_vectorizer,
            "--xgb_learning_rate",
            str(self.learning_rate),
            "--xgb_max_depth",
            str(self.max_depth),
            "--xgb_n_estimators",
            str(self.n_estimators),
            "--xgb_subsample",
            str(self.subsample),
            "--xgb_colsample_bytree",
            str(self.colsample_bytree),
            "--xgb_tree_method",
            tree_method,
            "--xgb_reg_lambda",
            str(self.reg_lambda),
            "--xgb_reg_alpha",
            str(self.reg_alpha),
        ] + list(self.vectorizer_cli)



# pylint: disable=too-many-instance-attributes
@dataclass
class SweepOutcome(BasePipelineSweepOutcome[SweepConfig]):
    """
    Metrics captured for a (study, configuration) sweep evaluation.

    Extends :class:`common.pipeline.types.BasePipelineSweepOutcome` with
    XGBoost-specific evaluation statistics.

    :param accuracy: Validation accuracy achieved by the configuration.
    :type accuracy: float
    :param coverage: Validation coverage achieved by the configuration.
    :type coverage: float
    :param evaluated: Number of evaluation rows.
    :type evaluated: int
    """

    accuracy: float
    coverage: float
    evaluated: int


@dataclass(frozen=True)
class SweepTask(  # pylint: disable=too-many-instance-attributes
    BaseSweepTask["SweepConfig"]
):
    """
    Extend :class:`common.opinion.sweep_types.BaseSweepTask` with XGBoost metadata.

    :param tree_method: Tree construction algorithm supplied to XGBoost.
    :type tree_method: str
    :param train_participant_studies: Participant study keys used for training.
    :type train_participant_studies: Tuple[str, ...]
    """

    tree_method: str
    train_participant_studies: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class StudySelection(BaseStudySelection[SweepOutcome]):
    """
    Selected configuration for the final evaluation of a participant study.

    :param study: Study metadata chosen for final evaluation.
    :type study: ~common.pipeline.types.StudySpec
    :param outcome: Winning sweep outcome leveraged for reporting.
    :type outcome: SweepOutcome
    """


@dataclass(frozen=True)
class SweepRunContext:
    """
    CLI arguments shared across sweep invocations.

    :param base_cli: Baseline CLI arguments applied to every sweep run.
    :type base_cli: Sequence[str]
    :param extra_cli: Additional CLI flags appended for each invocation.
    :type extra_cli: Sequence[str]
    :param sweep_dir: Root directory where sweep artefacts are written.
    :type sweep_dir: Path
    :param tree_method: Tree construction algorithm passed to XGBoost.
    :type tree_method: str
    :param jobs: Parallel worker count for sweep execution.
    :type jobs: int
    """

    base_cli: Sequence[str]
    extra_cli: Sequence[str]
    sweep_dir: Path
    tree_method: str
    jobs: int


@dataclass(frozen=True)
class FinalEvalContext:
    """
    Runtime configuration for final slate evaluations.

    :param base_cli: Baseline CLI arguments for :mod:`xgb.cli`.
    :type base_cli: Sequence[str]
    :param extra_cli: Additional CLI arguments forwarded to each invocation.
    :type extra_cli: Sequence[str]
    :param out_dir: Target directory for final evaluation artefacts.
    :type out_dir: Path
    :param tree_method: Tree construction algorithm passed to XGBoost.
    :type tree_method: str
    :param save_model_dir: Optional directory for persisted models.
    :type save_model_dir: Optional[Path]
    :param reuse_existing: Flag controlling reuse of cached metrics.
    :type reuse_existing: bool
    """

    base_cli: Sequence[str]
    extra_cli: Sequence[str]
    out_dir: Path
    tree_method: str
    save_model_dir: Path | None
    reuse_existing: bool


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class OpinionStageConfig:
    """
    Inputs required to launch the opinion regression stage.

    :param dataset: Dataset identifier passed to the opinion stage.
    :type dataset: str
    :param cache_dir: Cache directory for dataset loading.
    :type cache_dir: str
    :param base_out_dir: Base output directory for opinion artefacts.
    :type base_out_dir: Path
    :param extra_fields: Additional prompt columns appended to documents.
    :type extra_fields: Sequence[str]
    :param studies: Opinion study keys to evaluate.
    :type studies: Sequence[str]
    :param max_participants: Optional cap on participants per study.
    :type max_participants: int
    :param seed: Random seed for subsampling.
    :type seed: int
    :param max_features: Maximum TF-IDF features (``None`` keeps all).
    :type max_features: Optional[int]
    :param tree_method: Tree construction algorithm passed to XGBoost.
    :type tree_method: str
    :param overwrite: Flag controlling whether existing artefacts may be overwritten.
    :type overwrite: bool
    :param tfidf_config: TF-IDF vectoriser configuration applied during evaluation.
    :type tfidf_config: TfidfConfig
    :param word2vec_config: Word2Vec vectoriser configuration applied during evaluation.
    :type word2vec_config: Word2VecVectorizerConfig
    :param sentence_transformer_config: Sentence-transformer configuration for evaluations.
    :type sentence_transformer_config: SentenceTransformerVectorizerConfig
    :param word2vec_model_base: Optional base directory for Word2Vec model artefacts.
    :type word2vec_model_base: Optional[Path]
    :param reuse_existing: Flag enabling reuse of cached results.
    :type reuse_existing: bool
    """

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
        """Allow construction from either a composed base or flat kwargs.

        Flat kwargs expected: order_index, study, config, mae, rmse,
        artifact, accuracy_summary, r_squared.
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
            # ignore any leftover unknown kwargs for robustness
        if r_squared is None:
            r_squared = kwargs.pop("r_squared", None)
        if r_squared is None:
            raise TypeError("OpinionSweepOutcome requires r_squared")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "r_squared", float(r_squared))

    # Compatibility properties mirroring BaseOpinionSweepOutcome
    @property
    def order_index(self) -> int:  # pragma: no cover - simple forwarding
        """Deterministic ordering index assigned to the task.

        :rtype: int
        """
        return self.base.order_index

    @property
    def study(self) -> StudySpec:  # pragma: no cover - simple forwarding
        """Study metadata associated with the sweep.

        :rtype: ~common.pipeline.types.StudySpec
        """
        return self.base.study

    @property
    def config(self) -> SweepConfig:  # pragma: no cover - simple forwarding
        """Evaluated sweep configuration.

        :rtype: ~xgb.pipeline.context.SweepConfig
        """
        return self.base.config

    @property
    def mae(self) -> float:  # pragma: no cover - simple forwarding
        """Mean absolute error achieved by the configuration.

        :rtype: float
        """
        return self.base.mae

    @property
    def rmse(self) -> float:  # pragma: no cover - simple forwarding
        """Root mean squared error achieved by the configuration.

        :rtype: float
        """
        return self.base.rmse

    @property
    def artifact(self):  # pragma: no cover - simple forwarding
        """Metrics artefact containing the payload and storage path.

        :rtype: ~common.opinion.sweep_types.MetricsArtifact
        """
        return self.base.artifact

    @property
    def accuracy_summary(self):  # pragma: no cover - simple forwarding
        """Summary of directional accuracy metrics.

        :rtype: ~common.opinion.sweep_types.AccuracySummary
        """
        return self.base.accuracy_summary

    @property
    def metrics_path(self):  # pragma: no cover - simple forwarding
        """Filesystem path to the persisted metrics artefact.

        :rtype: ~pathlib.Path
        """
        return self.base.metrics_path

    @property
    def metrics(self):  # pragma: no cover - simple forwarding
        """Raw metrics payload loaded from disk.

        :rtype: Mapping[str, object]
        """
        return self.base.metrics

    @property
    def accuracy(self):  # pragma: no cover - simple forwarding
        """Directional accuracy achieved by the configuration.

        :rtype: Optional[float]
        """
        return self.base.accuracy

    @property
    def baseline_accuracy(self):  # pragma: no cover - simple forwarding
        """Directional accuracy achieved by the baseline configuration.

        :rtype: Optional[float]
        """
        return self.base.baseline_accuracy

    @property
    def accuracy_delta(self):  # pragma: no cover - simple forwarding
        """Improvement in accuracy over the baseline configuration.

        :rtype: Optional[float]
        """
        return self.base.accuracy_delta

    @property
    def eligible(self):  # pragma: no cover - simple forwarding
        """Number of evaluation examples contributing to accuracy metrics.

        :rtype: Optional[int]
        """
        return self.base.eligible


@dataclass(frozen=True)
class OpinionSweepTask(BaseOpinionSweepTask[SweepConfig]):
    """
    Extend :class:`common.opinion.sweep_types.BaseOpinionSweepTask` with the
    keyword arguments required by the XGBoost implementation.

    :param request_args: Keyword arguments passed to :func:`~knn.core.opinion.run_opinion_eval`.
    :type request_args: Mapping[str, object]
    :param feature_space: Feature space evaluated by the sweep task.
    :type feature_space: str
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

    # Backwards-compat initialiser: supports both grouped settings and legacy
    # flat keyword arguments used across the codebase and tests. This inflates
    # the apparent number of locals, so we selectively disable the lint.
    def __init__(  # pylint: disable=too-many-locals
        self,
        *,
        sweep_dir: Path,
        data: OpinionDataSettings | None = None,
        vectorizers: OpinionVectorizerSettings | None = None,
        xgb: OpinionXgbSettings | None = None,
        # Legacy flat kwargs for backwards compatibility
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
                sentence_transformer_config=
                (sentence_transformer_config or SentenceTransformerVectorizerConfig()),
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
        """Dataset identifier passed to opinion sweeps.

        :rtype: str
        """
        return self.data.dataset

    @property
    def cache_dir(self) -> str:  # pragma: no cover - simple forwarding
        """Dataset cache directory used during loading.

        :rtype: str
        """
        return self.data.cache_dir

    @property
    def extra_fields(self) -> Sequence[str]:  # pragma: no cover - simple forwarding
        """Extra text columns appended to prompt documents.

        :rtype: Sequence[str]
        """
        return self.data.extra_fields

    @property
    def max_participants(self) -> int:  # pragma: no cover - simple forwarding
        """Cap on participants per study (sampling control).

        :rtype: int
        """
        return self.data.max_participants

    @property
    def seed(self) -> int:  # pragma: no cover - simple forwarding
        """Random seed used for deterministic sampling.

        :rtype: int
        """
        return self.data.seed

    @property
    def max_features(self) -> int | None:  # pragma: no cover - simple forwarding
        """Maximum number of TF-IDF features, if constrained.

        :rtype: Optional[int]
        """
        return self.data.max_features

    @property
    def tfidf_config(self) -> TfidfConfig:  # pragma: no cover - simple forwarding
        """TF-IDF vectoriser configuration.

        :rtype: ~xgb.core.vectorizers.TfidfConfig
        """
        return self.vectorizers.tfidf_config

    @property
    def word2vec_config(self) -> Word2VecVectorizerConfig:  # pragma: no cover
        """Word2Vec vectoriser configuration.

        :rtype: ~xgb.core.vectorizers.Word2VecVectorizerConfig
        """
        return self.vectorizers.word2vec_config

    @property
    def sentence_transformer_config(
        self,
    ) -> SentenceTransformerVectorizerConfig:  # pragma: no cover
        """Sentence-transformer configuration for embedding extraction.

        :rtype: ~xgb.core.vectorizers.SentenceTransformerVectorizerConfig
        """
        return self.vectorizers.sentence_transformer_config

    @property
    def word2vec_model_base(self) -> Path | None:  # pragma: no cover
        """Optional base directory for persisted Word2Vec models.

        :rtype: Optional[~pathlib.Path]
        """
        return self.vectorizers.word2vec_model_base

    @property
    def tree_method(self) -> str:  # pragma: no cover - simple forwarding
        """XGBoost tree construction algorithm.

        :rtype: str
        """
        return self.xgb.tree_method

    @property
    def overwrite(self) -> bool:  # pragma: no cover - simple forwarding
        """Whether to overwrite existing artefacts during sweeps.

        :rtype: bool
        """
        return self.xgb.overwrite


@dataclass(frozen=True)
class _NextVideoCore:
    """Core next-video metrics and baselines."""

    accuracy: Optional[float] = None
    coverage: Optional[float] = None
    accuracy_eligible: Optional[float] = None
    evaluated: Optional[int] = None
    correct: Optional[int] = None
    correct_eligible: Optional[int] = None
    eligible: Optional[int] = None
    known_hits: Optional[int] = None
    known_total: Optional[int] = None
    known_availability: Optional[float] = None
    avg_probability: Optional[float] = None
    baseline_most_frequent_accuracy: Optional[float] = None
    random_baseline_accuracy: Optional[float] = None


@dataclass(frozen=True)
class _NextVideoMeta:
    """Metadata describing the evaluation context for next-video metrics."""

    dataset: Optional[str] = None
    issue: Optional[str] = None
    issue_label: Optional[str] = None
    study_label: Optional[str] = None


@dataclass(frozen=True)
class NextVideoMetricSummary:
    """Grouped next-video metrics with compatibility accessors."""

    core: _NextVideoCore
    meta: _NextVideoMeta

    # Compatibility properties
    @property
    def accuracy(self) -> Optional[float]:  # pragma: no cover - forwarding
        return self.core.accuracy

    @property
    def coverage(self) -> Optional[float]:  # pragma: no cover
        return self.core.coverage

    @property
    def accuracy_eligible(self) -> Optional[float]:  # pragma: no cover
        return self.core.accuracy_eligible

    @property
    def evaluated(self) -> Optional[int]:  # pragma: no cover
        return self.core.evaluated

    @property
    def correct(self) -> Optional[int]:  # pragma: no cover
        return self.core.correct

    @property
    def correct_eligible(self) -> Optional[int]:  # pragma: no cover
        return self.core.correct_eligible

    @property
    def eligible(self) -> Optional[int]:  # pragma: no cover
        return self.core.eligible

    @property
    def known_hits(self) -> Optional[int]:  # pragma: no cover
        return self.core.known_hits

    @property
    def known_total(self) -> Optional[int]:  # pragma: no cover
        return self.core.known_total

    @property
    def known_availability(self) -> Optional[float]:  # pragma: no cover
        return self.core.known_availability

    @property
    def avg_probability(self) -> Optional[float]:  # pragma: no cover
        return self.core.avg_probability

    @property
    def baseline_most_frequent_accuracy(self) -> Optional[float]:  # pragma: no cover
        return self.core.baseline_most_frequent_accuracy

    @property
    def random_baseline_accuracy(self) -> Optional[float]:  # pragma: no cover
        return self.core.random_baseline_accuracy

    @property
    def dataset(self) -> Optional[str]:  # pragma: no cover
        return self.meta.dataset

    @property
    def issue(self) -> Optional[str]:  # pragma: no cover
        return self.meta.issue

    @property
    def issue_label(self) -> Optional[str]:  # pragma: no cover
        return self.meta.issue_label

    @property
    def study_label(self) -> Optional[str]:  # pragma: no cover
        return self.meta.study_label

    @classmethod
    def create(
        cls,
        *,
        accuracy: Optional[float] = None,
        coverage: Optional[float] = None,
        accuracy_eligible: Optional[float] = None,
        evaluated: Optional[int] = None,
        correct: Optional[int] = None,
        correct_eligible: Optional[int] = None,
        eligible: Optional[int] = None,
        known_hits: Optional[int] = None,
        known_total: Optional[int] = None,
        known_availability: Optional[float] = None,
        avg_probability: Optional[float] = None,
        baseline_most_frequent_accuracy: Optional[float] = None,
        random_baseline_accuracy: Optional[float] = None,
        dataset: Optional[str] = None,
        issue: Optional[str] = None,
        issue_label: Optional[str] = None,
        study_label: Optional[str] = None,
    ) -> "NextVideoMetricSummary":
        core = _NextVideoCore(
            accuracy=accuracy,
            coverage=coverage,
            accuracy_eligible=accuracy_eligible,
            evaluated=evaluated,
            correct=correct,
            correct_eligible=correct_eligible,
            eligible=eligible,
            known_hits=known_hits,
            known_total=known_total,
            known_availability=known_availability,
            avg_probability=avg_probability,
            baseline_most_frequent_accuracy=baseline_most_frequent_accuracy,
            random_baseline_accuracy=random_baseline_accuracy,
        )
        meta = _NextVideoMeta(
            dataset=dataset,
            issue=issue,
            issue_label=issue_label,
            study_label=study_label,
        )
        return cls(core=core, meta=meta)


@dataclass(frozen=True)
class _OpinionAfter:
    mae_after: Optional[float] = None
    mae_change: Optional[float] = None
    rmse_after: Optional[float] = None
    r2_after: Optional[float] = None
    rmse_change: Optional[float] = None
    accuracy_after: Optional[float] = None


@dataclass(frozen=True)
class _OpinionBaseline:
    baseline_mae: Optional[float] = None
    baseline_rmse_change: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    baseline_calibration_slope: Optional[float] = None
    baseline_calibration_intercept: Optional[float] = None
    baseline_calibration_ece: Optional[float] = None
    baseline_kl_divergence_change: Optional[float] = None


@dataclass(frozen=True)
class _OpinionMeta:
    participants: Optional[int] = None
    eligible: Optional[int] = None
    dataset: Optional[str] = None
    split: Optional[str] = None
    label: Optional[str] = None


@dataclass(frozen=True)
class _OpinionCalibration:
    calibration_slope: Optional[float] = None
    calibration_intercept: Optional[float] = None
    calibration_ece: Optional[float] = None
    kl_divergence_change: Optional[float] = None


@dataclass(frozen=True)
class _OpinionDeltas:
    mae_delta: Optional[float] = None
    accuracy_delta: Optional[float] = None


@dataclass(frozen=True)
class OpinionSummary:
    """Grouped opinion-regression metrics with compatibility accessors."""

    after: _OpinionAfter
    baseline: _OpinionBaseline
    calibration: _OpinionCalibration
    deltas: _OpinionDeltas
    meta: _OpinionMeta

    # Properties exposing the flat attribute interface used by report code
    @property
    def mae_after(self) -> Optional[float]:  # pragma: no cover - forwarding
        return self.after.mae_after

    @property
    def mae_change(self) -> Optional[float]:  # pragma: no cover
        return self.after.mae_change

    @property
    def rmse_after(self) -> Optional[float]:  # pragma: no cover
        return self.after.rmse_after

    @property
    def r2_after(self) -> Optional[float]:  # pragma: no cover
        return self.after.r2_after

    @property
    def rmse_change(self) -> Optional[float]:  # pragma: no cover
        return self.after.rmse_change

    @property
    def accuracy_after(self) -> Optional[float]:  # pragma: no cover
        return self.after.accuracy_after

    @property
    def baseline_mae(self) -> Optional[float]:  # pragma: no cover
        return self.baseline.baseline_mae

    @property
    def baseline_rmse_change(self) -> Optional[float]:  # pragma: no cover
        return self.baseline.baseline_rmse_change

    @property
    def baseline_accuracy(self) -> Optional[float]:  # pragma: no cover
        return self.baseline.baseline_accuracy

    @property
    def calibration_slope(self) -> Optional[float]:  # pragma: no cover
        return self.calibration.calibration_slope

    @property
    def baseline_calibration_slope(self) -> Optional[float]:  # pragma: no cover
        return self.baseline.baseline_calibration_slope

    @property
    def calibration_intercept(self) -> Optional[float]:  # pragma: no cover
        return self.calibration.calibration_intercept

    @property
    def baseline_calibration_intercept(self) -> Optional[float]:  # pragma: no cover
        return self.baseline.baseline_calibration_intercept

    @property
    def calibration_ece(self) -> Optional[float]:  # pragma: no cover
        return self.calibration.calibration_ece

    @property
    def baseline_calibration_ece(self) -> Optional[float]:  # pragma: no cover
        return self.baseline.baseline_calibration_ece

    @property
    def kl_divergence_change(self) -> Optional[float]:  # pragma: no cover
        return self.calibration.kl_divergence_change

    @property
    def baseline_kl_divergence_change(self) -> Optional[float]:  # pragma: no cover
        return self.baseline.baseline_kl_divergence_change

    @property
    def participants(self) -> Optional[int]:  # pragma: no cover
        return self.meta.participants

    @property
    def eligible(self) -> Optional[int]:  # pragma: no cover
        return self.meta.eligible

    @property
    def dataset(self) -> Optional[str]:  # pragma: no cover
        return self.meta.dataset

    @property
    def split(self) -> Optional[str]:  # pragma: no cover
        return self.meta.split

    @property
    def label(self) -> Optional[str]:  # pragma: no cover
        return self.meta.label

    @property
    def mae_delta(self) -> Optional[float]:  # pragma: no cover
        return self.deltas.mae_delta

    @property
    def accuracy_delta(self) -> Optional[float]:  # pragma: no cover
        return self.deltas.accuracy_delta

    @classmethod
    def from_kwargs(cls, **kwargs) -> "OpinionSummary":
        """Construct a grouped summary from flat keyword arguments."""

        after = _OpinionAfter(
            mae_after=kwargs.get("mae_after"),
            mae_change=kwargs.get("mae_change"),
            rmse_after=kwargs.get("rmse_after"),
            r2_after=kwargs.get("r2_after"),
            rmse_change=kwargs.get("rmse_change"),
            accuracy_after=kwargs.get("accuracy_after"),
        )
        baseline = _OpinionBaseline(
            baseline_mae=kwargs.get("baseline_mae"),
            baseline_rmse_change=kwargs.get("baseline_rmse_change"),
            baseline_accuracy=kwargs.get("baseline_accuracy"),
            baseline_calibration_slope=kwargs.get("baseline_calibration_slope"),
            baseline_calibration_intercept=kwargs.get("baseline_calibration_intercept"),
            baseline_calibration_ece=kwargs.get("baseline_calibration_ece"),
            baseline_kl_divergence_change=kwargs.get("baseline_kl_divergence_change"),
        )
        calibration = _OpinionCalibration(
            calibration_slope=kwargs.get("calibration_slope"),
            calibration_intercept=kwargs.get("calibration_intercept"),
            calibration_ece=kwargs.get("calibration_ece"),
            kl_divergence_change=kwargs.get("kl_divergence_change"),
        )
        deltas = _OpinionDeltas(
            mae_delta=kwargs.get("mae_delta"),
            accuracy_delta=kwargs.get("accuracy_delta"),
        )
        meta = _OpinionMeta(
            participants=kwargs.get("participants"),
            eligible=kwargs.get("eligible"),
            dataset=kwargs.get("dataset"),
            split=kwargs.get("split"),
            label=kwargs.get("label"),
        )
        return cls(
            after=after,
            baseline=baseline,
            calibration=calibration,
            deltas=deltas,
            meta=meta,
        )


__all__ = [
    "FinalEvalContext",
    "NextVideoMetricSummary",
    "OpinionStageConfig",
    "OpinionSummary",
    "OpinionStudySelection",
    "OpinionSweepOutcome",
    "OpinionSweepTask",
    "OpinionSweepRunContext",
    "StudySelection",
    "StudySpec",
    "OpinionDataSettings",
    "OpinionVectorizerSettings",
    "OpinionXgbSettings",
    "SweepConfig",
    "SweepOutcome",
    "SweepRunContext",
    "SweepTask",
]
