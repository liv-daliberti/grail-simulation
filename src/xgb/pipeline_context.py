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
from typing import List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from common.pipeline_types import (
    OpinionStudySelection as BaseOpinionStudySelection,
    StudySelection as BaseStudySelection,
    StudySpec,
)

from .model import XGBoostBoosterParams


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
        Convert the sweep configuration into :class:`XGBoostBoosterParams`.

        :param tree_method: Tree construction algorithm passed to XGBoost.
        :type tree_method: str
        :returns: Booster parameter bundle mirroring this configuration.
        :rtype: XGBoostBoosterParams
        """

        return XGBoostBoosterParams(
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
class SweepOutcome:
    """
    Metrics captured for a (study, configuration) sweep evaluation.

    :param order_index: Deterministic ordering index assigned to the task.
    :type order_index: int
    :param study: Study metadata associated with the sweep.
    :type study: StudySpec
    :param config: Evaluated sweep configuration.
    :type config: SweepConfig
    :param accuracy: Validation accuracy achieved by the configuration.
    :type accuracy: float
    :param coverage: Validation coverage achieved by the configuration.
    :type coverage: float
    :param evaluated: Number of evaluation rows.
    :type evaluated: int
    :param metrics_path: Filesystem path to the metrics artefact.
    :type metrics_path: Path
    :param metrics: Raw metrics payload loaded from disk.
    :type metrics: Mapping[str, object]
    """

    order_index: int
    study: StudySpec
    config: SweepConfig
    accuracy: float
    coverage: float
    evaluated: int
    metrics_path: Path
    metrics: Mapping[str, object]


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class SweepTask:
    """
    Container describing a single sweep execution request.

    :param index: Stable index used for ordering and scheduling.
    :type index: int
    :param study: Study metadata being evaluated.
    :type study: StudySpec
    :param config: Sweep configuration executed for the study.
    :type config: SweepConfig
    :param base_cli: Baseline CLI arguments shared across tasks.
    :type base_cli: Tuple[str, ...]
    :param extra_cli: Additional passthrough CLI arguments.
    :type extra_cli: Tuple[str, ...]
    :param run_root: Directory under which sweep artefacts are stored.
    :type run_root: Path
    :param tree_method: Tree construction algorithm supplied to XGBoost.
    :type tree_method: str
    :param metrics_path: Target path for the ``metrics.json`` artefact.
    :type metrics_path: Path
    """

    index: int
    study: StudySpec
    config: SweepConfig
    base_cli: Tuple[str, ...]
    extra_cli: Tuple[str, ...]
    run_root: Path
    tree_method: str
    metrics_path: Path


@dataclass
class StudySelection(BaseStudySelection[SweepOutcome]):
    """
    Selected configuration for the final evaluation of a participant study.

    :param study: Study metadata chosen for final evaluation.
    :type study: StudySpec
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
    reuse_existing: bool


# pylint: disable=too-many-instance-attributes
@dataclass
class OpinionSweepOutcome:
    """
    Metrics captured for a (study, configuration) combination during opinion sweeps.

    :param order_index: Deterministic ordering index assigned to the task.
    :type order_index: int
    :param study: Study metadata associated with the sweep.
    :type study: StudySpec
    :param config: Evaluated sweep configuration.
    :type config: SweepConfig
    :param mae: Mean absolute error achieved by the configuration.
    :type mae: float
    :param rmse: Root mean squared error achieved by the configuration.
    :type rmse: float
    :param r_squared: Coefficient of determination achieved by the configuration.
    :type r_squared: float
    :param metrics_path: Filesystem path to the metrics artefact.
    :type metrics_path: Path
    :param metrics: Raw metrics payload loaded from disk.
    :type metrics: Mapping[str, object]
    :param accuracy: Directional accuracy achieved by the configuration.
    :type accuracy: Optional[float]
    :param baseline_accuracy: Directional accuracy achieved by the baseline.
    :type baseline_accuracy: Optional[float]
    :param accuracy_delta: Improvement in accuracy over the baseline.
    :type accuracy_delta: Optional[float]
    :param eligible: Number of evaluation examples contributing to accuracy metrics.
    :type eligible: Optional[int]
    """

    order_index: int
    study: StudySpec
    config: SweepConfig
    mae: float
    rmse: float
    r_squared: float
    metrics_path: Path
    metrics: Mapping[str, object]
    accuracy: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    accuracy_delta: Optional[float] = None
    eligible: Optional[int] = None


@dataclass(frozen=True)
class OpinionSweepTask:
    """
    Container describing a single opinion sweep execution request.

    :param index: Stable index used for ordering and scheduling.
    :type index: int
    :param study: Study metadata being evaluated.
    :type study: StudySpec
    :param config: Sweep configuration executed for the opinion task.
    :type config: SweepConfig
    :param request_args: Keyword arguments passed to :func:`run_opinion_eval`.
    :type request_args: Mapping[str, object]
    """

    index: int
    study: StudySpec
    config: SweepConfig
    request_args: Mapping[str, object]
    metrics_path: Path

# pylint: disable=too-few-public-methods
# Avoid triggering runtime generic parameter validation while still exposing
# the specialised type to static analyzers.
if TYPE_CHECKING:
    OpinionSelectionBase = BaseOpinionStudySelection[OpinionSweepOutcome]
else:
    OpinionSelectionBase = BaseOpinionStudySelection


class OpinionStudySelection(OpinionSelectionBase):
    """
    Selected configuration for the final opinion regression evaluation.

    :param study: Study metadata chosen for final evaluation.
    :type study: StudySpec
    :param outcome: Winning opinion sweep outcome leveraged for reporting.
    :type outcome: OpinionSweepOutcome
    """


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class OpinionSweepRunContext:
    """
    Configuration shared across opinion sweep evaluations.

    :param dataset: Dataset identifier passed to opinion sweeps.
    :type dataset: str
    :param cache_dir: Cache directory leveraged by dataset loading.
    :type cache_dir: str
    :param sweep_dir: Root directory where opinion sweep artefacts are stored.
    :type sweep_dir: Path
    :param extra_fields: Additional prompt columns appended to documents.
    :type extra_fields: Sequence[str]
    :param max_participants: Optional cap on participants per study.
    :type max_participants: int
    :param seed: Random seed for subsampling.
    :type seed: int
    :param max_features: Maximum TF-IDF features (``None`` allows all).
    :type max_features: Optional[int]
    :param tree_method: Tree construction algorithm passed to XGBoost.
    :type tree_method: str
    :param overwrite: Flag controlling whether existing artefacts may be overwritten.
    :type overwrite: bool
    """

    dataset: str
    cache_dir: str
    sweep_dir: Path
    extra_fields: Sequence[str]
    max_participants: int
    seed: int
    max_features: int | None
    tree_method: str
    overwrite: bool


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class NextVideoMetricSummary:
    """
    Normalised view of slate metrics emitted by the XGBoost evaluations.

    :param accuracy: Validation accuracy (``None`` when unavailable).
    :type accuracy: Optional[float]
    :param coverage: Validation coverage capturing known candidate recall.
    :type coverage: Optional[float]
    :param evaluated: Number of evaluation rows.
    :type evaluated: Optional[int]
    :param correct: Number of correct predictions.
    :type correct: Optional[int]
    :param known_hits: Correct predictions among known candidates.
    :type known_hits: Optional[int]
    :param known_total: Evaluations featuring at least one known candidate.
    :type known_total: Optional[int]
    :param known_availability: Fraction of evaluations containing a known candidate.
    :type known_availability: Optional[float]
    :param avg_probability: Mean probability recorded for known predictions.
    :type avg_probability: Optional[float]
    :param dataset: Dataset identifier.
    :type dataset: Optional[str]
    :param issue: Issue key under evaluation.
    :type issue: Optional[str]
    :param issue_label: Human-readable issue label.
    :type issue_label: Optional[str]
    :param study_label: Human-readable study label.
    :type study_label: Optional[str]
    """

    accuracy: Optional[float] = None
    coverage: Optional[float] = None
    evaluated: Optional[int] = None
    correct: Optional[int] = None
    known_hits: Optional[int] = None
    known_total: Optional[int] = None
    known_availability: Optional[float] = None
    avg_probability: Optional[float] = None
    dataset: Optional[str] = None
    issue: Optional[str] = None
    issue_label: Optional[str] = None
    study_label: Optional[str] = None


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class OpinionSummary:
    """
    Normalised view of opinion-regression metrics.

    :param mae_after: Mean absolute error achieved by the regressor.
    :type mae_after: Optional[float]
    :param rmse_after: Root mean squared error achieved by the regressor.
    :type rmse_after: Optional[float]
    :param r2_after: Coefficient of determination achieved by the regressor.
    :type r2_after: Optional[float]
    :param baseline_mae: Baseline MAE using the no-change predictor.
    :type baseline_mae: Optional[float]
    :param mae_delta: Absolute MAE improvement over the baseline.
    :type mae_delta: Optional[float]
    :param accuracy_after: Directional accuracy achieved by the regressor.
    :type accuracy_after: Optional[float]
    :param baseline_accuracy: Directional accuracy achieved by the baseline.
    :type baseline_accuracy: Optional[float]
    :param accuracy_delta: Improvement in directional accuracy over the baseline.
    :type accuracy_delta: Optional[float]
    :param participants: Number of participants in the evaluation split.
    :type participants: Optional[int]
    :param eligible: Number of evaluation examples contributing to accuracy metrics.
    :type eligible: Optional[int]
    :param dataset: Dataset identifier.
    :type dataset: Optional[str]
    :param split: Evaluation split name.
    :vartype split: Optional[str]
    :ivar label: Human-readable study label.
    :vartype label: Optional[str]
    """

    mae_after: Optional[float] = None
    rmse_after: Optional[float] = None
    r2_after: Optional[float] = None
    baseline_mae: Optional[float] = None
    mae_delta: Optional[float] = None
    accuracy_after: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    accuracy_delta: Optional[float] = None
    participants: Optional[int] = None
    eligible: Optional[int] = None
    dataset: Optional[str] = None
    split: Optional[str] = None
    label: Optional[str] = None


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
    "SweepConfig",
    "SweepOutcome",
    "SweepRunContext",
    "SweepTask",
]
