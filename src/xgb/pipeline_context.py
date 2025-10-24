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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
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

    :ivar text_vectorizer: Vectoriser identifier (e.g. ``tfidf``).
    :vartype text_vectorizer: str
    :ivar vectorizer_tag: Short tag used in directory and report labels.
    :vartype vectorizer_tag: str
    :ivar learning_rate: Booster learning rate.
    :vartype learning_rate: float
    :ivar max_depth: Tree depth explored during sweeps.
    :vartype max_depth: int
    :ivar n_estimators: Number of boosting rounds.
    :vartype n_estimators: int
    :ivar subsample: Row subsampling ratio.
    :vartype subsample: float
    :ivar colsample_bytree: Column subsampling ratio per tree.
    :vartype colsample_bytree: float
    :ivar reg_lambda: L2 regularisation weight.
    :vartype reg_lambda: float
    :ivar reg_alpha: L1 regularisation weight.
    :vartype reg_alpha: float
    :ivar vectorizer_cli: Extra CLI arguments associated with the vectoriser.
    :vartype vectorizer_cli: Tuple[str, ...]
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

    :ivar order_index: Deterministic ordering index assigned to the task.
    :vartype order_index: int
    :ivar study: Study metadata associated with the sweep.
    :vartype study: StudySpec
    :ivar config: Evaluated sweep configuration.
    :vartype config: SweepConfig
    :ivar accuracy: Validation accuracy achieved by the configuration.
    :vartype accuracy: float
    :ivar coverage: Validation coverage achieved by the configuration.
    :vartype coverage: float
    :ivar evaluated: Number of evaluation rows.
    :vartype evaluated: int
    :ivar metrics_path: Filesystem path to the metrics artefact.
    :vartype metrics_path: Path
    :ivar metrics: Raw metrics payload loaded from disk.
    :vartype metrics: Mapping[str, object]
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

    :ivar index: Stable index used for ordering and scheduling.
    :vartype index: int
    :ivar study: Study metadata being evaluated.
    :vartype study: StudySpec
    :ivar config: Sweep configuration executed for the study.
    :vartype config: SweepConfig
    :ivar base_cli: Baseline CLI arguments shared across tasks.
    :vartype base_cli: Tuple[str, ...]
    :ivar extra_cli: Additional passthrough CLI arguments.
    :vartype extra_cli: Tuple[str, ...]
    :ivar run_root: Directory under which sweep artefacts are stored.
    :vartype run_root: Path
    :ivar tree_method: Tree construction algorithm supplied to XGBoost.
    :vartype tree_method: str
    :ivar metrics_path: Target path for the ``metrics.json`` artefact.
    :vartype metrics_path: Path
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

    :ivar study: Study metadata chosen for final evaluation.
    :vartype study: StudySpec
    :ivar outcome: Winning sweep outcome leveraged for reporting.
    :vartype outcome: SweepOutcome
    """


@dataclass(frozen=True)
class SweepRunContext:
    """
    CLI arguments shared across sweep invocations.

    :ivar base_cli: Baseline CLI arguments applied to every sweep run.
    :vartype base_cli: Sequence[str]
    :ivar extra_cli: Additional CLI flags appended for each invocation.
    :vartype extra_cli: Sequence[str]
    :ivar sweep_dir: Root directory where sweep artefacts are written.
    :vartype sweep_dir: Path
    :ivar tree_method: Tree construction algorithm passed to XGBoost.
    :vartype tree_method: str
    :ivar jobs: Parallel worker count for sweep execution.
    :vartype jobs: int
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

    :ivar base_cli: Baseline CLI arguments for :mod:`xgb.cli`.
    :vartype base_cli: Sequence[str]
    :ivar extra_cli: Additional CLI arguments forwarded to each invocation.
    :vartype extra_cli: Sequence[str]
    :ivar out_dir: Target directory for final evaluation artefacts.
    :vartype out_dir: Path
    :ivar tree_method: Tree construction algorithm passed to XGBoost.
    :vartype tree_method: str
    :ivar save_model_dir: Optional directory for persisted models.
    :vartype save_model_dir: Optional[Path]
    :ivar reuse_existing: Flag controlling reuse of cached metrics.
    :vartype reuse_existing: bool
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

    :ivar dataset: Dataset identifier passed to the opinion stage.
    :vartype dataset: str
    :ivar cache_dir: Cache directory for dataset loading.
    :vartype cache_dir: str
    :ivar base_out_dir: Base output directory for opinion artefacts.
    :vartype base_out_dir: Path
    :ivar extra_fields: Additional prompt columns appended to documents.
    :vartype extra_fields: Sequence[str]
    :ivar studies: Opinion study keys to evaluate.
    :vartype studies: Sequence[str]
    :ivar max_participants: Optional cap on participants per study.
    :vartype max_participants: int
    :ivar seed: Random seed for subsampling.
    :vartype seed: int
    :ivar max_features: Maximum TF-IDF features (``None`` keeps all).
    :vartype max_features: Optional[int]
    :ivar tree_method: Tree construction algorithm passed to XGBoost.
    :vartype tree_method: str
    :ivar overwrite: Flag controlling whether existing artefacts may be overwritten.
    :vartype overwrite: bool
    :ivar reuse_existing: Flag enabling reuse of cached results.
    :vartype reuse_existing: bool
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

    :ivar order_index: Deterministic ordering index assigned to the task.
    :vartype order_index: int
    :ivar study: Study metadata associated with the sweep.
    :vartype study: StudySpec
    :ivar config: Evaluated sweep configuration.
    :vartype config: SweepConfig
    :ivar mae: Mean absolute error achieved by the configuration.
    :vartype mae: float
    :ivar rmse: Root mean squared error achieved by the configuration.
    :vartype rmse: float
    :ivar r_squared: Coefficient of determination achieved by the configuration.
    :vartype r_squared: float
    :ivar metrics_path: Filesystem path to the metrics artefact.
    :vartype metrics_path: Path
    :ivar metrics: Raw metrics payload loaded from disk.
    :vartype metrics: Mapping[str, object]
    :ivar accuracy: Directional accuracy achieved by the configuration.
    :vartype accuracy: Optional[float]
    :ivar baseline_accuracy: Directional accuracy achieved by the baseline.
    :vartype baseline_accuracy: Optional[float]
    :ivar accuracy_delta: Improvement in accuracy over the baseline.
    :vartype accuracy_delta: Optional[float]
    :ivar eligible: Number of evaluation examples contributing to accuracy metrics.
    :vartype eligible: Optional[int]
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

    :ivar index: Stable index used for ordering and scheduling.
    :vartype index: int
    :ivar study: Study metadata being evaluated.
    :vartype study: StudySpec
    :ivar config: Sweep configuration executed for the opinion task.
    :vartype config: SweepConfig
    :ivar request_args: Keyword arguments passed to :func:`run_opinion_eval`.
    :vartype request_args: Mapping[str, object]
    :ivar metrics_path: Target path for the ``metrics.json`` artefact.
    :vartype metrics_path: Path
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

    :ivar study: Study metadata chosen for final evaluation.
    :vartype study: StudySpec
    :ivar outcome: Winning opinion sweep outcome leveraged for reporting.
    :vartype outcome: OpinionSweepOutcome
    """


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class OpinionSweepRunContext:
    """
    Configuration shared across opinion sweep evaluations.

    :ivar dataset: Dataset identifier passed to opinion sweeps.
    :vartype dataset: str
    :ivar cache_dir: Cache directory leveraged by dataset loading.
    :vartype cache_dir: str
    :ivar sweep_dir: Root directory where opinion sweep artefacts are stored.
    :vartype sweep_dir: Path
    :ivar extra_fields: Additional prompt columns appended to documents.
    :vartype extra_fields: Sequence[str]
    :ivar max_participants: Optional cap on participants per study.
    :vartype max_participants: int
    :ivar seed: Random seed for subsampling.
    :vartype seed: int
    :ivar max_features: Maximum TF-IDF features (``None`` allows all).
    :vartype max_features: Optional[int]
    :ivar tree_method: Tree construction algorithm passed to XGBoost.
    :vartype tree_method: str
    :ivar overwrite: Flag controlling whether existing artefacts may be overwritten.
    :vartype overwrite: bool
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

    :ivar accuracy: Validation accuracy (``None`` when unavailable).
    :vartype accuracy: Optional[float]
    :ivar coverage: Validation coverage capturing known candidate recall.
    :vartype coverage: Optional[float]
    :ivar evaluated: Number of evaluation rows.
    :vartype evaluated: Optional[int]
    :ivar correct: Number of correct predictions.
    :vartype correct: Optional[int]
    :ivar known_hits: Correct predictions among known candidates.
    :vartype known_hits: Optional[int]
    :ivar known_total: Evaluations featuring at least one known candidate.
    :vartype known_total: Optional[int]
    :ivar known_availability: Fraction of evaluations containing a known candidate.
    :vartype known_availability: Optional[float]
    :ivar avg_probability: Mean probability recorded for known predictions.
    :vartype avg_probability: Optional[float]
    :ivar dataset: Dataset identifier.
    :vartype dataset: Optional[str]
    :ivar issue: Issue key under evaluation.
    :vartype issue: Optional[str]
    :ivar issue_label: Human-readable issue label.
    :vartype issue_label: Optional[str]
    :ivar study_label: Human-readable study label.
    :vartype study_label: Optional[str]
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

    :ivar mae_after: Mean absolute error achieved by the regressor.
    :vartype mae_after: Optional[float]
    :ivar rmse_after: Root mean squared error achieved by the regressor.
    :vartype rmse_after: Optional[float]
    :ivar r2_after: Coefficient of determination achieved by the regressor.
    :vartype r2_after: Optional[float]
    :ivar baseline_mae: Baseline MAE using the no-change predictor.
    :vartype baseline_mae: Optional[float]
    :ivar mae_delta: Absolute MAE improvement over the baseline.
    :vartype mae_delta: Optional[float]
    :ivar accuracy_after: Directional accuracy achieved by the regressor.
    :vartype accuracy_after: Optional[float]
    :ivar baseline_accuracy: Directional accuracy achieved by the baseline.
    :vartype baseline_accuracy: Optional[float]
    :ivar accuracy_delta: Improvement in directional accuracy over the baseline.
    :vartype accuracy_delta: Optional[float]
    :ivar participants: Number of participants in the evaluation split.
    :vartype participants: Optional[int]
    :ivar eligible: Number of evaluation examples contributing to accuracy metrics.
    :vartype eligible: Optional[int]
    :ivar dataset: Dataset identifier.
    :vartype dataset: Optional[str]
    :ivar split: Evaluation split name.
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
