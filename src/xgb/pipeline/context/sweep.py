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

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

from common.pipeline.types import BasePipelineSweepOutcome, StudySelection as BaseStudySelection
from common.opinion.sweep_types import BaseSweepTask

from ...core.model import XGBoostBoosterParams


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
    """Metrics captured for a (study, configuration) sweep evaluation."""

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
