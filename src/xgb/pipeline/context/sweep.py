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

"""XGBoost sweep dataclasses and helpers.

This module defines the configuration, task, outcome, and run-context
structures used to perform hyper-parameter sweeps for the XGBoost-based
text classification pipeline. It also provides utilities to convert
configurations into booster parameters and CLI argument lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

from common.pipeline.types import BasePipelineSweepOutcome, StudySelection as BaseStudySelection
from common.opinion.sweep_helpers import ExtrasSweepTask

from ...core.model import XGBoostBoosterParams


@dataclass(frozen=True)
class BoosterParams:
    """Booster hyper-parameters explored during sweeps."""

    __module__ = "xgb.pipeline.context"

    learning_rate: float
    max_depth: int
    n_estimators: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float

    def as_xgb(self, tree_method: str) -> XGBoostBoosterParams:
        """Convert to :class:`~xgb.core.model_config.XGBoostBoosterParams`.

        :param tree_method: Tree construction algorithm passed to XGBoost.
        :type tree_method: str
        :returns: Booster parameters compatible with the training CLI.
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

    __module__ = "xgb.pipeline.context"

    text_vectorizer: str
    vectorizer_tag: str
    booster: BoosterParams
    vectorizer_cli: Tuple[str, ...] = field(default_factory=tuple)

    # Backwards-compatible constructor accepting flat booster kwargs
    def __init__(
        self,
        text_vectorizer: str,
        vectorizer_tag: str,
        booster: BoosterParams | None = None,
        *,
        learning_rate: float | None = None,
        max_depth: int | None = None,
        n_estimators: int | None = None,
        subsample: float | None = None,
        colsample_bytree: float | None = None,
        reg_lambda: float | None = None,
        reg_alpha: float | None = None,
        vectorizer_cli: Tuple[str, ...] = (),
    ) -> None:
        object.__setattr__(self, "text_vectorizer", text_vectorizer)
        object.__setattr__(self, "vectorizer_tag", vectorizer_tag)
        if booster is None:
            # Construct booster from legacy flat arguments
            if None in (
                learning_rate,
                max_depth,
                n_estimators,
                subsample,
                colsample_bytree,
                reg_lambda,
                reg_alpha,
            ):
                raise TypeError(
                    "SweepConfig requires either a BoosterParams "
                    "instance or all flat booster kwargs"
                )
            booster = BoosterParams(
                learning_rate=float(learning_rate),
                max_depth=int(max_depth),
                n_estimators=int(n_estimators),
                subsample=float(subsample),
                colsample_bytree=float(colsample_bytree),
                reg_lambda=float(reg_lambda),
                reg_alpha=float(reg_alpha),
            )
        object.__setattr__(self, "booster", booster)
        object.__setattr__(self, "vectorizer_cli", tuple(vectorizer_cli))

    def label(self) -> str:
        """
        Produce a filesystem- and report-friendly identifier.

        :param self: Sweep configuration instance being labelled.
        :type self: ~xgb.pipeline.context.SweepConfig
        :returns: Composite label encoding vectoriser and booster parameters.
        :rtype: str
        """

        tag = self.vectorizer_tag or self.text_vectorizer
        booster = self.booster
        base = (
            f"lr{booster.learning_rate:g}_depth{booster.max_depth}_"
            f"estim{booster.n_estimators}_sub{booster.subsample:g}_"
            f"col{booster.colsample_bytree:g}_l2{booster.reg_lambda:g}_l1{booster.reg_alpha:g}"
        ).replace(".", "p")
        return f"{tag}_{base}"

    def booster_params(self, tree_method: str) -> XGBoostBoosterParams:
        """
        Convert the sweep configuration into :class:`~xgb.core.model_config.XGBoostBoosterParams`.

        :param tree_method: Tree construction algorithm passed to XGBoost.
        :type tree_method: str
        :returns: Booster parameter bundle mirroring this configuration.
        :rtype: XGBoostBoosterParams
        """

        return self.booster.as_xgb(tree_method)

    def cli_args(self, tree_method: str | None) -> List[str]:
        """
        Serialise the configuration into CLI arguments for :mod:`xgb.cli`.

        :param tree_method: Tree construction algorithm passed to XGBoost.
        :type tree_method: str
        :returns: List of CLI flags encoding the configuration.
        :rtype: List[str]
        """

        args: List[str] = [
            "--text_vectorizer",
            self.text_vectorizer,
            "--xgb_learning_rate",
            str(self.booster.learning_rate),
            "--xgb_max_depth",
            str(self.booster.max_depth),
            "--xgb_n_estimators",
            str(self.booster.n_estimators),
            "--xgb_subsample",
            str(self.booster.subsample),
            "--xgb_colsample_bytree",
            str(self.booster.colsample_bytree),
            "--xgb_reg_lambda",
            str(self.booster.reg_lambda),
            "--xgb_reg_alpha",
            str(self.booster.reg_alpha),
        ]
        if tree_method:
            args.extend(["--xgb_tree_method", tree_method])
        args.extend(list(self.vectorizer_cli))
        return args

    # Backwards-compatible attribute accessors
    @property
    def learning_rate(self) -> float:
        """Learning rate as a convenience proxy to ``booster.learning_rate``."""
        return self.booster.learning_rate

    @property
    def max_depth(self) -> int:
        """Maximum tree depth mirroring ``booster.max_depth``."""
        return self.booster.max_depth

    @property
    def n_estimators(self) -> int:
        """Number of boosting rounds from ``booster.n_estimators``."""
        return self.booster.n_estimators

    @property
    def subsample(self) -> float:
        """Row subsampling ratio taken from ``booster.subsample``."""
        return self.booster.subsample

    @property
    def colsample_bytree(self) -> float:
        """Column subsampling per tree taken from ``booster.colsample_bytree``."""
        return self.booster.colsample_bytree

    @property
    def reg_lambda(self) -> float:
        """L2 regularisation weight from ``booster.reg_lambda``."""
        return self.booster.reg_lambda

    @property
    def reg_alpha(self) -> float:
        """L1 regularisation weight from ``booster.reg_alpha``."""
        return self.booster.reg_alpha


@dataclass
class SweepOutcome(BasePipelineSweepOutcome["xgb.pipeline.context.SweepConfig"]):
    """Metrics captured for a (study, configuration) sweep evaluation.

    The measured fields are exposed as properties backed by the raw
    metrics mapping to keep the instance attribute count small.
    """

    __module__ = "xgb.pipeline.context"

    def __init__(
        self,
        *,
        order_index: int,
        study,
        config: "xgb.pipeline.context.SweepConfig",
        metrics_path: Path | None = None,
        metrics: dict | None = None,
        accuracy: float | None = None,
        coverage: float | None = None,
        evaluated: int | None = None,
    ) -> None:
        merged_metrics: dict = dict(metrics or {})
        if accuracy is not None:
            merged_metrics.setdefault("accuracy", float(accuracy))
        if coverage is not None:
            merged_metrics.setdefault("coverage", float(coverage))
        if evaluated is not None:
            merged_metrics.setdefault("evaluated", int(evaluated))
        # Call base dataclass __init__ to initialise standard fields
        super().__init__(
            order_index=int(order_index),
            study=study,
            config=config,
            metrics_path=(metrics_path or Path("")),
            metrics=merged_metrics,
        )

    @property
    def accuracy(self) -> float:
        """Accuracy measured on eligible rows when available.

        Falls back to overall accuracy if eligible-only metric is absent.
        Returns ``0.0`` if the metric cannot be parsed as a float.
        """
        value = self.metrics.get("accuracy_eligible") or self.metrics.get("accuracy") or 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @property
    def coverage(self) -> float:
        """Coverage (fraction of evaluated rows) as a float.

        Returns ``0.0`` if the metric is missing or invalid.
        """
        try:
            return float(self.metrics.get("coverage", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @property
    def evaluated(self) -> int:
        """Support size, i.e. number of evaluated rows.

        Returns ``0`` if the metric is missing or invalid.
        """
        try:
            return int(self.metrics.get("evaluated", 0))
        except (TypeError, ValueError):
            return 0


@dataclass(frozen=True)
class _SweepTaskExtras:
    """Additional XGBoost-specific task metadata grouped for compactness."""

    tree_method: str


class SweepTask(ExtrasSweepTask["xgb.pipeline.context.SweepConfig"]):
    """Extend :class:`common.opinion.sweep_types.BaseSweepTask` with XGBoost metadata."""

    def __init__(
        self,
        *,
        index: int,
        study,
        config: "xgb.pipeline.context.SweepConfig",
        base_cli: Tuple[str, ...],
        extra_cli: Tuple[str, ...],
        run_root: Path,
        tree_method: str,
        metrics_path: Path,
        train_participant_studies: Tuple[str, ...] = (),
    ) -> None:
        extras = _SweepTaskExtras(tree_method=str(tree_method or "hist"))
        # Use the shared initialiser from ExtrasSweepTask to avoid duplicate
        # forwarding boilerplate and keep logic in one place.
        self._init_shared(
            index=index,
            study=study,
            config=config,
            base_cli=base_cli,
            extra_cli=extra_cli,
            run_root=run_root,
            metrics_path=metrics_path,
            train_participant_studies=train_participant_studies,
            extras=extras,
        )

    _extras: "_SweepTaskExtras"

    @property
    def tree_method(self) -> str:  # pragma: no cover - simple forwarding
        """XGBoost tree construction algorithm used for this task."""
        return self._extras.tree_method

    # train_participant_studies provided by BaseSweepTask

    def as_dict(self) -> dict[str, object]:
        """Summarise task metadata for logging or diagnostics.

        Provides a lightweight, serialisable view exposing the core fields
        needed to understand what will be executed.
        """
        return {
            "index": self.index,
            "study": getattr(self.study, "key", str(self.study)),
            "config_vectorizer": getattr(self.config, "vectorizer_tag", None),
            "tree_method": self.tree_method,
            "run_root": str(self.run_root),
            "metrics_path": str(self.metrics_path),
        }


@dataclass
class StudySelection(BaseStudySelection[SweepOutcome]):
    """
    Selected configuration for the final evaluation of a participant study.

    :param study: Study metadata chosen for final evaluation.
    :type study: ~common.pipeline.types.StudySpec
    :param outcome: Winning sweep outcome leveraged for reporting.
    :type outcome: ~xgb.pipeline.context.SweepOutcome
    """

    __module__ = "xgb.pipeline.context"


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

    __module__ = "xgb.pipeline.context"

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

    __module__ = "xgb.pipeline.context"

    base_cli: Sequence[str]
    extra_cli: Sequence[str]
    out_dir: Path
    tree_method: str
    save_model_dir: Path | None
    reuse_existing: bool
