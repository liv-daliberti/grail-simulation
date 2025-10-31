#!/usr/bin/env python
"""Sweep outcomes and selections for the KNN pipeline.

Split from ``context.py`` to simplify that module and improve linting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from common.pipeline.types import (
    BasePipelineSweepOutcome,
    StudySelection as BaseStudySelection,
    narrow_opinion_selection,
)
from common.pipeline.types import StudySpec
from common.opinion.sweep_types import BaseOpinionSweepOutcome

# Forward imports to avoid circulars
from .context_config import SweepConfig


@dataclass
class _KnnSweepStats:
    """Grouped KNN slate metrics added on top of the base outcome."""

    feature_space: str
    accuracy: float
    best_k: int
    eligible: int


class SweepOutcome:
    """Persisted metrics for evaluating a configuration against a single study.

    Composition-based wrapper around
    :class:`common.pipeline.types.BasePipelineSweepOutcome` enriched with
    feature-space metadata and KNN-specific statistics.

    :param base: Base sweep outcome containing common metadata and metrics.
    :type base: ~common.pipeline.types.BasePipelineSweepOutcome
    :param knn: Extra KNN-specific metrics captured for the run.
    :type knn: _KnnSweepStats
    """

    def __init__(
        self,
        *,
        base: BasePipelineSweepOutcome[SweepConfig] | None = None,
        knn: _KnnSweepStats | None = None,
        # Legacy flat kwargs for backwards compatibility (used in tests)
        order_index: int | None = None,
        study: StudySpec | None = None,
        config: "SweepConfig" | None = None,
        metrics_path: Path | None = None,
        metrics: Mapping[str, object] | None = None,
        feature_space: str | None = None,
        accuracy: float | None = None,
        best_k: int | None = None,
        eligible: int | None = None,
    ) -> None:
        """Construct a KNN sweep outcome.

        Accepts either a composed ``base`` + ``knn`` pair, or a legacy flat set
        of keyword arguments (``order_index``, ``study``, ``config``,
        ``metrics_path``, ``metrics``, ``feature_space``, ``accuracy``,
        ``best_k``, ``eligible``).
        """
        if base is None or knn is None:
            # Build from legacy flat kwargs
            assert order_index is not None
            assert study is not None
            assert config is not None
            assert metrics_path is not None
            assert metrics is not None
            assert feature_space is not None
            assert accuracy is not None
            assert best_k is not None
            assert eligible is not None
            base = BasePipelineSweepOutcome[SweepConfig](
                order_index=order_index,
                study=study,
                config=config,
                metrics_path=metrics_path,
                metrics=metrics,
            )
            knn = _KnnSweepStats(
                feature_space=feature_space,
                accuracy=accuracy,
                best_k=best_k,
                eligible=eligible,
            )
        self._base = base
        self._knn = knn

    # Base-forwarded properties
    @property
    def order_index(self) -> int:  # pragma: no cover - simple forwarding
        """Return the stable ordering index for this outcome.

        :returns: Zero-based submission order index.
        :rtype: int
        """
        return self._base.order_index

    @property
    def study(self) -> StudySpec:  # pragma: no cover - simple forwarding
        """Return the study metadata associated with the outcome.

        :returns: Study specification used for the evaluation.
        :rtype: ~common.pipeline.types.StudySpec
        """
        return self._base.study

    @property
    def config(self) -> SweepConfig:  # pragma: no cover - simple forwarding
        """Return the configuration evaluated for the outcome.

        :returns: Pipeline configuration that produced the metrics.
        :rtype: ~knn.pipeline.context.SweepConfig
        """
        return self._base.config

    @property
    def metrics_path(self) -> Path:  # pragma: no cover - simple forwarding
        """Return filesystem path to the persisted metrics artefact.

        :returns: Path to ``metrics.json`` for this outcome.
        :rtype: pathlib.Path
        """
        return self._base.metrics_path

    @property
    def metrics(self) -> Mapping[str, object]:  # pragma: no cover - simple forwarding
        """Return raw metrics payload loaded from disk.

        :returns: Mapping captured by the evaluation stage.
        :rtype: Mapping[str, object]
        """
        return self._base.metrics

    @property
    def feature_space(self) -> str:  # pragma: no cover - simple forwarding
        """Return the feature space label used for this evaluation.

        :returns: Feature space identifier (e.g., ``"tfidf"``).
        :rtype: str
        """
        return self._knn.feature_space

    @property
    def accuracy(self) -> float:  # pragma: no cover - simple forwarding
        """Return the held-out accuracy achieved by the configuration.

        :returns: Accuracy on the evaluation split.
        :rtype: float
        """
        return self._knn.accuracy

    @property
    def best_k(self) -> int:  # pragma: no cover - simple forwarding
        """Return the selected ``k`` for the KNN model.

        :returns: Best-performing neighbor count.
        :rtype: int
        """
        return self._knn.best_k

    @property
    def eligible(self) -> int:  # pragma: no cover - simple forwarding
        """Return the number of eligible examples considered.

        :returns: Count of examples included in metric computation.
        :rtype: int
        """
        return self._knn.eligible


@dataclass
class _KnnOpinionExtras:
    """Grouped opinion metrics added by the KNN pipeline."""

    feature_space: str
    r2_score: float
    baseline_mae: Optional[float]
    mae_delta: Optional[float]
    best_k: int
    participants: int


class OpinionSweepOutcome:
    """Opinion regression metrics for a (study, configuration) evaluation.

    Composition-based wrapper around
    :class:`common.opinion.sweep_types.BaseOpinionSweepOutcome` enriched with
    KNN-specific opinion metrics.

    :param base: Base opinion outcome carrying common metrics and artefacts.
    :type base: ~common.opinion.sweep_types.BaseOpinionSweepOutcome
    :param knn: Extra KNN opinion metrics captured for the run.
    :type knn: _KnnOpinionExtras
    """

    def __init__(
        self,
        *,
        base: BaseOpinionSweepOutcome[SweepConfig] | None = None,
        knn: _KnnOpinionExtras | None = None,
        # Legacy flat kwargs for backwards compatibility
        order_index: int | None = None,
        study: StudySpec | None = None,
        config: "SweepConfig" | None = None,
        mae: float | None = None,
        rmse: float | None = None,
        artifact: object | None = None,
        accuracy_summary: object | None = None,
        feature_space: str | None = None,
        r2_score: float | None = None,
        baseline_mae: float | None = None,
        mae_delta: float | None = None,
        best_k: int | None = None,
        participants: int | None = None,
    ) -> None:
        """Construct an opinion sweep outcome.

        Accepts composed ``base`` + ``knn`` or legacy flat kwargs used by tests.
        """
        if base is None or knn is None:
            # Build the composed objects from flat kwargs
            assert order_index is not None
            assert study is not None
            assert config is not None
            assert mae is not None
            assert rmse is not None
            assert artifact is not None
            assert accuracy_summary is not None
            assert feature_space is not None
            assert r2_score is not None
            assert best_k is not None
            assert participants is not None
            base = BaseOpinionSweepOutcome[SweepConfig](
                order_index=order_index,
                study=study,
                config=config,
                mae=mae,
                rmse=rmse,
                artifact=artifact,  # type: ignore[arg-type]
                accuracy_summary=accuracy_summary,  # type: ignore[arg-type]
            )
            knn = _KnnOpinionExtras(
                feature_space=feature_space,
                r2_score=r2_score,
                baseline_mae=baseline_mae,
                mae_delta=mae_delta,
                best_k=best_k,
                participants=participants,
            )
        self._base = base
        self._knn = knn

    # Base-forwarded properties
    @property
    def order_index(self) -> int:  # pragma: no cover - simple forwarding
        """Return the stable ordering index for this outcome."""
        return self._base.order_index

    @property
    def study(self) -> StudySpec:  # pragma: no cover - simple forwarding
        """Return the study metadata associated with the outcome."""
        return self._base.study

    @property
    def config(self) -> SweepConfig:  # pragma: no cover - simple forwarding
        """Return the configuration evaluated for the outcome."""
        return self._base.config

    @property
    def mae(self) -> float:  # pragma: no cover - simple forwarding
        """Mean absolute error achieved by the configuration."""
        return self._base.mae

    @property
    def rmse(self) -> float:  # pragma: no cover - simple forwarding
        """Root mean squared error achieved by the configuration."""
        return self._base.rmse

    @property
    def artifact(self):  # pragma: no cover - simple forwarding
        """Return the metrics artefact wrapper for this outcome."""
        return self._base.artifact

    @property
    def metrics_path(self) -> Path:  # pragma: no cover - simple forwarding
        """Filesystem path to the persisted metrics JSON."""
        return self._base.metrics_path

    @property
    def metrics(self) -> Mapping[str, object]:  # pragma: no cover - simple forwarding
        """Raw metrics payload loaded from disk."""
        return self._base.metrics

    @property
    def accuracy(self) -> Optional[float]:  # pragma: no cover - simple forwarding
        """Directional accuracy achieved by the configuration."""
        return self._base.accuracy

    @property
    def baseline_accuracy(self) -> Optional[float]:  # pragma: no cover - simple forwarding
        """Baseline directional accuracy for comparison."""
        return self._base.baseline_accuracy

    @property
    def accuracy_delta(self) -> Optional[float]:  # pragma: no cover - simple forwarding
        """Improvement over baseline directional accuracy."""
        return self._base.accuracy_delta

    @property
    def eligible(self) -> Optional[int]:  # pragma: no cover - simple forwarding
        """Number of examples contributing to the accuracy figures."""
        return self._base.eligible

    @property
    def feature_space(self) -> str:  # pragma: no cover - simple forwarding
        """Return the feature space label used for opinion evaluation.

        :returns: Feature space identifier (e.g., ``"word2vec"``).
        :rtype: str
        """
        return self._knn.feature_space

    @property
    def r2_score(self) -> float:  # pragma: no cover - simple forwarding
        """Return the coefficient of determination (R²).

        :returns: R² score of the regression.
        :rtype: float
        """
        return self._knn.r2_score

    @property
    def baseline_mae(self) -> Optional[float]:  # pragma: no cover - forwarding
        """Return the baseline MAE used for comparison, if available.

        :returns: Baseline mean absolute error or ``None`` if unknown.
        :rtype: float | None
        """
        return self._knn.baseline_mae

    @property
    def mae_delta(self) -> Optional[float]:  # pragma: no cover - forwarding
        """Return the improvement over baseline MAE (baseline - MAE).

        :returns: MAE delta or ``None`` when baseline is unavailable.
        :rtype: float | None
        """
        return self._knn.mae_delta

    @property
    def best_k(self) -> int:  # pragma: no cover - simple forwarding
        """Return the selected ``k`` for the KNN regressor.

        :returns: Best-performing neighbor count.
        :rtype: int
        """
        return self._knn.best_k

    @property
    def participants(self) -> int:  # pragma: no cover - simple forwarding
        """Return the number of participants included in the study.

        :returns: Participant count used for the opinion metrics.
        :rtype: int
        """
        return self._knn.participants


class StudySelection(BaseStudySelection[SweepOutcome]):
    """Selected configuration for a specific study within a feature space."""

    @property
    def accuracy(self) -> float:
        """Return the held-out accuracy achieved by the selection."""
        return self.outcome.accuracy

    @property
    def best_k(self) -> int:
        """Return the selected ``k`` for the study.

        Several reporting and evaluation helpers access ``selection.best_k``;
        surface it here as a thin proxy to the underlying outcome.
        """
        return int(self.outcome.best_k)


OpinionSelectionBase = narrow_opinion_selection(OpinionSweepOutcome)


class OpinionStudySelection(OpinionSelectionBase):
    """Selected configuration for the final opinion evaluation."""

    @property
    def best_k(self) -> int:
        """Return the selected ``k`` for the study."""
        return int(self.outcome.best_k)
