#!/usr/bin/env python
"""Summary data structures shared across the KNN pipeline.

Contains normalised slate and opinion summary views used by reports and
selection logic. Split out from ``context.py`` to reduce module size and
improve Sphinx autodoc clarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

from common.opinion import OpinionCalibrationMetrics


@dataclass(frozen=True)
class _AccuracyStats:
    value: Optional[float] = None
    ci95: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class _Counts:
    total: Optional[int] = None
    eligible: Optional[int] = None


class MetricSummary:
    """
    Normalised slice of slate evaluation metrics used across reports.

    :param accuracy: Validation accuracy for the selected configuration.
    :type accuracy: Optional[float]
    :param accuracy_ci: 95% confidence interval for :attr:`accuracy`.
    :type accuracy_ci: Optional[Tuple[float, float]]
    :param baseline: Baseline accuracy from the most-frequent-gold comparator.
    :type baseline: Optional[float]
    :param baseline_ci: 95% confidence interval for :attr:`baseline`.
    :type baseline_ci: Optional[Tuple[float, float]]
    :param random_baseline: Expected accuracy for a random slate selection baseline.
    :type random_baseline: Optional[float]
    :param best_k: ``k`` value delivering the best validation accuracy.
    :type best_k: Optional[int]
    :param n_total: Total number of evaluation rows considered.
    :type n_total: Optional[int]
    :param n_eligible: Number of rows eligible for the final metric.
    :type n_eligible: Optional[int]
    :param accuracy_all_rows: Accuracy over all rows (eligible and ineligible).
    :type accuracy_all_rows: Optional[float]
    """

    # Hints for static analyzers (attributes are assigned via __setattr__).
    _acc: _AccuracyStats
    _baseline: _AccuracyStats
    _counts: _Counts
    best_k: Optional[int]
    random_baseline: Optional[float]
    accuracy_all_rows: Optional[float]

    def __init__(self, *, inputs: Mapping[str, object]) -> None:
        """
        Construct a metric summary from a typed mapping.

        Expected keys in ``inputs`` (all optional):
        ``accuracy`` (float), ``accuracy_ci`` (tuple[float, float]),
        ``baseline`` (float), ``baseline_ci`` (tuple[float, float]),
        ``random_baseline`` (float), ``best_k`` (int),
        ``n_total`` (int), ``n_eligible`` (int), ``accuracy_all_rows`` (float).
        """
        acc_val = inputs.get("accuracy")  # type: ignore[assignment]
        acc_ci = inputs.get("accuracy_ci")  # type: ignore[assignment]
        base_val = inputs.get("baseline")  # type: ignore[assignment]
        base_ci = inputs.get("baseline_ci")  # type: ignore[assignment]
        object.__setattr__(
            self, "_acc", _AccuracyStats(value=acc_val if isinstance(acc_val, (int, float)) else acc_val, ci95=acc_ci if isinstance(acc_ci, tuple) else acc_ci)  # type: ignore[arg-type]
        )
        object.__setattr__(
            self, "_baseline", _AccuracyStats(value=base_val if isinstance(base_val, (int, float)) else base_val, ci95=base_ci if isinstance(base_ci, tuple) else base_ci)  # type: ignore[arg-type]
        )
        object.__setattr__(
            self,
            "random_baseline",
            inputs.get("random_baseline"),  # type: ignore[assignment]
        )
        object.__setattr__(self, "best_k", inputs.get("best_k"))  # type: ignore[assignment]
        object.__setattr__(
            self,
            "_counts",
            _Counts(
                total=inputs.get("n_total"),  # type: ignore[arg-type]
                eligible=inputs.get("n_eligible"),  # type: ignore[arg-type]
            ),
        )
        object.__setattr__(
            self,
            "accuracy_all_rows",
            inputs.get("accuracy_all_rows"),  # type: ignore[assignment]
        )

    # Backwards-compatible accessors
    @property
    def accuracy(self) -> Optional[float]:  # pragma: no cover - simple forwarding
        """Validation accuracy for the selected configuration.

        :rtype: Optional[float]
        """
        return self._acc.value

    @property
    def accuracy_ci(self) -> Optional[Tuple[float, float]]:  # pragma: no cover
        """95% confidence interval for :attr:`accuracy`.

        :rtype: Optional[Tuple[float, float]]
        """
        return self._acc.ci95

    @property
    def baseline(self) -> Optional[float]:  # pragma: no cover
        """Baseline accuracy from a standard comparator.

        :rtype: Optional[float]
        """
        return self._baseline.value

    @property
    def baseline_ci(self) -> Optional[Tuple[float, float]]:  # pragma: no cover
        """95% confidence interval for :attr:`baseline`.

        :rtype: Optional[Tuple[float, float]]
        """
        return self._baseline.ci95

    @property
    def n_total(self) -> Optional[int]:  # pragma: no cover
        """Total number of evaluation rows considered.

        :rtype: Optional[int]
        """
        return self._counts.total

    @property
    def n_eligible(self) -> Optional[int]:  # pragma: no cover
        """Number of rows eligible for the final metric.

        :rtype: Optional[int]
        """
        return self._counts.eligible


class OpinionSummary(OpinionCalibrationMetrics):
    """
    Normalised view of opinion-regression metrics.


    :param mae: Mean absolute error for the selected configuration.
    :type mae: Optional[float]
    :param rmse: Root-mean-square error for the selected configuration.
    :type rmse: Optional[float]
    :param r2_score: Coefficient of determination capturing explained variance.
    :type r2_score: Optional[float]
    :param mae_change: Normalised change in MAE relative to the baseline.
    :type mae_change: Optional[float]
    :param rmse_change: Root-mean-square error on the opinion-change signal.
    :type rmse_change: Optional[float]
    :param baseline_mae: Baseline MAE measured using pre-study opinions.
    :type baseline_mae: Optional[float]
    :param baseline_rmse_change: Baseline RMSE on the opinion-change signal.
    :type baseline_rmse_change: Optional[float]
    :param mae_delta: Absolute delta between :attr:`mae` and :attr:`baseline_mae`.
    :type mae_delta: Optional[float]
    :param accuracy: Directional accuracy comparing predicted opinion shifts.
    :type accuracy: Optional[float]
    :param calibration_slope: Calibration slope between predicted and actual opinion deltas.
    :type calibration_slope: Optional[float]
    :param calibration_intercept: Calibration intercept between predicted and actual opinion deltas.
    :type calibration_intercept: Optional[float]
    :param calibration_ece: Expected calibration error computed over opinion-change bins.
    :type calibration_ece: Optional[float]
    :param kl_divergence_change: KL divergence between predicted and actual change distributions.
    :type kl_divergence_change: Optional[float]
    :param calibration_bins: Optional tuple of bin summaries backing
        :attr:`~common.opinion.OpinionCalibrationMetrics.calibration_ece`.
    :type calibration_bins: Optional[Tuple[Mapping[str, float], ...]]
    :param best_k: Neighbourhood size delivering the final metrics.
    :type best_k: Optional[int]
    :param participants: Number of participants included in the evaluation split.
    :type participants: Optional[int]
    :param eligible: Count of evaluation examples used to compute accuracy metrics.
    :type eligible: Optional[int]
    :param dataset: Name of the dataset used to compute the metrics.
    :type dataset: Optional[str]
    :param split: Dataset split powering the evaluation (e.g. ``train``, ``validation``).
    :type split: Optional[str]
    :note: Baseline and calibration deltas are documented in
        :class:`~common.opinion.OpinionCalibrationMetrics`.
    """

    # Hints for static analyzers; members are assigned in __init__.
    _regression: "OpinionSummary.Regression"
    _accuracy: Optional[float]
    _calibration_bins: Optional[Tuple[Mapping[str, float], ...]]

    @dataclass(frozen=True)
    class Inputs:
        """Grouped construction inputs reducing local variables in __init__."""

        calibration: OpinionCalibrationMetrics = field(
            default_factory=OpinionCalibrationMetrics
        )
        regression: "OpinionSummary.Regression" = field(
            default_factory=lambda: OpinionSummary.Regression()
        )
        accuracy: Optional[float] = None
        calibration_bins: Optional[Tuple[Mapping[str, float], ...]] = None

    @dataclass(frozen=True)
    class PrimaryRegression:
        """Primary regression metrics for opinion models."""

        mae: Optional[float] = None
        rmse: Optional[float] = None
        r2_score: Optional[float] = None

    @dataclass(frozen=True)
    class ChangeStats:
        """Opinion-change signal metrics, including baseline reference."""

        mae_change: Optional[float] = None
        rmse_change: Optional[float] = None
        baseline_rmse_change: Optional[float] = None

    @dataclass(frozen=True)
    class BaselineStats:
        """Baseline opinion regression metrics and deltas."""

        baseline_mae: Optional[float] = None
        mae_delta: Optional[float] = None

    @dataclass(frozen=True)
    class Regression:
        """Grouped opinion-regression metrics, reducing attribute count."""

        primary: "OpinionSummary.PrimaryRegression" = field(
            default_factory=lambda: OpinionSummary.PrimaryRegression()
        )
        change: "OpinionSummary.ChangeStats" = field(
            default_factory=lambda: OpinionSummary.ChangeStats()
        )
        baseline: "OpinionSummary.BaselineStats" = field(
            default_factory=lambda: OpinionSummary.BaselineStats()
        )
        best_k: Optional[int] = None

    def __init__(self, *, inputs: "OpinionSummary.Inputs") -> None:
        # initialise base dataclass using grouped calibration inputs
        super().__init__(
            baseline_accuracy=inputs.calibration.baseline_accuracy,
            accuracy_delta=inputs.calibration.accuracy_delta,
            calibration_slope=inputs.calibration.calibration_slope,
            baseline_calibration_slope=inputs.calibration.baseline_calibration_slope,
            calibration_intercept=inputs.calibration.calibration_intercept,
            baseline_calibration_intercept=inputs.calibration.baseline_calibration_intercept,
            calibration_ece=inputs.calibration.calibration_ece,
            baseline_calibration_ece=inputs.calibration.baseline_calibration_ece,
            kl_divergence_change=inputs.calibration.kl_divergence_change,
            baseline_kl_divergence_change=inputs.calibration.baseline_kl_divergence_change,
            participants=inputs.calibration.participants,
            eligible=inputs.calibration.eligible,
            dataset=inputs.calibration.dataset,
            split=inputs.calibration.split,
        )
        # set opinion-specific fields
        object.__setattr__(self, "_regression", inputs.regression)
        object.__setattr__(self, "_accuracy", inputs.accuracy)
        object.__setattr__(self, "_calibration_bins", inputs.calibration_bins)

    # Backwards-compatible accessors
    @property
    def mae(self) -> Optional[float]:  # pragma: no cover - simple forwarding
        """Mean absolute error of the selected configuration.

        :rtype: Optional[float]
        """
        return self._regression.primary.mae

    @property
    def rmse(self) -> Optional[float]:  # pragma: no cover
        """Root-mean-square error of the selected configuration.

        :rtype: Optional[float]
        """
        return self._regression.primary.rmse

    @property
    def r2_score(self) -> Optional[float]:  # pragma: no cover
        """Coefficient of determination (R²) for the selected configuration.

        :rtype: Optional[float]
        """
        return self._regression.primary.r2_score

    @property
    def mae_change(self) -> Optional[float]:  # pragma: no cover
        """MAE on the opinion-change signal.

        :rtype: Optional[float]
        """
        return self._regression.change.mae_change

    @property
    def rmse_change(self) -> Optional[float]:  # pragma: no cover
        """RMSE on the opinion-change signal.

        :rtype: Optional[float]
        """
        return self._regression.change.rmse_change

    @property
    def baseline_mae(self) -> Optional[float]:  # pragma: no cover
        """Baseline MAE measured with pre-study opinions.

        :rtype: Optional[float]
        """
        return self._regression.baseline.baseline_mae

    @property
    def baseline_rmse_change(self) -> Optional[float]:  # pragma: no cover
        """Baseline RMSE on the opinion-change signal.

        :rtype: Optional[float]
        """
        return self._regression.change.baseline_rmse_change

    @property
    def mae_delta(self) -> Optional[float]:  # pragma: no cover
        """Delta between :attr:`mae` and :attr:`baseline_mae`.

        :rtype: Optional[float]
        """
        return self._regression.baseline.mae_delta

    @property
    def best_k(self) -> Optional[int]:  # pragma: no cover
        """Neighbourhood size delivering the final metrics.

        :rtype: Optional[int]
        """
        return self._regression.best_k

    @property
    def accuracy(self) -> Optional[float]:  # pragma: no cover
        """Directional accuracy comparing predicted opinion shifts."""
        return self._accuracy

    @property
    def calibration_bins(self) -> Optional[Tuple[Mapping[str, float], ...]]:  # pragma: no cover
        """Optional underlying calibration bin summaries (ECE provenance)."""
        return self._calibration_bins
