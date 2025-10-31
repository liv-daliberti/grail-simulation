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
from typing import Optional


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

