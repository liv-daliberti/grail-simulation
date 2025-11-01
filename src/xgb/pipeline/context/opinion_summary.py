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

"""Opinion summary structures with legacy attribute compatibility.

The :class:`~xgb.pipeline.context.OpinionSummary` dataclass groups related metrics into logical
sub-structures and implements compatibility access for legacy flat field
names via a small forwarding map. A convenience constructor builds an
instance from keyword arguments using those flat names.
"""

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
    """Grouped opinion-regression metrics with compatibility accessors.

    To keep the public surface stable without a long list of forwarding
    properties, attribute access for the legacy flat names is implemented
    via ``__getattr__`` using a small lookup table.
    """

    __module__ = "xgb.pipeline.context"

    after: _OpinionAfter
    baseline: _OpinionBaseline
    calibration: _OpinionCalibration
    deltas: _OpinionDeltas
    meta: _OpinionMeta

    # Map legacy flat attribute names to the grouped fields
    _FORWARD_MAP = {
        "mae_after": ("after", "mae_after"),
        "mae_change": ("after", "mae_change"),
        "rmse_after": ("after", "rmse_after"),
        "r2_after": ("after", "r2_after"),
        "rmse_change": ("after", "rmse_change"),
        "accuracy_after": ("after", "accuracy_after"),
        "baseline_mae": ("baseline", "baseline_mae"),
        "baseline_rmse_change": ("baseline", "baseline_rmse_change"),
        "baseline_accuracy": ("baseline", "baseline_accuracy"),
        "calibration_slope": ("calibration", "calibration_slope"),
        "baseline_calibration_slope": ("baseline", "baseline_calibration_slope"),
        "calibration_intercept": ("calibration", "calibration_intercept"),
        "baseline_calibration_intercept": (
            "baseline",
            "baseline_calibration_intercept",
        ),
        "calibration_ece": ("calibration", "calibration_ece"),
        "baseline_calibration_ece": ("baseline", "baseline_calibration_ece"),
        "kl_divergence_change": ("calibration", "kl_divergence_change"),
        "baseline_kl_divergence_change": (
            "baseline",
            "baseline_kl_divergence_change",
        ),
        "participants": ("meta", "participants"),
        "eligible": ("meta", "eligible"),
        "dataset": ("meta", "dataset"),
        "split": ("meta", "split"),
        "label": ("meta", "label"),
        "mae_delta": ("deltas", "mae_delta"),
        "accuracy_delta": ("deltas", "accuracy_delta"),
    }

    def __getattr__(self, name: str):  # pragma: no cover - simple forwarding
        try:
            group, attr = self._FORWARD_MAP[name]
        except KeyError as exc:  # maintain normal AttributeError semantics
            raise AttributeError(name) from exc
        return getattr(getattr(self, group), attr)

    def __dir__(self):  # pragma: no cover - aids introspection only
        # Include both dataclass fields and the flattened legacy names
        base = set(super().__dir__())
        base.update(self._FORWARD_MAP.keys())
        return sorted(base)

    @classmethod
    def from_kwargs(cls, **kwargs) -> "OpinionSummary":
        """
        Construct a grouped summary from flat keyword arguments.

        :param kwargs: Flat fields with keys such as ``mae_after``, ``baseline_mae``,
            ``calibration_slope``, ``kl_divergence_change``, and metadata like
            ``participants`` and ``label``.
        :type kwargs: dict
        :returns: Grouped :class:`~xgb.pipeline.context.OpinionSummary` instance.
        :rtype: ~xgb.pipeline.context.OpinionSummary
        """

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
