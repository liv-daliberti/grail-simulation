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

"""Neighbourhood selection utilities for the KNN evaluator."""

from __future__ import annotations

from typing import Dict, List, Sequence


def parse_k_values(k_default: int, sweep: str) -> List[int]:
    """
    Derive the sorted set of ``k`` values requested for evaluation.

    :param k_default: Baseline ``k`` value used when the sweep is empty.
    :param sweep: Comma-delimited string of additional ``k`` candidates.
    :returns: Strictly positive ``k`` values in ascending order. Falls back to
        ``k_default`` (or ``25`` when unset) if no valid integers are supplied.
    """

    values = {int(k_default)} if k_default else set()
    for token in sweep.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError:
            continue
    k_vals = sorted(k for k in values if k > 0)
    return k_vals or [int(k_default) if k_default else 25]


def select_best_k(
    k_values: Sequence[int],
    accuracy_by_k: Dict[int, float],
    *,
    method: str = "elbow",
) -> int:
    """
    Select ``k`` using either max-accuracy or an elbow heuristic.

    :param k_values: Sorted sequence of evaluated ``k`` values.
    :param accuracy_by_k: Observed eligible-only accuracy for each ``k`` on the
        validation split.
    :param method: Selection strategy (default: ``"elbow"``). ``"max"`` picks
        the accuracy-maximising ``k``. ``"elbow"`` picks the first ``k`` where
        marginal gains fall below half the initial slope, falling back to
        max-accuracy when the heuristic is not applicable.
    :returns: Selected ``k`` per the requested method.
    """

    method_norm = (method or "max").strip().lower()
    if method_norm not in {"max", "elbow"}:
        method_norm = "max"

    if method_norm == "max":
        return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))

    if len(k_values) <= 2:
        return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))
    accuracies = [accuracy_by_k.get(k, 0.0) for k in k_values]
    slopes: List[float] = []
    for idx in range(1, len(k_values)):
        delta_acc = accuracies[idx] - accuracies[idx - 1]
        delta_k = k_values[idx] - k_values[idx - 1]
        slopes.append(delta_acc / delta_k if delta_k else 0.0)
    if not slopes:
        return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))
    first_slope = slopes[0]
    threshold = max(first_slope * 0.5, 0.001)
    for idx, slope in enumerate(slopes[1:], start=1):
        if slope <= threshold:
            return k_values[idx]
    return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))


__all__ = ["parse_k_values", "select_best_k"]
