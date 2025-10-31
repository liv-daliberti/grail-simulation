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

"""Metric helpers and bootstrap utilities for KNN evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from common.evaluation.utils import (
    BootstrapCounts,
    group_key_for_example,
    safe_div,
    summarise_grouped_accuracy_from_counts,
)


def _accuracy_for_rows(rows: Sequence[Mapping[str, Any]], k_val: int) -> float:
    """
    Return accuracy for ``rows`` using predictions at ``k_val``.

    :param rows: Iterable of evaluation records produced by the evaluator.
    :param k_val: ``k`` value whose predictions should be assessed.
    :returns: Accuracy measured over eligible rows at the requested ``k``.
    """

    if not rows:
        return 0.0
    correct = 0
    total = 0
    for row in rows:
        if not row.get("eligible"):
            continue
        total += 1
        pred = row["predictions_by_k"].get(k_val)
        if pred is not None and int(pred) == int(row["gold_index"]):
            correct += 1
    return safe_div(correct, total)


def _baseline_accuracy_for_rows(
    rows: Sequence[Mapping[str, Any]],
    baseline_index: Optional[int],
) -> float:
    """
    Return accuracy for the most frequent baseline over ``rows``.

    :param rows: Iterable of evaluation records produced by the evaluator.
    :param baseline_index: Index representing the baseline recommendation.
    :returns: Accuracy achieved by always predicting ``baseline_index``.
    """

    if baseline_index is None:
        return 0.0
    if not rows:
        return 0.0
    correct = 0
    total = 0
    for row in rows:
        if not row.get("eligible"):
            continue
        total += 1
        if int(row.get("gold_index", -1)) == int(baseline_index):
            correct += 1
    return safe_div(correct, total)


def bootstrap_uncertainty(
    *,
    rows: Sequence[Mapping[str, Any]],
    best_k: int,
    baseline_index: Optional[int],
    replicates: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """
    Return bootstrap-based uncertainty estimates for accuracy metrics.

    :param rows: Iterable of evaluation rows or metrics to analyse.
    :param best_k: Neighbourhood size selected as optimal for the evaluation.
    :param baseline_index: Precomputed index that produces baseline recommendations.
    :param replicates: Number of bootstrap replicates to sample.
    :param seed: Seed used to initialise pseudo-random operations.
    :returns: Mapping containing accuracy uncertainty summaries or ``None``.
    """

    if replicates <= 0:
        return None
    eligible_rows = [row for row in rows if row.get("eligible")]
    if not eligible_rows:
        return None
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for idx, row in enumerate(eligible_rows):
        group_key = row.get("group_key") or group_key_for_example(row, idx)
        grouped.setdefault(group_key, []).append(row)
    if len(grouped) < 2:
        return None

    def _model_metric(items: Sequence[Mapping[str, Any]]) -> float:
        return _accuracy_for_rows(items, best_k)

    baseline_metric = None
    if baseline_index is not None:

        def _baseline_metric(items: Sequence[Mapping[str, Any]]) -> float:
            return _baseline_accuracy_for_rows(items, baseline_index)

        baseline_metric = _baseline_metric

    return summarise_grouped_accuracy_from_counts(
        grouped=grouped,
        counts=BootstrapCounts(
            n_rows=len(eligible_rows),
            n_bootstrap=replicates,
            seed=seed,
        ),
        model_metric=_model_metric,
        baseline_metric=baseline_metric,
    )


__all__ = ["bootstrap_uncertainty"]
