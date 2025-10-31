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

"""Dataset-level evaluation utilities for the KNN baseline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from common.evaluation.utils import group_key_for_example, safe_div

from ..data import SOLUTION_COLUMN
from ..features import extract_slate_items
from ..index import knn_predict_among_slate_multi
from .indexes import SlateQueryConfig
from .k_selection import select_best_k
from .utils import BUCKET_LABELS, bin_nopts, bucket_from_pos, canon


@dataclass(frozen=True)
class RowEvaluationConfig:
    """Immutable configuration controlling per-row evaluation."""

    k_values: Sequence[int]
    knn_index: Mapping[str, Any]
    query_config: SlateQueryConfig


@dataclass
class EvaluationState:
    """Mutable aggregates captured while iterating over a dataset split."""

    bucket_stats: MutableMapping[str, MutableMapping[str, int]]
    per_k_stats: MutableMapping[int, MutableMapping[str, int]]
    single_multi_stats: MutableMapping[str, float | int]
    gold_hist: MutableMapping[int, int]

    @classmethod
    def initialise(cls, k_values: Sequence[int]) -> "EvaluationState":
        """
        Return a fresh state scaffolded for ``k_values``.

        :param k_values: Sequence of neighbourhood sizes being tracked.
        :returns: New :class:`EvaluationState` instance with zeroed counts.
        """

        per_k = {int(k): {"eligible": 0, "correct": 0} for k in k_values}
        buckets = {
            "position_seen": {bucket: 0 for bucket in BUCKET_LABELS},
            "options_seen": {bucket: 0 for bucket in BUCKET_LABELS},
            "options_eligible": {bucket: 0 for bucket in BUCKET_LABELS},
            "options_correct": {bucket: 0 for bucket in BUCKET_LABELS},
        }
        single_multi: Dict[str, float | int] = {
            "seen_single": 0,
            "seen_multi": 0,
            "elig_single": 0,
            "elig_multi": 0,
            "corr_single": 0,
            "corr_multi": 0,
            "rand_inverse_sum": 0.0,
            "rand_inverse_count": 0,
        }
        return cls(
            bucket_stats=buckets,
            per_k_stats=per_k,
            single_multi_stats=single_multi,
            gold_hist={},
        )


@dataclass(frozen=True)
class DatasetEvaluationOptions:
    """Optional modifiers for dataset evaluation runs."""

    capture_rows: bool = False
    log_label: str = "eval"
    max_examples: Optional[int] = None
    log_k: Optional[int] = None
    k_select_method: Optional[str] = None


@dataclass(frozen=True)
class DatasetEvaluationRequest:
    """Inputs required to score a dataset split using the KNN index."""

    dataset: Any
    k_values: Sequence[int]
    knn_index: Mapping[str, Any]
    extra_fields: Sequence[str]
    metric: str
    options: DatasetEvaluationOptions = field(default_factory=DatasetEvaluationOptions)


def _update_random_baseline(
    single_multi_stats: MutableMapping[str, float | int],
    n_options: int,
) -> None:
    """
    Update the inverse-random baseline aggregates for an eligible slate.

    :param single_multi_stats: Running statistics keyed by single/multi-slate labels.
    :param n_options: Number of candidate options on the current slate.
    :returns: ``None``. Mutates ``single_multi_stats`` in-place.
    """

    if n_options <= 0:
        return
    single_multi_stats["rand_inverse_sum"] += 1.0 / n_options
    single_multi_stats["rand_inverse_count"] += 1


def _progress_message(
    *,
    state: EvaluationState,
    config: RowEvaluationConfig,
    method: str,
    log_k_value: Optional[int],
) -> str:
    """
    Build the logging string summarising evaluation progress.

    :param state: Evaluation aggregates captured so far.
    :param config: Immutable configuration describing the evaluation setup.
    :param method: K-selection method used to pick the logging accuracy.
    :param log_k_value: Specific ``k`` requested for logging, if any.
    :returns: Human-readable accuracy string for the current iteration.
    """

    if log_k_value is not None:
        stats = state.per_k_stats[log_k_value]
        return f"  acc@{log_k_value}={safe_div(stats['correct'], stats['eligible']):.3f}"

    accuracy_now = {
        int(k): safe_div(stats["correct"], stats["eligible"])
        for k, stats in state.per_k_stats.items()
    }
    chosen_k = select_best_k(config.k_values, accuracy_now, method=method)
    stats_now = state.per_k_stats.get(int(chosen_k), {"correct": 0, "eligible": 0})
    return f"  acc@{int(chosen_k)}={safe_div(stats_now['correct'], stats_now['eligible']):.3f}"


def _resolve_gold_index(
    example: Mapping[str, Any],
    slate_pairs: Sequence[Tuple[str, Any]],
) -> int:
    """
    Determine the 1-based gold index for the provided example.

    The raw ``gold_index`` field is honoured when present; otherwise the function
    attempts to align the ``solution`` field against the slate entries.

    :param example: Dataset row containing the gold metadata.
    :param slate_pairs: Ordered slate options as ``(title, candidate_id)`` tuples.
    :returns: Resolved 1-based gold index, or ``-1`` when unresolved.
    """

    gold_value = example.get("gold_index")
    try:
        gold_index = int(gold_value)
    except (TypeError, ValueError):
        gold_index = -1

    if gold_index > 0 or not slate_pairs:
        return gold_index

    gold_raw = str(example.get(SOLUTION_COLUMN, "")).strip()
    if not gold_raw:
        return gold_index

    canonical_gold = canon(gold_raw)
    for option_index, (title, vid) in enumerate(slate_pairs, start=1):
        if gold_raw == vid or canonical_gold == canon(title):
            return option_index
    return gold_index


def _resolve_position(example: Mapping[str, Any], gold_index: int) -> int:
    """
    Resolve the position index for logging into position buckets.

    :param example: Dataset row providing the raw ``video_index``.
    :param gold_index: Resolved 1-based gold index for the example.
    :returns: Zero-based position index, ``-1`` when not available.
    """

    try:
        position_value = example.get("video_index")
        position = int(position_value)
    except (TypeError, ValueError):
        position = -1

    if position < 0 < gold_index:
        return gold_index - 1
    return position


def _update_option_counters(
    state: EvaluationState,
    *,
    eligible: bool,
    n_options: int,
    n_bucket: str,
    gold_index: int,
) -> None:
    """
    Update slate-level counters for the observed example.

    :param state: Mutable evaluation aggregates.
    :param eligible: Whether the current row contributes to accuracy metrics.
    :param n_options: Number of options shown on the slate.
    :param n_bucket: Bucket label corresponding to ``n_options``.
    :param gold_index: 1-based index of the correct option, if resolvable.
    :returns: ``None``. Mutates the supplied state.
    """

    stats = state.single_multi_stats
    if n_options == 1:
        stats["seen_single"] += 1
    elif n_options > 1:
        stats["seen_multi"] += 1

    if not eligible:
        return

    state.bucket_stats["options_eligible"][n_bucket] += 1
    state.gold_hist[gold_index] = state.gold_hist.get(gold_index, 0) + 1
    _update_random_baseline(stats, n_options)
    if n_options == 1:
        stats["elig_single"] += 1
    else:
        stats["elig_multi"] += 1


def _update_per_k_stats_for_row(
    per_k_stats: MutableMapping[int, MutableMapping[str, int]],
    predictions: Mapping[int, Any],
    gold_index: int,
) -> None:
    """
    Update per-k eligibility and correctness counters for an eligible row.

    :param per_k_stats: Mutable per-k aggregates.
    :param predictions: Model predictions keyed by ``k``.
    :param gold_index: 1-based index of the correct option.
    :returns: ``None``. Updates ``per_k_stats`` in-place.
    """

    for k, pred in predictions.items():
        k_stats = per_k_stats[int(k)]
        k_stats["eligible"] += 1
        if pred is not None and int(pred) == gold_index:
            k_stats["correct"] += 1


def _determine_limit(dataset_size: int, max_examples: Optional[int]) -> int:
    """
    Resolve the number of examples to evaluate for the current run.

    :param dataset_size: Total number of examples in the split.
    :param max_examples: Optional cap on the number of rows to evaluate.
    :returns: Maximum number of rows to score.
    """

    if not max_examples or max_examples <= 0:
        return dataset_size
    return min(dataset_size, max_examples)


def _select_log_k(
    per_k_stats: Mapping[int, Mapping[str, int]],
    desired_log_k: Optional[int],
) -> Optional[int]:
    """
    Select the ``k`` value used for periodic progress logging.

    :param per_k_stats: Aggregated per-k statistics tracked by the evaluation.
    :param desired_log_k: Preferred ``k`` supplied by the caller, if any.
    :returns: Chosen logging ``k`` (exact or nearest), or ``None``.
    """

    if desired_log_k is None:
        return None

    desired = int(desired_log_k)
    if desired in per_k_stats:
        return desired
    if not per_k_stats:
        return None
    return min(per_k_stats.keys(), key=lambda key_k: abs(key_k - desired))


def accumulate_row(
    example: Mapping[str, Any],
    *,
    state: EvaluationState,
    config: RowEvaluationConfig,
    row_index: int,
) -> Dict[str, Any]:
    """
    Process a single evaluation example and update aggregate statistics.

    :param example: Dataset row representing one recommendation slate.
    :param state: Mutable aggregate state capturing running tallies.
    :param config: Immutable configuration describing index/query behaviour.
    :param row_index: Index of the row within the evaluation split.
    :returns: Serialised per-example record including predictions for each ``k``.
    """

    slate_pairs = extract_slate_items(example)
    n_options = len(slate_pairs)
    n_bucket = bin_nopts(n_options)
    state.bucket_stats["options_seen"][n_bucket] += 1
    gold_index = _resolve_gold_index(example, slate_pairs)
    position = _resolve_position(example, gold_index)
    position_bucket = bucket_from_pos(position)
    state.bucket_stats["position_seen"][position_bucket] += 1

    predictions = knn_predict_among_slate_multi(
        knn_index=config.knn_index,
        example=example,
        k_values=config.k_values,
        config=config.query_config,
    )

    eligible = gold_index > 0 and n_options > 0
    _update_option_counters(
        state,
        eligible=eligible,
        n_options=n_options,
        n_bucket=n_bucket,
        gold_index=gold_index,
    )

    if eligible:
        _update_per_k_stats_for_row(state.per_k_stats, predictions, gold_index)

    return {
        "predictions_by_k": predictions,
        "gold_index": int(gold_index),
        "n_options": int(n_options),
        "n_options_bucket": n_bucket,
        "eligible": bool(eligible),
        "position_index": int(position),
        "position_bucket": position_bucket,
        "issue_value": example.get("issue"),
        "group_key": group_key_for_example(example, row_index),
    }


def evaluate_dataset_split(request: DatasetEvaluationRequest) -> Dict[str, Any]:
    """
    Return aggregate statistics for ``request.dataset`` using the provided index.

    :param request: Dataset evaluation request describing the evaluation run.
    :returns: Dictionary containing rows, aggregate stats, and counts.
    """

    options = request.options
    limit = _determine_limit(len(request.dataset), options.max_examples)

    k_values_int = [int(k) for k in request.k_values]
    query_config = SlateQueryConfig(
        text_fields=tuple(request.extra_fields),
        lowercase=True,
        metric=request.metric,
    )
    config = RowEvaluationConfig(
        k_values=k_values_int,
        knn_index=request.knn_index,
        query_config=query_config,
    )
    state = EvaluationState.initialise(k_values_int)
    rows: List[Dict[str, Any]] = []

    log_k_value = _select_log_k(state.per_k_stats, options.log_k)

    start_time = time.time()
    method = (options.k_select_method or "max").strip().lower()

    for idx in range(int(limit)):
        row = accumulate_row(
            request.dataset[int(idx)],
            state=state,
            config=config,
            row_index=int(idx),
        )
        if options.capture_rows:
            rows.append(row)
        if (idx + 1) % 25 == 0:
            elapsed = time.time() - start_time
            acc_message = _progress_message(
                state=state,
                config=config,
                method=method,
                log_k_value=log_k_value,
            )
            logging.info(
                "[%s] %d/%d  elapsed=%.1fs%s",
                options.log_label,
                idx + 1,
                limit,
                elapsed,
                acc_message,
            )

    return {
        "rows": rows if options.capture_rows else [],
        "bucket_stats": state.bucket_stats,
        "per_k_stats": state.per_k_stats,
        "single_multi_stats": state.single_multi_stats,
        "gold_hist": state.gold_hist,
        "n_examples": int(limit),
    }


def update_correct_counts(
    rows: Sequence[Mapping[str, Any]],
    best_k: int,
    bucket_stats: MutableMapping[str, MutableMapping[str, int]],
    single_multi_stats: MutableMapping[str, int],
) -> None:
    """
    Update bucket-level correctness tallies for the selected ``best_k``.

    :param rows: Iterable of per-example prediction records.
    :param best_k: Selected ``k`` used to judge correctness.
    :param bucket_stats: Mutable dictionary storing per-bucket correctness.
    :param single_multi_stats: Mutable dictionary tracking single vs multi counts.
    """

    for row in rows:
        if not row["eligible"]:
            continue
        prediction = row["predictions_by_k"].get(best_k)
        if prediction is None:
            continue
        if int(prediction) == row["gold_index"]:
            bucket_stats["options_correct"][row["n_options_bucket"]] += 1
            if row["n_options"] == 1:
                single_multi_stats["corr_single"] += 1
            else:
                single_multi_stats["corr_multi"] += 1


__all__ = [
    "DatasetEvaluationRequest",
    "RowEvaluationConfig",
    "accumulate_row",
    "evaluate_dataset_split",
    "update_correct_counts",
    "DatasetEvaluationOptions",
]
