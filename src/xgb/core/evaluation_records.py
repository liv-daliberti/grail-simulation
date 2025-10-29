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

"""Prediction-record utilities extracted from the XGBoost evaluation module."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from common.evaluation.utils import group_key_for_example

from .data import SOLUTION_COLUMN
from .evaluation_probabilities import candidate_probabilities, probability_context
from .evaluation_types import EvaluationConfig, PredictionOutcome
from .features import extract_slate_items
from .model import XGBoostSlateModel, predict_among_slate
from .utils import canon_video_id, get_logger

logger = get_logger("xgb.eval")


def collect_prediction_records(  # pylint: disable=too-many-locals
    model: XGBoostSlateModel,
    eval_ds,
    config: EvaluationConfig,
) -> List[Tuple[int, PredictionOutcome]]:
    """Collect indexed prediction outcomes for the evaluation split.

    :param model: Trained slate model used to score candidate lists.
    :param eval_ds: Iterable representing the evaluation dataset split.
    :param config: Evaluation configuration controlling limits and extra fields.
    :returns: Ordered ``(row_index, PredictionOutcome)`` tuples.
    """

    records: List[Tuple[int, PredictionOutcome]] = []
    correct_so_far = 0
    elig_seen_so_far = 0
    elig_correct_so_far = 0
    last_log = 0

    for index, example in enumerate(eval_ds):
        if config.eval_max and len(records) >= config.eval_max:
            break
        outcome = evaluate_single_example(
            model=model,
            example=example,
            extra_fields=config.extra_fields,
        )
        records.append((index, outcome))

        if outcome.correct:
            correct_so_far += 1
        if outcome.eligible:
            elig_seen_so_far += 1
            if outcome.correct:
                elig_correct_so_far += 1

        if len(records) - last_log >= 50:
            last_log = len(records)
            overall_acc = correct_so_far / len(records) if records else 0.0
            elig_acc = (
                (elig_correct_so_far / elig_seen_so_far) if elig_seen_so_far else 0.0
            )
            logger.info(
                "[XGBoost][Eval] processed=%d overall_acc=%.4f eligible_acc=%.4f",
                len(records),
                overall_acc,
                elig_acc,
            )
    return records


def evaluate_single_example(  # pylint: disable=too-many-locals
    *,
    model: XGBoostSlateModel,
    example: Mapping[str, Any],
    extra_fields: Sequence[str],
) -> PredictionOutcome:
    """Score an individual example and return the prediction outcome bundle.

    :param model: Trained slate model used for inference.
    :param example: Dataset row containing prompt text and slate candidates.
    :param extra_fields: Additional columns appended to the prompt document.
    :returns: :class:`PredictionOutcome` describing the model decision.
    """

    prediction_idx, probability_map = predict_among_slate(
        model,
        example,
        extra_fields=extra_fields,
    )
    slate = extract_slate_items(example)
    option_count = len(slate)
    gold_id = example.get(SOLUTION_COLUMN) or ""
    gold_id_canon = canon_video_id(gold_id)
    raw_gold_index = example.get("gold_index")
    gold_index: Optional[int] = None
    if raw_gold_index is not None:
        try:
            parsed_index = int(raw_gold_index)
        except (TypeError, ValueError):
            parsed_index = None
        if parsed_index is not None and 1 <= parsed_index <= option_count:
            gold_index = parsed_index
    if gold_index is None and gold_id_canon:
        for idx, (_title, candidate_id) in enumerate(slate, start=1):
            if canon_video_id(candidate_id) == gold_id_canon:
                gold_index = idx
                break

    predicted_id = slate[prediction_idx - 1][1] if (
        prediction_idx is not None and 1 <= prediction_idx <= option_count
    ) else ""

    candidate_probs, known_candidates = candidate_probabilities(slate, probability_map)
    probability_ctx = probability_context(
        prediction_idx=prediction_idx,
        candidate_probs=candidate_probs,
        known_candidates=known_candidates,
        gold_id_canon=gold_id_canon,
    )

    predicted_id_canon = canon_video_id(predicted_id)
    correct = predicted_id_canon == gold_id_canon and bool(predicted_id_canon)
    eligible = bool(
        gold_index is not None
        and option_count > 0
        and 1 <= gold_index <= option_count
    )

    return PredictionOutcome(
        prediction_index=prediction_idx,
        predicted_id=predicted_id,
        gold_video_id=gold_id,
        candidate_probs=candidate_probs,
        best_probability=probability_ctx.best_probability,
        known_candidate_seen=bool(known_candidates),
        known_candidate_hit=probability_ctx.known_candidate_hit,
        record_probability=probability_ctx.record_probability,
        correct=correct,
        option_count=option_count,
        gold_index=gold_index,
        eligible=eligible,
    )


def compute_group_keys(eval_ds, limit: int) -> List[str]:
    """Compute grouping keys for the first ``limit`` evaluation examples.

    :param eval_ds: Iterable of dataset rows.
    :param limit: Maximum number of rows inspected when generating keys.
    :returns: List of grouping keys aligned with the dataset order.
    """

    keys: List[str] = []
    for idx, row in enumerate(eval_ds):
        if idx >= limit:
            break
        try:
            keys.append(group_key_for_example(row, idx))
        except (TypeError, AttributeError):
            keys.append(f"row::{idx}")
    return keys


__all__ = [
    "collect_prediction_records",
    "compute_group_keys",
    "evaluate_single_example",
    "group_key_for_example",
]
