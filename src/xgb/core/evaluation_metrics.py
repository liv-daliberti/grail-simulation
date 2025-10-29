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

"""Metric aggregation helpers for the XGBoost evaluation pipeline."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from common.evaluation.utils import build_participant_bootstrap_summary, safe_div

from .evaluation_records import collect_prediction_records
from .evaluation_types import (
    EvaluationConfig,
    IssueMetrics,
    OutcomeSummary,
    PredictionOutcome,
)
from .model import XGBoostSlateModel


def bootstrap_uncertainty(  # pylint: disable=too-many-locals
    *,
    records: Sequence[Tuple[int, PredictionOutcome]],
    group_keys: Sequence[str],
    baseline_index: Optional[int],
    replicates: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Estimate eligible-only accuracy uncertainty via participant bootstrap.

    :param records: Indexed prediction outcomes from evaluation.
    :param group_keys: Group identifiers aligning with ``records`` for resampling.
    :param baseline_index: Optional baseline index used for comparison curves.
    :param replicates: Number of bootstrap resamples to generate.
    :param seed: Seed forwarded to ``numpy.random.default_rng``.
    :returns: Bootstrap summary payload with mean/CI statistics, or ``None`` when unavailable.
    """

    if replicates <= 0 or not records:
        return None

    grouped: Dict[str, List[int]] = {}
    elig_indices: List[int] = []
    for idx, (_index, outcome) in enumerate(records):
        key = group_keys[idx] if idx < len(group_keys) else f"row::{idx}"
        if outcome.eligible:
            grouped.setdefault(key, []).append(idx)
            elig_indices.append(idx)
    if len(grouped) < 2 or not elig_indices:
        return None

    keys = list(grouped.keys())
    rng = np.random.default_rng(seed)

    def _acc_for_indices(indices: Sequence[int]) -> float:
        """Compute model accuracy for the given record ``indices``.

        :param indices: Record indices referencing the local ``records`` sequence.
        :returns: Share of correct predictions across ``indices``.
        """
        correct = sum(1 for i in indices if records[i][1].correct)
        return correct / len(indices) if indices else 0.0

    def _baseline_acc_for_indices(indices: Sequence[int]) -> float:
        """Compute baseline accuracy for the provided record ``indices``.

        :param indices: Record indices referencing the local ``records`` sequence.
        :returns: Share of baseline hits across ``indices``; ``0.0`` if baseline unset.
        """
        if baseline_index is None:
            return 0.0
        correct = 0
        for i in indices:
            outcome = records[i][1]
            if outcome.gold_index == baseline_index:
                correct += 1
        return correct / len(indices) if indices else 0.0

    model_samples: List[float] = []
    baseline_samples: List[float] = []
    for _ in range(replicates):
        sampled_rows: List[int] = []
        sampled_group_indices = rng.integers(0, len(keys), size=len(keys))
        for gidx in sampled_group_indices:
            sampled_rows.extend(grouped[keys[gidx]])
        model_samples.append(_acc_for_indices(sampled_rows))
        if baseline_index is not None:
            baseline_samples.append(_baseline_acc_for_indices(sampled_rows))

    return build_participant_bootstrap_summary(
        model_samples=model_samples,
        baseline_samples=baseline_samples or None,
        n_groups=len(grouped),
        n_rows=len(elig_indices),
        n_bootstrap=replicates,
        seed=seed,
    )


def summarise_records(  # pylint: disable=too-many-locals
    records: List[Tuple[int, PredictionOutcome]],
    config: EvaluationConfig,
    issue_slug: str,
    model: XGBoostSlateModel,
) -> IssueMetrics:
    """Aggregate prediction records into an :class:`IssueMetrics` summary.

    :param records: Indexed prediction outcomes for the evaluation split.
    :param config: Evaluation configuration describing dataset metadata.
    :param issue_slug: Normalised issue identifier for reporting.
    :param model: Model bundle used to augment metrics with training parameters.
    :returns: Metrics bundle ready for serialisation to ``metrics.json``.
    """

    summary = summarise_outcomes(records)
    baseline_top_index: Optional[int] = None
    baseline_count = 0
    if summary.gold_hist:
        baseline_top_index, baseline_count = max(
            summary.gold_hist.items(),
            key=lambda item: item[1],
        )
    baseline_accuracy: Optional[float] = None
    if baseline_count and summary.eligible:
        baseline_accuracy = safe_div(baseline_count, summary.eligible)
    random_accuracy: Optional[float] = None
    if summary.random_inverse_count:
        random_accuracy = safe_div(
            summary.random_inverse_sum,
            summary.random_inverse_count,
        )

    baseline_payload: Dict[str, Any] = {"accuracy": baseline_accuracy}
    if baseline_top_index is not None:
        baseline_payload["top_index"] = baseline_top_index
        baseline_payload["count"] = baseline_count

    correct_eligible = sum(
        outcome.correct for _, outcome in records if outcome.eligible
    )
    accuracy_eligible = safe_div(correct_eligible, summary.eligible)
    known_accuracy = safe_div(summary.known_hits, summary.known_total)
    known_availability = safe_div(summary.known_total, summary.evaluated)
    evaluated_predicted = sum(
        outcome.prediction_index is not None for _, outcome in records
    )
    correct_predicted = sum(
        outcome.correct and outcome.prediction_index is not None for _, outcome in records
    )
    accuracy_predicted = (
        safe_div(correct_predicted, evaluated_predicted)
        if evaluated_predicted
        else 0.0
    )

    return IssueMetrics(
        issue=issue_slug,
        participant_studies=tuple(config.participant_studies),
        dataset_source=config.dataset_source,
        evaluated=summary.evaluated,
        correct=summary.correct,
        accuracy=safe_div(summary.correct, summary.evaluated),
        correct_eligible=int(correct_eligible),
        accuracy_eligible=float(accuracy_eligible),
        known_candidate_hits=summary.known_hits,
        known_candidate_total=summary.known_total,
        coverage=safe_div(summary.known_hits, summary.known_total),
        known_accuracy=known_accuracy,
        known_availability=known_availability,
        evaluated_predicted=evaluated_predicted,
        accuracy_predicted=accuracy_predicted,
        avg_probability=summary.avg_probability,
        eligible=summary.eligible,
        timestamp=time.time(),
        extra_fields=tuple(config.extra_fields),
        xgboost_params=model_params(model),
        baseline_most_frequent_gold_index=baseline_payload,
        random_baseline_expected_accuracy=random_accuracy,
    )


def records_to_predictions(
    records: List[Tuple[int, PredictionOutcome]],
    issue_slug: str,
) -> List[Dict[str, Any]]:
    """Serialise prediction outcomes into JSON-friendly dictionaries.

    :param records: Indexed prediction outcomes emitted during evaluation.
    :param issue_slug: Issue identifier included in each serialised entry.
    :returns: List of dictionaries mirroring the predictions JSONL format.
    """

    return [
        {
            "issue": issue_slug,
            "index": index,
            "prediction_index": outcome.prediction_index,
            "predicted_video_id": outcome.predicted_id,
            "gold_video_id": outcome.gold_video_id,
            "correct": outcome.correct,
            "probabilities": outcome.candidate_probs,
        }
        for index, outcome in records
    ]


def accuracy_curve_from_records(
    records: Sequence[Tuple[int, PredictionOutcome]],
    *,
    target_points: int = 50,
) -> Dict[str, Any]:
    """Build cumulative accuracy checkpoints for plotting learning curves.

    :param records: Ordered prediction outcomes produced during evaluation.
    :param target_points: Approximate number of checkpoints to retain.
    :returns: Mapping containing accuracy checkpoints, totals, and stride.
    """

    total = len(records)
    if total == 0:
        return {
            "accuracy_by_step": {},
            "eligible_accuracy_by_step": {},
            "n_examples": 0,
            "stride": 0,
        }
    target_points = max(1, target_points)
    stride = max(1, total // target_points)
    checkpoints: Dict[str, float] = {}
    elig_checkpoints: Dict[str, float] = {}
    correct = 0
    elig_correct = 0
    elig_seen = 0
    for idx, (_index, outcome) in enumerate(records, start=1):
        if outcome.correct:
            correct += 1
        if outcome.eligible:
            elig_seen += 1
            if outcome.correct:
                elig_correct += 1
        if idx == total or idx % stride == 0:
            checkpoints[str(idx)] = safe_div(correct, idx)
            if elig_seen > 0:
                elig_checkpoints[str(idx)] = safe_div(elig_correct, elig_seen)
    if str(total) not in checkpoints:
        checkpoints[str(total)] = safe_div(correct, total)
        if elig_seen > 0:
            elig_checkpoints[str(total)] = safe_div(elig_correct, elig_seen)
    return {
        "accuracy_by_step": checkpoints,
        "eligible_accuracy_by_step": elig_checkpoints,
        "n_examples": total,
        "stride": stride,
    }


def curve_metrics_from_training_history(
    history: Optional[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Convert XGBoost evaluation history into round-based accuracy curves.

    :param history: Evaluation history returned by ``XGBClassifier.evals_result``.
    :returns: Curve payload with per-round accuracy/error, or ``None`` when unavailable.
    """

    if not history or not isinstance(history, Mapping):
        return None

    train_payload = history.get("validation_0")
    eval_payload = history.get("validation_1")
    if not isinstance(train_payload, Mapping) or not isinstance(eval_payload, Mapping):
        return None

    train_errors = train_payload.get("merror")
    eval_errors = eval_payload.get("merror")
    if not train_errors or not eval_errors:
        return None

    def _accuracy_series(error_sequence: Sequence[Any]) -> Dict[str, float]:
        """Convert an ``merror`` series into per-round accuracy values.

        :param error_sequence: Raw ``merror`` values enumerated by boosting round.
        :returns: Mapping from 1-based round string to clipped accuracy value.
        """
        return {
            str(idx + 1): float(max(0.0, min(1.0, 1.0 - float(value))))
            for idx, value in enumerate(error_sequence)
        }

    def _error_series(error_sequence: Sequence[Any]) -> Dict[str, float]:
        """Format an ``merror`` sequence for plotting keyed by round.

        :param error_sequence: Raw ``merror`` values enumerated by boosting round.
        :returns: Mapping from 1-based round string to error value.
        """
        return {
            str(idx + 1): float(value)
            for idx, value in enumerate(error_sequence)
        }

    return {
        "metric": "merror",
        "axis_label": "Boosting rounds",
        "y_label": "Classification accuracy",
        "train": {
            "accuracy_by_round": _accuracy_series(train_errors),
            "merror_by_round": _error_series(train_errors),
            "n_rounds": len(train_errors),
        },
        "eval": {
            "accuracy_by_round": _accuracy_series(eval_errors),
            "merror_by_round": _error_series(eval_errors),
            "n_rounds": len(eval_errors),
        },
    }


def curve_metrics_for_split(
    model: XGBoostSlateModel,
    dataset,
    extra_fields: Sequence[str],
    *,
    target_points: int = 50,
) -> Dict[str, Any]:
    """Compute cumulative accuracy metrics for an arbitrary dataset split.

    :param model: Trained slate model used for inference.
    :param dataset: Iterable of dataset rows to evaluate.
    :param extra_fields: Additional columns appended to the prompt document.
    :param target_points: Approximate number of checkpoints to retain.
    :returns: Accuracy curve payload mirroring :func:`accuracy_curve_from_records`.
    """

    config = EvaluationConfig(
        dataset_source="curve",
        extra_fields=tuple(extra_fields),
        eval_max=0,
        participant_studies=(),
    )
    records = collect_prediction_records(model, dataset, config)
    return accuracy_curve_from_records(records, target_points=target_points)


def summarise_outcomes(
    records: List[Tuple[int, PredictionOutcome]]
) -> OutcomeSummary:
    """Aggregate prediction outcomes into summary counts.

    :param records: Sequence of ``(index, PredictionOutcome)`` tuples.
    :returns: :class:`OutcomeSummary` capturing accuracy, coverage, and baseline stats.
    """

    outcomes = [outcome for _, outcome in records]
    evaluated = len(outcomes)
    known_total = sum(outcome.known_candidate_seen for outcome in outcomes)
    known_hits = sum(outcome.known_candidate_hit for outcome in outcomes)
    eligible = 0
    gold_hist: Dict[int, int] = {}
    random_inverse_sum = 0.0
    random_inverse_count = 0
    probability_values = [
        outcome.best_probability
        for outcome in outcomes
        if outcome.record_probability
    ]
    avg_probability = float(np.mean(probability_values)) if probability_values else 0.0
    correct = sum(outcome.correct for outcome in outcomes)
    for outcome in outcomes:
        if (
            outcome.eligible
            and outcome.gold_index is not None
            and outcome.option_count > 0
        ):
            eligible += 1
            gold_hist[outcome.gold_index] = gold_hist.get(outcome.gold_index, 0) + 1
            random_inverse_sum += 1.0 / outcome.option_count
            random_inverse_count += 1
    return OutcomeSummary(
        evaluated=evaluated,
        correct=correct,
        known_hits=known_hits,
        known_total=known_total,
        avg_probability=avg_probability,
        eligible=eligible,
        gold_hist=gold_hist,
        random_inverse_sum=random_inverse_sum,
        random_inverse_count=random_inverse_count,
    )


def model_params(model: XGBoostSlateModel) -> Dict[str, Any]:
    """Return a serialisable view of relevant XGBoost parameters.

    :param model: Model bundle whose configuration should be summarised.
    :returns: Dictionary containing the key training parameters and vectoriser metadata.
    """

    params = model.booster.get_params()
    selected = {
        key: params.get(key)
        for key in [
            "objective",
            "eval_metric",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "tree_method",
            "reg_lambda",
            "reg_alpha",
        ]
    }
    selected["extra_fields"] = list(model.extra_fields)
    if hasattr(model.vectorizer, "metadata"):
        vectorizer_meta = model.vectorizer.metadata()  # type: ignore[assignment]
        selected["vectorizer"] = vectorizer_meta
        selected["n_features"] = int(vectorizer_meta.get("dimension", 0))
    else:
        selected["n_features"] = int(getattr(model.vectorizer, "max_features", 0) or 0)
    selected["n_classes"] = int(len(model.label_encoder.classes_))
    return selected


__all__ = [
    "accuracy_curve_from_records",
    "bootstrap_uncertainty",
    "curve_metrics_for_split",
    "curve_metrics_from_training_history",
    "model_params",
    "records_to_predictions",
    "summarise_outcomes",
    "summarise_records",
]
