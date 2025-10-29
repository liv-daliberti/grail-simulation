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

"""Shared dataclasses used throughout the XGBoost evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass  # pylint: disable=too-many-instance-attributes
class IssueMetrics:
    """Container describing evaluation metrics for a single issue.

    :ivar issue: Human-readable issue label.
    :ivar participant_studies: Participant study filters applied to evaluation rows.
    :ivar dataset_source: Identifier of the dataset (path or Hub id).
    :ivar evaluated: Number of evaluation rows processed.
    :ivar correct: Count of correct slate selections.
    :ivar accuracy: Overall accuracy across all evaluated rows.
    :ivar known_candidate_hits: Correct selections where the candidate was seen during training.
    :ivar known_candidate_total: Evaluations containing at least one known candidate.
    :ivar coverage: Fraction of evaluations with at least one known candidate.
    :ivar avg_probability: Mean probability assigned to known predictions.
    :ivar eligible: Count of slates containing the gold option.
    :ivar timestamp: UNIX timestamp when the metrics were produced.
    :ivar extra_fields: Extra prompt fields incorporated during evaluation.
    :ivar xgboost_params: Serialised model parameter summary.
    :ivar correct_eligible: Correct selections restricted to eligible slates.
    :ivar accuracy_eligible: Accuracy among eligible slates.
    :ivar known_accuracy: Accuracy on rows with known candidates.
    :ivar known_availability: Ratio of rows with known candidates to those evaluated.
    :ivar evaluated_predicted: Evaluations where the model emitted a prediction.
    :ivar accuracy_predicted: Accuracy among rows with predictions.
    :ivar baseline_most_frequent_gold_index: Baseline statistics for the most frequent gold index.
    :ivar random_baseline_expected_accuracy: Expected accuracy of a random chooser baseline.
    :ivar curve_metrics: Learning-curve payload summarising cumulative accuracy trends.
    :ivar curve_metrics_path: Optional filesystem path where curve metrics were persisted.
    :ivar accuracy_ci_95: Bootstrap 95% confidence interval for eligible-only accuracy.
    :ivar accuracy_uncertainty: Full bootstrap payload including model/baseline statistics.
    """

    issue: str
    participant_studies: Sequence[str]
    dataset_source: str
    evaluated: int
    correct: int
    accuracy: float
    known_candidate_hits: int
    known_candidate_total: int
    coverage: float
    avg_probability: float
    eligible: int
    timestamp: float
    extra_fields: Sequence[str]
    xgboost_params: Dict[str, Any]
    correct_eligible: int = 0
    accuracy_eligible: float = 0.0
    known_accuracy: Optional[float] = None
    known_availability: Optional[float] = None
    evaluated_predicted: int = 0
    accuracy_predicted: float = 0.0
    baseline_most_frequent_gold_index: Dict[str, Any] = field(default_factory=dict)
    random_baseline_expected_accuracy: Optional[float] = None
    curve_metrics: Optional[Dict[str, Any]] = None
    curve_metrics_path: Optional[str] = None
    accuracy_ci_95: Optional[Dict[str, float]] = None
    accuracy_uncertainty: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)  # pylint: disable=too-many-instance-attributes
class EvaluationConfig:
    """Configuration bundle shared across evaluation helpers.

    :ivar dataset_source: Identifier for the dataset source (path or Hub id).
    :ivar extra_fields: Additional column names appended to prompt documents.
    :ivar eval_max: Maximum number of evaluation rows to process (0 keeps all).
    :ivar participant_studies: Participant study filters applied to evaluation splits.
    """

    dataset_source: str
    extra_fields: Sequence[str]
    eval_max: int
    participant_studies: Sequence[str]


@dataclass(frozen=True)
class IssueEvaluationContext:
    """Static context shared across issue evaluations within a single run.

    :ivar dataset_source: Identifier for the dataset source.
    :ivar extra_fields: Extra prompt fields used for feature construction.
    :ivar train_study_tokens: Participant study filters for the training split.
    :ivar eval_study_tokens: Participant study filters for the evaluation split.
    """

    dataset_source: str
    extra_fields: Sequence[str]
    train_study_tokens: Sequence[str]
    eval_study_tokens: Sequence[str]


@dataclass(frozen=True)  # pylint: disable=too-many-instance-attributes
class PredictionOutcome:
    """Result bundle for a single evaluation example.

    :ivar prediction_index: 1-based predicted option index (``None`` when abstaining).
    :ivar predicted_id: Video identifier returned by the model.
    :ivar gold_video_id: Ground-truth video identifier.
    :ivar candidate_probs: Mapping of slate indices to predicted probabilities.
    :ivar best_probability: Probability associated with the predicted option.
    :ivar known_candidate_seen: Flag indicating at least one known candidate was present.
    :ivar known_candidate_hit: Flag indicating the prediction matched a known candidate.
    :ivar record_probability: Flag controlling whether the probability contributes to aggregates.
    :ivar correct: ``True`` when the predicted candidate matches the gold id.
    :ivar option_count: Number of slate options presented to the model.
    :ivar gold_index: 1-based index of the gold candidate (when present).
    :ivar eligible: ``True`` when the slate contained the gold candidate.
    """

    prediction_index: Optional[int]
    predicted_id: str
    gold_video_id: str
    candidate_probs: Dict[int, float]
    best_probability: float
    known_candidate_seen: bool
    known_candidate_hit: bool
    record_probability: bool
    correct: bool
    option_count: int
    gold_index: Optional[int]
    eligible: bool


@dataclass(frozen=True)
class ProbabilityContext:
    """Aggregated probability metadata for a slate prediction.

    :ivar best_probability: Probability assigned to the chosen candidate.
    :ivar record_probability: ``True`` when the probability should be recorded.
    :ivar known_candidate_hit: ``True`` when the predicted candidate matches the gold id and was known.
    """

    best_probability: float
    record_probability: bool
    known_candidate_hit: bool


@dataclass(frozen=True)  # pylint: disable=too-many-instance-attributes
class OutcomeSummary:
    """Aggregated metrics derived from prediction outcomes.

    :ivar evaluated: Number of evaluation rows processed.
    :ivar correct: Count of correct slate selections.
    :ivar known_hits: Correct selections among known candidates.
    :ivar known_total: Evaluations containing at least one known candidate.
    :ivar avg_probability: Mean probability recorded for known predictions.
    :ivar eligible: Count of slates where the gold option was present.
    :ivar gold_hist: Histogram of observed gold indices for eligible slates.
    :ivar random_inverse_sum: Sum of ``1 / n_options`` contributions for random baseline accuracy.
    :ivar random_inverse_count: Number of eligible slates contributing to the random baseline.
    """

    evaluated: int
    correct: int
    known_hits: int
    known_total: int
    avg_probability: float
    eligible: int
    gold_hist: Dict[int, int]
    random_inverse_sum: float
    random_inverse_count: int


__all__ = [
    "EvaluationConfig",
    "IssueEvaluationContext",
    "IssueMetrics",
    "OutcomeSummary",
    "PredictionOutcome",
    "ProbabilityContext",
]
