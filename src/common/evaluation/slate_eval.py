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

"""Shared slate-evaluation data structures and helpers.

The KNN, GPT-4o, and other slate-ranking baselines compute identical aggregate
metrics over per-example observations (accuracy, formatting rate, bucketed
breakdowns, etc.).  Keeping the accumulator logic in this module reduces code
duplication across the individual baselines.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .utils import safe_div

__all__ = [
    "BucketAccumulator",
    "EvaluationAccumulator",
    "EvaluationFilters",
    "SlateMetricsRequest",
    "Observation",
    "bucket_from_options",
    "bucket_from_position",
]


@dataclass
class BucketAccumulator:
    """
    Accumulate per-bucket evaluation counts used to compute rates.

    Tracks counts for each bucket key:
    - ``seen``: Total examples observed.
    - ``eligible``: Examples considered eligible for accuracy (valid gold/option count).
    - ``correct``: Eligible examples predicted correctly.
    - ``parsed``: Model outputs that could be parsed into an index.
    - ``formatted``: Model outputs that included the expected answer tag.
    """

    seen: Counter[str] = field(default_factory=Counter)
    eligible: Counter[str] = field(default_factory=Counter)
    correct: Counter[str] = field(default_factory=Counter)
    parsed: Counter[str] = field(default_factory=Counter)
    formatted: Counter[str] = field(default_factory=Counter)

    def record_seen(self, key: str) -> None:
        """
        Increment the seen counter for ``key``.

        :param key: Bucket identifier being updated.
        :returns: ``None``.
        """

        self.seen[key] += 1

    def record_formatted(self, key: str) -> None:
        """
        Increment the correctly formatted counter for ``key``.

        :param key: Bucket identifier being updated.
        :returns: ``None``.
        """

        self.formatted[key] += 1

    def record_parsed(self, key: str) -> None:
        """
        Increment the parsed counter for ``key``.

        :param key: Bucket identifier being updated.
        :returns: ``None``.
        """

        self.parsed[key] += 1

    def record_eligible(self, key: str) -> None:
        """
        Increment the eligible counter for ``key``.

        :param key: Bucket identifier being updated.
        :returns: ``None``.
        """

        self.eligible[key] += 1

    def record_correct(self, key: str) -> None:
        """
        Increment the correct counter for ``key``.

        :param key: Bucket identifier being updated.
        :returns: ``None``.
        """

        self.correct[key] += 1

    def summary(
        self,
        order: Sequence[str] | None = None,
        skip_empty: bool = False,
    ) -> Dict[str, Dict[str, float | int]]:
        """
        Return aggregated metrics per bucket.

        :param order: Optional explicit ordering for bucket keys.
        :param skip_empty: When ``True`` omit buckets with zero observations.
        :returns: Mapping from bucket to aggregate counts and rates.
        """

        keys = order if order is not None else sorted(self.seen.keys())
        result: Dict[str, Dict[str, float | int]] = {}
        for key in keys:
            seen = int(self.seen.get(key, 0))
            if skip_empty and seen == 0:
                continue
            eligible = int(self.eligible.get(key, 0))
            correct = int(self.correct.get(key, 0))
            parsed = int(self.parsed.get(key, 0))
            formatted = int(self.formatted.get(key, 0))
            result[key] = {
                "n_seen": seen,
                "n_eligible": eligible,
                "correct": correct,
                "accuracy": safe_div(correct, eligible),
                "parsed_rate": safe_div(parsed, seen),
                "format_rate": safe_div(formatted, seen),
            }
        return result

    def histogram(self, order: Sequence[str], attr: str) -> Dict[str, int]:
        """
        Return integer histogram for the requested attribute.

        :param order: Ordered list of buckets to include.
        :param attr: Name of the counter attribute to read (e.g. ``\"seen\"``).
        :returns: Mapping of bucket key to integer count.
        """

        counter: Counter[str] = getattr(self, attr)
        return {key: int(counter.get(key, 0)) for key in order}

    def ratio(self, numerator: str, denominator: str, order: Sequence[str]) -> Dict[str, float]:
        """
        Return ratios for ``numerator`` divided by ``denominator``.

        :param numerator: Counter attribute used as the numerator.
        :param denominator: Counter attribute used as the denominator.
        :param order: Ordered list of buckets to include in the output.
        :returns: Mapping of bucket key to ratio value.
        """

        num: Counter[str] = getattr(self, numerator)
        denom: Counter[str] = getattr(self, denominator)
        return {
            key: safe_div(num.get(key, 0), denom.get(key, 0))
            for key in order
        }


@dataclass(frozen=True)
class Observation:  # pylint: disable=too-many-instance-attributes
    """Snapshot of a single evaluation example."""

    issue_label: str
    study_label: str
    position_bucket: str
    option_bucket: str
    option_count: int
    gold_index: int
    parsed_index: Optional[int]
    is_formatted: bool
    eligible: bool
    is_correct: bool


@dataclass(frozen=True)
class EvaluationFilters:
    """Filter selections applied during evaluation."""

    issues: Sequence[str]
    studies: Sequence[str]


@dataclass(frozen=True)
class SlateMetricsRequest:
    """Payload describing the metadata included with evaluation metrics."""

    model_name: str
    dataset_name: str
    eval_split: str
    filters: EvaluationFilters
    position_order: Sequence[str] = ("1", "2", "3", "4", "5+", "unknown")
    option_order: Sequence[str] = ("1", "2", "3", "4", "5+")


@dataclass
class EvaluationAccumulator:  # pylint: disable=too-many-instance-attributes
    """Stateful accumulator tracking evaluation metrics across examples."""

    position: BucketAccumulator = field(default_factory=BucketAccumulator)
    options: BucketAccumulator = field(default_factory=BucketAccumulator)
    issue: BucketAccumulator = field(default_factory=BucketAccumulator)
    study: BucketAccumulator = field(default_factory=BucketAccumulator)
    total_seen: int = 0
    format_ok: int = 0
    parsed_ok: int = 0
    eligible_overall: int = 0
    correct_overall: int = 0
    seen_single: int = 0
    seen_multi: int = 0
    eligible_single: int = 0
    eligible_multi: int = 0
    correct_single: int = 0
    correct_multi: int = 0
    parsed_multi: int = 0
    formatted_multi: int = 0
    gold_hist: Counter[int] = field(default_factory=Counter)
    all_gold_indices: list[int] = field(default_factory=list)
    option_counts: list[int] = field(default_factory=list)

    def observe(self, obs: Observation) -> None:
        """
        Update aggregates for a single example observation.

        :param obs: Observation containing per-example evaluation data.
        :returns: ``None``.
        """

        self.total_seen += 1
        self._record_seen_buckets(obs)
        self._record_formatting_state(obs)
        self._record_option_structure(obs)
        self._record_eligibility(obs)
        self._record_correctness(obs)

    def _record_seen_buckets(self, obs: Observation) -> None:
        """
        Record initial bucket counters for the observation.

        :param obs: Observation containing bucket labels.
        :returns: ``None``.
        """

        self.position.record_seen(obs.position_bucket)
        self.issue.record_seen(obs.issue_label)
        self.study.record_seen(obs.study_label)
        self.options.record_seen(obs.option_bucket)

    def _record_formatting_state(self, obs: Observation) -> None:
        """
        Track formatting and parsing totals.

        :param obs: Observation describing formatting and parsing state.
        :returns: ``None``.
        """

        if obs.is_formatted:
            self.format_ok += 1
            self.issue.record_formatted(obs.issue_label)
            self.study.record_formatted(obs.study_label)
            self.options.record_formatted(obs.option_bucket)
        if obs.parsed_index is not None:
            self.parsed_ok += 1
            self.issue.record_parsed(obs.issue_label)
            self.study.record_parsed(obs.study_label)
            self.options.record_parsed(obs.option_bucket)

    def _record_option_structure(self, obs: Observation) -> None:
        """
        Track single-option vs. multi-option observations.

        :param obs: Observation describing option counts and formatting.
        :returns: ``None``.
        """

        if obs.option_count == 1:
            self.seen_single += 1
            return
        self.seen_multi += 1
        if obs.is_formatted:
            self.formatted_multi += 1
        if obs.parsed_index is not None:
            self.parsed_multi += 1

    def _record_eligibility(self, obs: Observation) -> None:
        """
        Capture eligibility-related aggregates.

        :param obs: Observation including eligibility and gold index details.
        :returns: ``None``.
        """

        if not obs.eligible:
            return
        self.eligible_overall += 1
        self.position.record_eligible(obs.position_bucket)
        self.issue.record_eligible(obs.issue_label)
        self.study.record_eligible(obs.study_label)
        self.options.record_eligible(obs.option_bucket)
        if obs.option_count == 1:
            self.eligible_single += 1
        else:
            self.eligible_multi += 1
        if obs.gold_index > 0:
            self.gold_hist[obs.gold_index] += 1
            self.all_gold_indices.append(obs.gold_index)
        if obs.option_count > 0:
            self.option_counts.append(obs.option_count)

    def _record_correctness(self, obs: Observation) -> None:
        """
        Capture correctness totals.

        :param obs: Observation including correctness and bucket membership.
        :returns: ``None``.
        """

        if not obs.is_correct:
            return
        self.correct_overall += 1
        self.position.record_correct(obs.position_bucket)
        self.issue.record_correct(obs.issue_label)
        self.study.record_correct(obs.study_label)
        self.options.record_correct(obs.option_bucket)
        if obs.option_count == 1:
            self.correct_single += 1
        else:
            self.correct_multi += 1

    def accuracy(self) -> float:
        """Return overall accuracy computed over eligible examples."""

        return safe_div(self.correct_overall, self.eligible_overall)

    def parsed_rate(self) -> float:
        """Return rate of successfully parsed completions."""

        return safe_div(self.parsed_ok, self.total_seen)

    def format_rate(self) -> float:
        """Return rate of outputs that matched the expected format."""

        return safe_div(self.format_ok, self.total_seen)

    def position_summary(self, order: Sequence[str]) -> Dict[str, Dict[str, float | int]]:
        """
        Return aggregated position metrics.

        :param order: Desired ordering of position buckets.
        :returns: Mapping of bucket names to aggregate counts and rates.
        """

        return self.position.summary(order=order)

    def options_summary(self, order: Sequence[str]) -> Dict[str, Any]:
        """
        Return aggregated option-count metrics.

        :param order: Desired ordering of option-count buckets.
        :returns: Mapping containing histograms and ratio metrics by bucket.
        """

        return {
            "hist_seen": self.options.histogram(order, "seen"),
            "hist_eligible": self.options.histogram(order, "eligible"),
            "hist_correct": self.options.histogram(order, "correct"),
            "accuracy": self.options.ratio("correct", "eligible", order),
            "parsed_rate": self.options.ratio("parsed", "seen", order),
            "format_rate": self.options.ratio("formatted", "seen", order),
        }

    def group_summary(self) -> Dict[str, Dict[str, Dict[str, float | int]]]:
        """Return per-issue and per-study aggregates."""

        return {
            "by_issue": self.issue.summary(skip_empty=True),
            "by_participant_study": self.study.summary(skip_empty=True),
        }

    def single_multi_summary(self) -> Dict[str, float | int]:
        """Return metrics comparing single-option and multi-option prompts."""

        return {
            "n_single": int(self.seen_single),
            "n_multi": int(self.seen_multi),
            "eligible_single": int(self.eligible_single),
            "eligible_multi": int(self.eligible_multi),
            "accuracy_single": safe_div(self.correct_single, self.eligible_single),
            "accuracy_multi": safe_div(self.correct_multi, self.eligible_multi),
            "parsed_rate_multi": safe_div(self.parsed_multi, max(1, self.seen_multi)),
            "format_rate_multi": safe_div(self.formatted_multi, max(1, self.seen_multi)),
        }

    def baseline_metrics(self) -> tuple[Dict[str, int], Dict[str, Any], float]:
        """Return gold index distribution and baseline accuracies."""

        distribution = {str(key): int(value) for key, value in sorted(self.gold_hist.items())}
        if not distribution:
            baseline = {"top_index": None, "count": 0, "accuracy": 0.0}
            expected_random = 0.0
        else:
            top_index = max(self.gold_hist.items(), key=lambda kv: kv[1])[0]
            baseline = {
                "top_index": top_index,
                "count": int(self.gold_hist[top_index]),
                "accuracy": safe_div(
                    sum(1 for idx in self.all_gold_indices if idx == top_index),
                    self.eligible_overall,
                ),
            }
            expected_random = (
                float(np.mean([1.0 / count for count in self.option_counts]))
                if self.option_counts
                else 0.0
            )
        return distribution, baseline, expected_random

    def metrics_payload(self, request: SlateMetricsRequest) -> Dict[str, Any]:
        """
        Return the metrics blob written to disk after evaluation.

        :param request: Request metadata describing the evaluation run.
        :returns: Serializable dictionary persisted to reporting artefacts.
        """

        gold_distribution, baseline_most_frequent, random_baseline = self.baseline_metrics()
        return {
            "model": request.model_name,
            "dataset": request.dataset_name,
            "split": request.eval_split,
            "n_total": int(self.total_seen),
            "n_eligible": int(self.eligible_overall),
            "accuracy_overall": self.accuracy(),
            "parsed_rate": self.parsed_rate(),
            "format_rate": self.format_rate(),
            "position_stats": self.position_summary(request.position_order),
            "by_n_options": self.options_summary(request.option_order),
            "split_single_vs_multi": self.single_multi_summary(),
            "group_metrics": self.group_summary(),
            "filters": {
                "issues": list(request.filters.issues),
                "studies": list(request.filters.studies),
            },
            "gold_index_distribution": gold_distribution,
            "baseline_most_frequent_gold_index": baseline_most_frequent,
            "random_baseline_expected_accuracy": random_baseline,
            "notes": (
                "Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown."
            ),
        }


def bucket_from_position(position_index: int) -> str:
    """
    Map a zero-based position index to an accuracy bucket label.

    :param position_index: Zero-based rank of the gold item within the slate.
    :returns: Bucket label describing the rank.
    """

    if position_index < 0:
        return "unknown"
    if position_index == 0:
        return "1"
    if position_index == 1:
        return "2"
    if position_index == 2:
        return "3"
    if position_index == 3:
        return "4"
    return "5+"


def bucket_from_options(count: int) -> str:
    """
    Normalise the number of slate options into histogram buckets.

    :param count: Total number of options shown to the participant.
    :returns: Bucket label describing the option count.
    """

    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count == 3:
        return "3"
    if count == 4:
        return "4"
    return "5+"
