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
class _NextVideoCore:
    """Core next-video metrics and baselines."""

    accuracy: Optional[float] = None
    coverage: Optional[float] = None
    accuracy_eligible: Optional[float] = None
    evaluated: Optional[int] = None
    correct: Optional[int] = None
    correct_eligible: Optional[int] = None
    eligible: Optional[int] = None
    known_hits: Optional[int] = None
    known_total: Optional[int] = None
    known_availability: Optional[float] = None
    avg_probability: Optional[float] = None
    baseline_most_frequent_accuracy: Optional[float] = None
    random_baseline_accuracy: Optional[float] = None


@dataclass(frozen=True)
class _NextVideoMeta:
    """Metadata describing the evaluation context for next-video metrics."""

    dataset: Optional[str] = None
    issue: Optional[str] = None
    issue_label: Optional[str] = None
    study_label: Optional[str] = None


@dataclass(frozen=True)
class NextVideoMetricSummary:
    """Grouped next-video metrics with compatibility accessors."""

    core: _NextVideoCore
    meta: _NextVideoMeta

    # Compatibility properties
    @property
    def accuracy(self) -> Optional[float]:  # pragma: no cover - forwarding
        return self.core.accuracy

    @property
    def coverage(self) -> Optional[float]:  # pragma: no cover
        return self.core.coverage

    @property
    def accuracy_eligible(self) -> Optional[float]:  # pragma: no cover
        return self.core.accuracy_eligible

    @property
    def evaluated(self) -> Optional[int]:  # pragma: no cover
        return self.core.evaluated

    @property
    def correct(self) -> Optional[int]:  # pragma: no cover
        return self.core.correct

    @property
    def correct_eligible(self) -> Optional[int]:  # pragma: no cover
        return self.core.correct_eligible

    @property
    def eligible(self) -> Optional[int]:  # pragma: no cover
        return self.core.eligible

    @property
    def known_hits(self) -> Optional[int]:  # pragma: no cover
        return self.core.known_hits

    @property
    def known_total(self) -> Optional[int]:  # pragma: no cover
        return self.core.known_total

    @property
    def known_availability(self) -> Optional[float]:  # pragma: no cover
        return self.core.known_availability

    @property
    def avg_probability(self) -> Optional[float]:  # pragma: no cover
        return self.core.avg_probability

    @property
    def baseline_most_frequent_accuracy(self) -> Optional[float]:  # pragma: no cover
        return self.core.baseline_most_frequent_accuracy

    @property
    def random_baseline_accuracy(self) -> Optional[float]:  # pragma: no cover
        return self.core.random_baseline_accuracy

    @property
    def dataset(self) -> Optional[str]:  # pragma: no cover
        return self.meta.dataset

    @property
    def issue(self) -> Optional[str]:  # pragma: no cover
        return self.meta.issue

    @property
    def issue_label(self) -> Optional[str]:  # pragma: no cover
        return self.meta.issue_label

    @property
    def study_label(self) -> Optional[str]:  # pragma: no cover
        return self.meta.study_label

    @classmethod
    def create(
        cls,
        *,
        accuracy: Optional[float] = None,
        coverage: Optional[float] = None,
        accuracy_eligible: Optional[float] = None,
        evaluated: Optional[int] = None,
        correct: Optional[int] = None,
        correct_eligible: Optional[int] = None,
        eligible: Optional[int] = None,
        known_hits: Optional[int] = None,
        known_total: Optional[int] = None,
        known_availability: Optional[float] = None,
        avg_probability: Optional[float] = None,
        baseline_most_frequent_accuracy: Optional[float] = None,
        random_baseline_accuracy: Optional[float] = None,
        dataset: Optional[str] = None,
        issue: Optional[str] = None,
        issue_label: Optional[str] = None,
        study_label: Optional[str] = None,
    ) -> "NextVideoMetricSummary":
        core = _NextVideoCore(
            accuracy=accuracy,
            coverage=coverage,
            accuracy_eligible=accuracy_eligible,
            evaluated=evaluated,
            correct=correct,
            correct_eligible=correct_eligible,
            eligible=eligible,
            known_hits=known_hits,
            known_total=known_total,
            known_availability=known_availability,
            avg_probability=avg_probability,
            baseline_most_frequent_accuracy=baseline_most_frequent_accuracy,
            random_baseline_accuracy=random_baseline_accuracy,
        )
        meta = _NextVideoMeta(
            dataset=dataset,
            issue=issue,
            issue_label=issue_label,
            study_label=study_label,
        )
        return cls(core=core, meta=meta)

