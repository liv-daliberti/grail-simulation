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

"""Dataclasses and helpers supporting GPT-4o opinion evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import IO, List, Mapping, Sequence

from importlib import import_module
_common_opinion = import_module("common.opinion")
OpinionArtifacts = _common_opinion.OpinionArtifacts
OpinionEvaluationResult = _common_opinion.OpinionEvaluationResult
OpinionSpec = _common_opinion.OpinionSpec
OpinionStudyResult = _common_opinion.OpinionStudyResult
compute_opinion_metrics = _common_opinion.compute_opinion_metrics


@dataclass(frozen=True)
class OpinionFilters:
    """Filter configuration applied prior to evaluation."""

    issues: set[str]
    studies: set[str]

    def allows(self, issue: str, study: str) -> bool:
        """Return ``True`` when ``issue``/``study`` pass the configured filters.

        :param issue: Issue label for the current row.
        :param study: Participant study key for the current row.
        :returns: ``True`` if both fields match the active filter sets (or when
            the corresponding filter is empty).
        :rtype: bool
        """
        issue_key = issue.lower().strip() if issue else ""
        study_key = study.lower().strip() if study else ""
        if self.issues and issue_key not in self.issues:
            return False
        if self.studies and study_key not in self.studies:
            return False
        return True


@dataclass(frozen=True)
class OpinionRuntime:
    """Runtime configuration for GPT-4o opinion inference."""

    temperature: float
    max_tokens: int
    top_p: float | None
    deployment: str | None
    retries: int
    retry_delay: float


@dataclass(frozen=True)
class OpinionLimits:
    """Execution limits and flags applied during evaluation."""

    eval_max: int
    direction_tolerance: float
    overwrite: bool


@dataclass(frozen=True)
class OpinionSettings:
    """Resolved configuration shared across the evaluation run."""

    dataset_name: str
    cache_dir: str | None
    filters: OpinionFilters
    requested_specs: Sequence[str]
    limits: OpinionLimits
    runtime: OpinionRuntime


@dataclass
class CombinedAccumulator:
    """Accumulate per-study metrics for the combined summary."""

    truth_before: List[float]
    truth_after: List[float]
    pred_after: List[float]

    def extend(
        self,
        truth_before: Sequence[float],
        truth_after: Sequence[float],
        pred_after: Sequence[float],
    ) -> None:
        """Extend the accumulator with additional study-level vectors.

        :param truth_before: Pre-study opinion indices appended in-order.
        :param truth_after: Post-study opinion indices appended in-order.
        :param pred_after: Model-predicted post-study opinion indices.
        :returns: ``None``.
        """
        self.truth_before.extend(truth_before)
        self.truth_after.extend(truth_after)
        self.pred_after.extend(pred_after)

    def compute_metrics(self, direction_tolerance: float) -> Mapping[str, object]:
        """Return combined metrics using the accumulated vectors.

        :param direction_tolerance: Absolute delta treated as no change when
            computing direction accuracy.
        :returns: Metrics mapping matching :func:`common.opinion.compute_opinion_metrics`.
        :rtype: Mapping[str, object]
        """
        if not self.truth_after:
            return {}
        return compute_opinion_metrics(
            truth_after=self.truth_after,
            truth_before=self.truth_before,
            pred_after=self.pred_after,
            direction_tolerance=direction_tolerance,
        )


@dataclass
class StudyPredictionBatch:
    """Capture inference artefacts for a single study evaluation."""

    payloads: List[Mapping[str, object]]
    truth_before: List[float]
    truth_after: List[float]
    pred_after: List[float]


@dataclass(frozen=True)
class StudyMetricsPayload:
    """Bundle metrics and participant counts for artefact generation."""

    participants: int
    metrics: Mapping[str, object]
    baseline: Mapping[str, object]


@dataclass(frozen=True)
class CachedStudyPayload:
    """Cached metrics payload reconstructed from disk."""

    participants: int
    metrics: Mapping[str, object]
    baseline: Mapping[str, object]
    study_label: str
    issue: str


@dataclass(frozen=True)
class CachedPredictionVectors:
    """Cached prediction vectors reconstructed from disk."""

    truth_before: List[float]
    truth_after: List[float]
    pred_after: List[float]


@dataclass(frozen=True)
class QALogEntry:
    """Payload describing a single QA log record."""

    idx: int
    spec: OpinionSpec
    messages: Sequence[Mapping[str, object]]
    raw_output: str


@dataclass(frozen=True)
class ExampleProcessingContext:
    """Resources shared by per-example processing helpers."""

    qa_log: IO[str]
    batch: StudyPredictionBatch


__all__ = [
    "CachedPredictionVectors",
    "CachedStudyPayload",
    "CombinedAccumulator",
    "ExampleProcessingContext",
    "OpinionArtifacts",
    "OpinionEvaluationResult",
    "OpinionFilters",
    "OpinionLimits",
    "OpinionRuntime",
    "OpinionSettings",
    "OpinionStudyResult",
    "QALogEntry",
    "StudyMetricsPayload",
    "StudyPredictionBatch",
]
