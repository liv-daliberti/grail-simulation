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

"""Next-video evaluation metrics data structures and helpers.

This module defines a small set of immutable dataclasses that organize
metrics commonly produced when evaluating "what is the next video" style
recommendation tasks. The public entry point is
:class:`NextVideoMetricSummary`, which groups metrics into logical buckets
and exposes compatibility properties so existing code that expects a flat
object interface can continue to work.

The design aims to:

- Provide clear, typed containers for metrics that are easy to document and
  render with Sphinx.
- Keep the objects immutable so that summaries can be safely passed around
  without accidental mutation.
- Offer a convenient :py:meth:`NextVideoMetricSummary.create` constructor that
  accepts either pre-built grouped dataclasses or a flat mapping of keyword
  arguments (useful for deserialization and quick tests).

Examples
--------

Create a summary from flat keyword arguments:

>>> from xgb.pipeline_context import NextVideoMetricSummary
>>> summary = NextVideoMetricSummary.create(
...     accuracy=0.42,
...     coverage=0.95,
...     evaluated=1000,
...     correct=420,
...     dataset="wage",
...     study_label="grpo-baseline",
... )
>>> float(summary.accuracy)
0.42
>>> int(summary.evaluated)
1000

The same summary can be constructed using the grouped dataclasses when
additional control is needed.
"""
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

from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass(frozen=True)
class _NextVideoRates:
    """Rate-based metrics for next-video evaluation.

    Attributes
    ----------
    accuracy : Optional[float]
        Top-1 accuracy across all evaluated examples. ``None`` if unknown.
    coverage : Optional[float]
        Proportion of targets for which the system produced a prediction.
    accuracy_eligible : Optional[float]
        Accuracy computed on the subset of eligible examples.
    avg_probability : Optional[float]
        Mean model probability assigned to the chosen next item.
    """

    accuracy: Optional[float] = None
    coverage: Optional[float] = None
    accuracy_eligible: Optional[float] = None
    avg_probability: Optional[float] = None


@dataclass(frozen=True)
class _NextVideoCounts:
    """Count-based metrics for next-video evaluation.

    Attributes
    ----------
    evaluated : Optional[int]
        Number of examples included in evaluation after filtering.
    correct : Optional[int]
        Number of correct next-item predictions among evaluated examples.
    correct_eligible : Optional[int]
        Correct predictions counted only over eligible examples.
    eligible : Optional[int]
        Number of examples considered eligible for evaluation.
    """

    evaluated: Optional[int] = None
    correct: Optional[int] = None
    correct_eligible: Optional[int] = None
    eligible: Optional[int] = None


@dataclass(frozen=True)
class _NextVideoKnown:
    """Known-candidate metrics for next-video evaluation.

    These metrics help characterize whether the correct target was even
    present among the considered candidates.

    Attributes
    ----------
    known_hits : Optional[int]
        Count of targets that appear within the candidate set.
    known_total : Optional[int]
        Total number of targets assessed for known-candidate availability.
    known_availability : Optional[float]
        Share of targets with at least one known candidate available.
    """

    known_hits: Optional[int] = None
    known_total: Optional[int] = None
    known_availability: Optional[float] = None


@dataclass(frozen=True)
class _NextVideoBaselines:
    """Baseline comparator metrics for next-video evaluation.

    Attributes
    ----------
    baseline_most_frequent_accuracy : Optional[float]
        Accuracy of a "most frequent" baseline recommender.
    random_baseline_accuracy : Optional[float]
        Accuracy of a uniform random baseline recommender.
    """

    baseline_most_frequent_accuracy: Optional[float] = None
    random_baseline_accuracy: Optional[float] = None


@dataclass(frozen=True)
class _NextVideoCore:
    """Core next-video metrics grouped by category.

    Attributes
    ----------
    rates : _NextVideoRates
        Rate-based metrics (e.g., accuracy, coverage).
    counts : _NextVideoCounts
        Raw count metrics (e.g., evaluated, correct).
    known : _NextVideoKnown
        Known-candidate availability metrics.
    baselines : _NextVideoBaselines
        Baseline comparator metrics for context and sanity checks.
    """

    rates: _NextVideoRates = field(default_factory=_NextVideoRates)
    counts: _NextVideoCounts = field(default_factory=_NextVideoCounts)
    known: _NextVideoKnown = field(default_factory=_NextVideoKnown)
    baselines: _NextVideoBaselines = field(default_factory=_NextVideoBaselines)


@dataclass(frozen=True)
class _NextVideoMeta:
    """Metadata describing the evaluation context for next-video metrics.

    Attributes
    ----------
    dataset : Optional[str]
        Dataset identifier or split name used for evaluation.
    issue : Optional[str]
        Issue identifier or key for grouping evaluations.
    issue_label : Optional[str]
        Human-readable label for the issue group.
    study_label : Optional[str]
        External study, sweep, or experiment label.
    """

    dataset: Optional[str] = None
    issue: Optional[str] = None
    issue_label: Optional[str] = None
    study_label: Optional[str] = None


@dataclass(frozen=True)
class NextVideoMetricSummary:
    """Grouped next-video metrics with compatibility accessors.

    This object intentionally exposes a flat property interface so that
    existing code which referenced attributes on older flat structures can
    transparently migrate without sweeping changes, while internally keeping
    metrics grouped for clarity and documentation purposes.

    Attributes
    ----------
    core : _NextVideoCore
        Grouped rate, count, known-candidate, and baseline metrics.
    meta : _NextVideoMeta
        Metadata describing the dataset, issue, and study labels.
    """

    core: _NextVideoCore
    meta: _NextVideoMeta

    # Compatibility properties
    @property
    def accuracy(self) -> Optional[float]:  # pragma: no cover - forwarding
        """Top-1 accuracy across evaluated examples.

        Returns
        -------
        Optional[float]
            Accuracy, or ``None`` if not provided.
        """
        return self.core.rates.accuracy

    @property
    def coverage(self) -> Optional[float]:  # pragma: no cover
        """Proportion of targets for which a prediction exists.

        Returns
        -------
        Optional[float]
            Coverage rate, or ``None`` if unknown.
        """
        return self.core.rates.coverage

    @property
    def accuracy_eligible(self) -> Optional[float]:  # pragma: no cover
        """Accuracy restricted to eligible examples only.

        Returns
        -------
        Optional[float]
            Accuracy on eligible subset, or ``None``.
        """
        return self.core.rates.accuracy_eligible

    @property
    def evaluated(self) -> Optional[int]:  # pragma: no cover
        """Number of examples included in evaluation.

        Returns
        -------
        Optional[int]
            Evaluated count, or ``None`` if not provided.
        """
        return self.core.counts.evaluated

    @property
    def correct(self) -> Optional[int]:  # pragma: no cover
        """Correct predictions among evaluated examples.

        Returns
        -------
        Optional[int]
            Count of correct predictions, or ``None``.
        """
        return self.core.counts.correct

    @property
    def correct_eligible(self) -> Optional[int]:  # pragma: no cover
        """Correct predictions measured over eligible examples only.

        Returns
        -------
        Optional[int]
            Count of correct predictions on the eligible subset, or ``None``.
        """
        return self.core.counts.correct_eligible

    @property
    def eligible(self) -> Optional[int]:  # pragma: no cover
        """Number of examples considered eligible for evaluation.

        Returns
        -------
        Optional[int]
            Eligible count, or ``None`` if unknown.
        """
        return self.core.counts.eligible

    @property
    def known_hits(self) -> Optional[int]:  # pragma: no cover
        """Targets that appear within the candidate set.

        Returns
        -------
        Optional[int]
            Known-candidate hit count, or ``None``.
        """
        return self.core.known.known_hits

    @property
    def known_total(self) -> Optional[int]:  # pragma: no cover
        """Total number of targets assessed for known-candidate availability.

        Returns
        -------
        Optional[int]
            Total known-candidate assessments, or ``None``.
        """
        return self.core.known.known_total

    @property
    def known_availability(self) -> Optional[float]:  # pragma: no cover
        """Share of targets with at least one known candidate available.

        Returns
        -------
        Optional[float]
            Known-candidate availability rate, or ``None``.
        """
        return self.core.known.known_availability

    @property
    def avg_probability(self) -> Optional[float]:  # pragma: no cover
        """Average model probability assigned to the selected next item.

        Returns
        -------
        Optional[float]
            Mean probability, or ``None`` if not computed.
        """
        return self.core.rates.avg_probability

    @property
    def baseline_most_frequent_accuracy(self) -> Optional[float]:  # pragma: no cover
        """Accuracy of a most-frequent-item baseline recommender.

        Returns
        -------
        Optional[float]
            Baseline accuracy, or ``None``.
        """
        return self.core.baselines.baseline_most_frequent_accuracy

    @property
    def random_baseline_accuracy(self) -> Optional[float]:  # pragma: no cover
        """Accuracy of a uniform-random baseline recommender.

        Returns
        -------
        Optional[float]
            Baseline accuracy, or ``None``.
        """
        return self.core.baselines.random_baseline_accuracy

    @property
    def dataset(self) -> Optional[str]:  # pragma: no cover
        """Dataset identifier or split name used for evaluation.

        Returns
        -------
        Optional[str]
            Dataset identifier, or ``None``.
        """
        return self.meta.dataset

    @property
    def issue(self) -> Optional[str]:  # pragma: no cover
        """Issue identifier or key for grouping evaluations.

        Returns
        -------
        Optional[str]
            Issue identifier, or ``None``.
        """
        return self.meta.issue

    @property
    def issue_label(self) -> Optional[str]:  # pragma: no cover
        """Human-readable label associated with the issue group.

        Returns
        -------
        Optional[str]
            Issue label, or ``None``.
        """
        return self.meta.issue_label

    @property
    def study_label(self) -> Optional[str]:  # pragma: no cover
        """External study, sweep, or experiment label.

        Returns
        -------
        Optional[str]
            Study label, or ``None``.
        """
        return self.meta.study_label

    @classmethod
    def create(
        cls,
        *,
        core: Optional[_NextVideoCore] = None,
        meta: Optional[_NextVideoMeta] = None,
        **flat: object,
    ) -> "NextVideoMetricSummary":
        """Construct a summary from grouped dataclasses or flat kwargs.

        This helper accepts either pre-constructed ``core`` and ``meta``
        dataclasses, or a flat mapping of keyword arguments whose keys match
        the fields of :class:`~xgb.pipeline.context.metrics_next_video._NextVideoCore`
        and :class:`~xgb.pipeline.context.metrics_next_video._NextVideoMeta`.

        Parameters
        ----------
        core : Optional[_NextVideoCore], optional
            Pre-built core metrics bundle to use as-is.
        meta : Optional[_NextVideoMeta], optional
            Pre-built meta information bundle to use as-is.
        flat : object
            Flat keyword arguments used to populate missing bundles.

        Returns
        -------
        NextVideoMetricSummary
            Newly constructed summary instance.
        """
        if core is None:
            # Build grouped dataclasses from flat kwargs so callers can remain
            # unaware of the internal grouping used to satisfy pylint limits.
            def _build(dc_type):
                names = {f.name for f in fields(dc_type)}
                return dc_type(**{k: flat.get(k) for k in names})  # type: ignore[misc]

            rates = _build(_NextVideoRates)
            counts = _build(_NextVideoCounts)
            known = _build(_NextVideoKnown)
            baselines = _build(_NextVideoBaselines)
            core = _NextVideoCore(
                rates=rates, counts=counts, known=known, baselines=baselines
            )
        if meta is None:
            meta_field_names = {f.name for f in fields(_NextVideoMeta)}
            meta_kwargs = {k: flat.get(k) for k in meta_field_names}
            meta = _NextVideoMeta(**meta_kwargs)  # type: ignore[arg-type]
        return cls(core=core, meta=meta)
