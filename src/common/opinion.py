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

"""Opinion study dataclasses and helpers shared by multiple pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, TypeVar


@dataclass(frozen=True)
class OpinionSpec:
    """Configuration describing one opinion-study index."""
    key: str
    issue: str
    label: str
    before_column: str
    after_column: str

    def build_example_kwargs(
        self,
        *,
        participant_id: str,
        document: str,
        before: float,
        after: float,
    ) -> dict[str, object]:
        """Return the standard keyword arguments for an ``OpinionExample``."""

        return opinion_example_kwargs(
            participant_id=participant_id,
            participant_study=self.key,
            issue=self.issue,
            document=document,
            before=before,
            after=after,
        )


@dataclass
class OpinionExample:
    """Collapsed participant-level prompt and opinion measurements."""
    participant_id: str
    participant_study: str
    issue: str
    document: str
    before: float
    after: float


@dataclass(frozen=True)
class OpinionExampleInputs:
    """Standardised collection of participant identifiers and opinion values."""

    participant_id: str
    document: str
    before: float
    after: float


def make_opinion_inputs(
    *,
    participant_id: str,
    document: str,
    before: float,
    after: float,
) -> OpinionExampleInputs:
    """
    Return a normalised :class:`OpinionExampleInputs` bundle.

    Creating the dataclass via a helper keeps pipeline implementations concise
    while ensuring duplicate code detected by pylint's similarity checker is
    consolidated in one location.
    """

    return OpinionExampleInputs(
        participant_id=participant_id,
        document=document,
        before=before,
        after=after,
    )


ExampleT = TypeVar("ExampleT")


@dataclass(frozen=True)
class OpinionCalibrationMetrics:  # pylint: disable=too-many-instance-attributes
    """Common calibration and baseline metrics shared across pipelines."""

    baseline_accuracy: Optional[float] = None
    accuracy_delta: Optional[float] = None
    calibration_slope: Optional[float] = None
    baseline_calibration_slope: Optional[float] = None
    calibration_intercept: Optional[float] = None
    baseline_calibration_intercept: Optional[float] = None
    calibration_ece: Optional[float] = None
    baseline_calibration_ece: Optional[float] = None
    kl_divergence_change: Optional[float] = None
    baseline_kl_divergence_change: Optional[float] = None
    participants: Optional[int] = None
    eligible: Optional[int] = None
    dataset: Optional[str] = None
    split: Optional[str] = None


def build_opinion_example(
    spec: OpinionSpec,
    *,
    factory: Callable[..., ExampleT],
    inputs: OpinionExampleInputs,
    **extra_fields: object,
) -> ExampleT:
    """
    Construct an opinion example instance using the provided factory callable.

    :param spec: Opinion specification describing the study identifiers.
    :type spec: OpinionSpec
    :param factory: Callable (usually a dataclass) used to instantiate the example.
    :type factory: Callable[..., ExampleT]
    :param inputs: Structured collection of participant identifiers and values.
    :type inputs: OpinionExampleInputs
    :param extra_fields: Additional keyword arguments forwarded to ``factory``.
    :type extra_fields: object
    :returns: Instance created by ``factory`` populated with shared fields.
    :rtype: ExampleT
    """

    base_kwargs = spec.build_example_kwargs(
        participant_id=inputs.participant_id,
        document=inputs.document,
        before=inputs.before,
        after=inputs.after,
    )
    return factory(**base_kwargs, **extra_fields)


def make_opinion_example(
    spec: OpinionSpec,
    inputs: OpinionExampleInputs,
    *,
    factory: Callable[..., ExampleT] = OpinionExample,
    **extra_fields: object,
) -> ExampleT:
    """
    Convenience wrapper around :func:`build_opinion_example`.

    Provides a lightweight helper so pipelines can override the output factory
    while reusing the standardised opinion inputs bundle.
    """

    return build_opinion_example(
        spec,
        factory=factory,
        inputs=inputs,
        **extra_fields,
    )


DEFAULT_SPECS: Tuple[OpinionSpec, ...] = (
    OpinionSpec(
        key="study1",
        issue="gun_control",
        label="Study 1 – Gun Control (MTurk)",
        before_column="gun_index",
        after_column="gun_index_2",
    ),
    OpinionSpec(
        key="study2",
        issue="minimum_wage",
        label="Study 2 – Minimum Wage (MTurk)",
        before_column="mw_index_w1",
        after_column="mw_index_w2",
    ),
    OpinionSpec(
        key="study3",
        issue="minimum_wage",
        label="Study 3 – Minimum Wage (YouGov)",
        before_column="mw_index_w1",
        after_column="mw_index_w2",
    ),
)


def float_or_none(value: Any) -> Optional[float]:
    """Return ``value`` as ``float`` when possible, otherwise ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def opinion_example_kwargs(  # pylint: disable=too-many-arguments
    *,
    participant_id: str,
    participant_study: str,
    issue: str,
    document: str,
    before: float,
    after: float,
) -> dict[str, object]:
    """

    Return keyword arguments common to opinion example dataclasses.



    :param participant_id: Value provided for ``participant_id``.

    :type participant_id: str

    :param participant_study: Value provided for ``participant_study``.

    :type participant_study: str

    :param issue: Value provided for ``issue``.

    :type issue: str

    :param document: Value provided for ``document``.

    :type document: str

    :param before: Value provided for ``before``.

    :type before: float

    :param after: Value provided for ``after``.

    :type after: float

    :returns: Result produced by ``opinion_example_kwargs``.

    :rtype: dict[str, object]

    """


    return {
        "participant_id": participant_id,
        "participant_study": participant_study,
        "issue": issue,
        "document": document,
        "before": before,
        "after": after,
    }


__all__ = [
    "DEFAULT_SPECS",
    "OpinionExample",
    "OpinionExampleInputs",
    "OpinionCalibrationMetrics",
    "OpinionSpec",
    "build_opinion_example",
    "make_opinion_example",
    "float_or_none",
    "opinion_example_kwargs",
]
