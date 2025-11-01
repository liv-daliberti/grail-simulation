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
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar
import logging


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
        """
        Return the standard keyword arguments for an ``OpinionExample``.

        :param participant_id: Participant identifier associated with the example.
        :param document: Prompt document presented to the participant.
        :param before: Pre-study opinion value.
        :param after: Post-study opinion value.
        :returns: Dictionary of keyword arguments suitable for
            :func:`~common.opinion.opinion_example_kwargs`.
        """

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
    Return a normalised :class:`~common.opinion.OpinionExampleInputs` bundle.

    Creating the dataclass via a helper keeps pipeline implementations concise
    while ensuring duplicate code detected by pylint's similarity checker is
    consolidated in one location.

    :param participant_id: Participant identifier associated with the example.
    :param document: Prompt document presented to the participant.
    :param before: Pre-study opinion value.
    :param after: Post-study opinion value.
    :returns: Dataclass encapsulating the participant identifiers and opinion scores.
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
    :type spec: ~common.opinion.OpinionSpec
    :param factory: Callable (usually a dataclass) used to instantiate the example.
    :type factory: Callable[..., ExampleT]
    :param inputs: Structured collection of participant identifiers and values.
    :type inputs: ~common.opinion.OpinionExampleInputs
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
    Convenience wrapper around :func:`~common.opinion.models.build_opinion_example`.

    Provides a lightweight helper so pipelines can override the output factory
    while reusing the standardised opinion inputs bundle.

    :param spec: Opinion specification describing the study identifiers.
    :param inputs: Normalised participant inputs produced by
        :func:`~common.opinion.models.make_opinion_inputs`.
    :param factory: Callable constructing the final example instance.
    :param extra_fields: Additional keyword arguments forwarded to ``factory``.
    :returns: Instance created by ``factory`` populated with shared fields.
    """

    return build_opinion_example(
        spec,
        factory=factory,
        inputs=inputs,
        **extra_fields,
    )


def make_opinion_example_from_values(
    spec: OpinionSpec,
    participant_id: str,
    document: str,
    scores: Tuple[float, float],
    *,
    factory: Callable[..., ExampleT] = OpinionExample,
    **extra_fields: object,
) -> ExampleT:
    """
    Construct an opinion example directly from raw participant values.

    Consolidates the recurring ``make_opinion_inputs`` + ``make_opinion_example``
    pattern across opinion pipelines into a single helper, eliminating duplicate
    code blocks flagged by pylint.

    :param scores: Tuple containing the ``before`` and ``after`` opinion scores.
    """

    before, after = scores
    inputs = make_opinion_inputs(
        participant_id=participant_id,
        document=document,
        before=before,
        after=after,
    )
    return make_opinion_example(
        spec,
        inputs,
        factory=factory,
        **extra_fields,
    )

def exclude_eval_participants(
    train_examples: Sequence[ExampleT],
    eval_examples: Sequence[ExampleT],
    *,
    logger: logging.Logger,
    study_key: str,
    prefix: str = "[OPINION]",
) -> List[ExampleT]:
    """
    Return ``train_examples`` filtered to remove participants in ``eval_examples``.

    Consolidating the overlap removal logic keeps pipelines free of duplicate
    list comprehensions that pylint otherwise flags under the duplicate-code
    checker.

    :param train_examples: Training examples collected for the study.
    :param eval_examples: Evaluation examples collected for the study.
    :param logger: Logger used for emitting overlap summaries.
    :param study_key: Study identifier included in the log message.
    :param prefix: Prefix inserted at the beginning of the log message.
    :returns: Filtered list of training examples without overlapping participants.
    """

    eval_participants = {
        getattr(example, "participant_id", None) for example in eval_examples
    }
    eval_participants.discard(None)
    if not eval_participants:
        return list(train_examples)

    filtered = [
        example
        for example in train_examples
        if getattr(example, "participant_id", None) not in eval_participants
    ]
    removed = len(train_examples) - len(filtered)
    if removed:
        logger.info(
            "%s Removed %d train participants overlapping validation for study=%s",
            prefix,
            removed,
            study_key,
        )
    return filtered


def ensure_train_examples(
    train_examples: Sequence[ExampleT],
    *,
    logger: logging.Logger,
    message: str,
    args: Sequence[object] | None = None,
) -> bool:
    """
    Return ``True`` when ``train_examples`` is non-empty, otherwise log a warning.

    Centralises the empty-train guard so opinion pipelines avoid duplicating
    identical warning blocks when overlap removal exhausts the training set.

    :param train_examples: Training examples collected for the study.
    :param logger: Logger used for emitting warning messages.
    :param message: Log message template passed to ``logging.Logger.warning``.
    :param args: Optional sequence of arguments interpolated into ``message``.
    :returns: ``True`` if ``train_examples`` contains elements, otherwise ``False``.
    """

    if train_examples:
        return True
    logger.warning(message, *(tuple(args) if args is not None else ()))
    return False


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


def log_participant_counts(
    logger: logging.Logger,
    *,
    study_key: str,
    train_count: int,
    eval_count: int,
    prefix: str = "[OPINION]",
) -> None:
    """
    Emit a standardised log line for train/eval participant counts.

    :param logger: Logger receiving the formatted message.
    :param study_key: Study identifier included in the log line.
    :param train_count: Number of participants observed in the training split.
    :param eval_count: Number of participants observed in the evaluation split.
    :param prefix: Log message prefix inserted before the structured fields.
    :returns: ``None``.
    """

    logger.info(
        "%s study=%s train_participants=%d eval_participants=%d",
        prefix,
        study_key,
        train_count,
        eval_count,
    )



def float_or_none(value: Any) -> Optional[float]:
    """
    Return ``value`` as ``float`` when possible, otherwise ``None``.

    :param value: Candidate numeric value.
    :returns: Float representation or ``None`` when parsing fails or value is NaN.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
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
    "make_opinion_inputs",
    "make_opinion_example",
    "float_or_none",
    "opinion_example_kwargs",
]
