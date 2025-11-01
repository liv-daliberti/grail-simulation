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

"""Generic selection containers shared across pipeline packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Mapping, Type, TypeVar, TYPE_CHECKING, cast

from .models import StudySpec


ConfigT = TypeVar("ConfigT")
OutcomeT = TypeVar("OutcomeT")


@dataclass
class BasePipelineSweepOutcome(Generic[ConfigT]):
    """
    Base metrics captured for a (study, configuration) pipeline sweep evaluation.

    :param order_index: Deterministic ordering index assigned to the task.
    :type order_index: int
    :param study: Study metadata associated with the sweep.
    :type study: ~common.pipeline.types.StudySpec
    :param config: Evaluated sweep configuration.
    :type config: ConfigT
    :param metrics_path: Filesystem path to the metrics artefact.
    :type metrics_path: Path
    :param metrics: Raw metrics payload loaded from disk.
    :type metrics: Mapping[str, object]
    """

    order_index: int
    study: "common.pipeline.types.StudySpec"
    config: ConfigT
    metrics_path: Path
    metrics: Mapping[str, object]


@dataclass
class StudySelection(Generic[OutcomeT]):
    """Container describing the selected outcome for a study.

    :param study: Study metadata that produced the selected outcome.
    :type study: ~common.pipeline.types.StudySpec
    :param outcome: Outcome object that contains the runnable configuration.
    :type outcome: OutcomeT
    """

    study: "common.pipeline.types.StudySpec"
    outcome: OutcomeT

    @property
    def config(self):  # type: ignore[override]
        """Expose the configuration associated with the selected outcome.

        :returns: Pipeline configuration stored on the selected outcome.
        :rtype: object
        """
        return self.outcome.config

    @property
    def evaluation_slug(self) -> str:
        """Return the evaluation slug derived from the selected study.

        :returns: Canonical evaluation slug associated with ``study``.
        :rtype: str
        """
        return self.study.evaluation_slug


@dataclass
class OpinionStudySelection(Generic[OutcomeT]):
    """Container describing the selected opinion outcome for a study.

    :param study: Study metadata that produced the selected outcome.
    :type study: ~common.pipeline.types.StudySpec
    :param outcome: Opinion outcome chosen for downstream processing.
    :type outcome: OutcomeT
    """

    study: "common.pipeline.types.StudySpec"
    outcome: OutcomeT

    @property
    def config(self):  # type: ignore[override]
        """Expose the configuration associated with the selected opinion outcome.

        :returns: Pipeline configuration stored on the selected opinion outcome.
        :rtype: object
        """
        return self.outcome.config


def narrow_opinion_selection(
    _outcome_type: Type[OutcomeT],
) -> type["OpinionStudySelection[OutcomeT]"]:
    """
    Return the :class:`OpinionStudySelection` specialised for ``OutcomeT``.

    Static type checkers understand the specialised generic returned from this helper,
    while at runtime the underlying class is reused without instantiating the generic.

    :param _outcome_type: Outcome type used purely for static analysis narrowing.
    :type _outcome_type: Type[OutcomeT]
    :returns: Opinion study selection type parameterised by ``OutcomeT``.
    :rtype: type[OpinionStudySelection[OutcomeT]]
    """

    if TYPE_CHECKING:
        return cast("type[OpinionStudySelection[OutcomeT]]", OpinionStudySelection)
    return OpinionStudySelection


__all__ = [
    "BasePipelineSweepOutcome",
    "OpinionStudySelection",
    "narrow_opinion_selection",
    "StudySelection",
    "StudySpec",
]
