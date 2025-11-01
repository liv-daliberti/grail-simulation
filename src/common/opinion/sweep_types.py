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

"""Common opinion sweep data structures shared across pipeline packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Mapping, Optional, Tuple, TypeVar



ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class BaseSweepTask(Generic[ConfigT]):  # pylint: disable=too-many-instance-attributes
    """Describe shared attributes for sweep execution tasks.

    :param index: Stable ordinal used to preserve submission order.
    :type index: int
    :param study: Study metadata associated with the task.
    :type study: ~common.pipeline.types.StudySpec
    :param config: Pipeline-specific hyper-parameter configuration.
    :type config: ConfigT
    :param base_cli: Baseline CLI arguments reused across tasks.
    :type base_cli: Tuple[str, ...]
    :param extra_cli: Additional passthrough CLI arguments.
    :type extra_cli: Tuple[str, ...]
    :param run_root: Directory where sweep artefacts are stored.
    :type run_root: Path
    :param metrics_path: Expected location of the metrics artefact.
    :type metrics_path: Path
    :param train_participant_studies: Optional set of study keys used to restrict
        training to a subset of participants (e.g., within-study only). Defaults
        to an empty tuple which many pipelines interpret as within-study.
    :type train_participant_studies: Tuple[str, ...]
    """

    __module__ = "common.opinion"

    index: int
    study: "common.pipeline.types.StudySpec"
    config: ConfigT
    base_cli: Tuple[str, ...]
    extra_cli: Tuple[str, ...]
    run_root: Path
    metrics_path: Path
    train_participant_studies: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BaseOpinionSweepTask(Generic[ConfigT]):
    """Describe shared attributes for opinion sweep execution tasks.

    :param index: Stable ordinal used to preserve submission order.
    :type index: int
    :param study: Study metadata associated with the task.
    :type study: ~common.pipeline.types.StudySpec
    :param config: Pipeline-specific hyper-parameter configuration.
    :type config: ConfigT
    :param metrics_path: Expected location of the metrics artefact.
    :type metrics_path: Path
    """

    __module__ = "common.opinion"

    index: int
    study: "common.pipeline.types.StudySpec"
    config: ConfigT
    metrics_path: Path


@dataclass(frozen=True)
class MetricsArtifact:
    """Container bundling the on-disk metrics artefact and its payload."""

    __module__ = "common.opinion"

    path: Path
    payload: Mapping[str, object]


@dataclass(frozen=True)
class AccuracySummary:
    """Capture directional accuracy metrics emitted by sweep evaluations."""

    __module__ = "common.opinion"

    value: Optional[float] = None
    baseline: Optional[float] = None
    delta: Optional[float] = None
    eligible: Optional[int] = None


@dataclass
class BaseOpinionSweepOutcome(Generic[ConfigT]):
    """Metrics captured for a (study, configuration) opinion sweep evaluation.

    :param order_index: Deterministic ordering index assigned to the task.
    :type order_index: int
    :param study: Study metadata associated with the sweep.
    :type study: ~common.pipeline.types.StudySpec
    :param config: Evaluated sweep configuration.
    :type config: ConfigT
    :param mae: Mean absolute error achieved by the configuration.
    :type mae: float
    :param rmse: Root mean squared error achieved by the configuration.
    :type rmse: float
    :param artifact: Metrics artefact containing the payload and storage path.
    :type artifact: MetricsArtifact
    :param accuracy_summary: Summary of directional accuracy metrics.
    :type accuracy_summary: AccuracySummary
    """

    __module__ = "common.opinion"

    order_index: int
    study: "common.pipeline.types.StudySpec"
    config: ConfigT
    mae: float
    rmse: float
    artifact: MetricsArtifact
    accuracy_summary: AccuracySummary

    @property
    def metrics_path(self) -> Path:
        """Return the filesystem path to the persisted metrics artefact."""

        return self.artifact.path

    @property
    def metrics(self) -> Mapping[str, object]:
        """Return the raw metrics payload loaded from disk."""

        return self.artifact.payload

    @property
    def accuracy(self) -> Optional[float]:
        """Directional accuracy achieved by the configuration."""

        return self.accuracy_summary.value

    @property
    def baseline_accuracy(self) -> Optional[float]:
        """Directional accuracy achieved by the baseline configuration."""

        return self.accuracy_summary.baseline

    @property
    def accuracy_delta(self) -> Optional[float]:
        """Improvement in accuracy over the baseline configuration."""

        return self.accuracy_summary.delta

    @property
    def eligible(self) -> Optional[int]:
        """Number of evaluation examples contributing to accuracy metrics."""

        return self.accuracy_summary.eligible


SWEEP_PUBLIC = (
    "AccuracySummary",
    "BaseOpinionSweepOutcome",
    "BaseOpinionSweepTask",
    "BaseSweepTask",
    "MetricsArtifact",
)

__all__ = list(SWEEP_PUBLIC)
