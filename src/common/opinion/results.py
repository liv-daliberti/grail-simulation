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

"""Shared opinion evaluation artefact and result dataclasses."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import OpinionSpec


@dataclass(frozen=True)
class OpinionArtifacts:
    """Filesystem artefacts generated while evaluating an opinion study."""

    metrics: Path
    predictions: Path
    qa_log: Path


@dataclass(frozen=True)
class OpinionStudyResult:  # pylint: disable=too-many-instance-attributes
    """Normalised payload capturing per-study opinion evaluation outputs."""

    study_key: str
    study_label: str
    issue: str
    participants: int
    eligible: int
    metrics: Mapping[str, object]
    baseline: Mapping[str, object]
    artifacts: OpinionArtifacts
    spec: "OpinionSpec | None" = None

    @property
    def metrics_path(self) -> Path:
        """Return the metrics JSON artefact path."""
        return self.artifacts.metrics

    @property
    def predictions_path(self) -> Path:
        """Return the predictions JSONL artefact path."""
        return self.artifacts.predictions

    @property
    def qa_log_path(self) -> Path:
        """Return the QA log artefact path."""
        return self.artifacts.qa_log


@dataclass(frozen=True)
class OpinionEvaluationResult:
    """
    Aggregated opinion evaluation payload combining per-study metrics.

    The constructor accepts either a mapping keyed by study identifier or a
    sequence of :class:`OpinionStudyResult` instances. Internally results are
    normalised to an ordered mapping to provide consistent access patterns.
    """

    _studies: Mapping[str, OpinionStudyResult]
    combined_metrics: Mapping[str, object]
    config_label: str | None = None

    def __init__(
        self,
        *,
        studies: Mapping[str, OpinionStudyResult] | Sequence[OpinionStudyResult] | None = None,
        combined_metrics: Mapping[str, object] | None = None,
        config_label: str | None = None,
    ) -> None:
        normalised = self._normalise_studies(studies)
        object.__setattr__(self, "_studies", normalised)
        object.__setattr__(self, "combined_metrics", dict(combined_metrics or {}))
        object.__setattr__(self, "config_label", config_label)

    @staticmethod
    def _normalise_studies(
        studies: Mapping[str, OpinionStudyResult] | Sequence[OpinionStudyResult] | None,
    ) -> Mapping[str, OpinionStudyResult]:
        """Return an ordered mapping keyed by study identifier."""
        if not studies:
            return OrderedDict()
        if isinstance(studies, Mapping):
            return OrderedDict(studies.items())
        ordered: "OrderedDict[str, OpinionStudyResult]" = OrderedDict()
        for study in studies:
            ordered[study.study_key] = study
        return ordered

    @property
    def studies(self) -> Mapping[str, OpinionStudyResult]:
        """Mapping of study key to the corresponding evaluation result."""
        return self._studies

    def get(self, key: str, default: OpinionStudyResult | None = None) -> OpinionStudyResult | None:
        """Return the study result for ``key`` if present."""
        return self._studies.get(key, default)

    def values(self):
        """Convenience helper mirroring mapping ``values``."""
        return self._studies.values()

    def items(self):
        """Convenience helper mirroring mapping ``items``."""
        return self._studies.items()

    def keys(self):
        """Convenience helper mirroring mapping ``keys``."""
        return self._studies.keys()

    def __len__(self) -> int:
        return len(self._studies)

    def __bool__(self) -> bool:
        return bool(self._studies)
