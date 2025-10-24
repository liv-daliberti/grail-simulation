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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .pipeline_models import StudySpec


OutcomeT = TypeVar("OutcomeT")


@dataclass
class StudySelection(Generic[OutcomeT]):
    """Container describing the selected outcome for a study.

    :param study: Study metadata that produced the selected outcome.
    :type study: StudySpec
    :param outcome: Outcome object that contains the runnable configuration.
    :type outcome: OutcomeT
    """

    study: StudySpec
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
    :type study: StudySpec
    :param outcome: Opinion outcome chosen for downstream processing.
    :type outcome: OutcomeT
    """

    study: StudySpec
    outcome: OutcomeT

    @property
    def config(self):  # type: ignore[override]
        """Expose the configuration associated with the selected opinion outcome.

        :returns: Pipeline configuration stored on the selected opinion outcome.
        :rtype: object
        """
        return self.outcome.config


__all__ = ["OpinionStudySelection", "StudySelection", "StudySpec"]
