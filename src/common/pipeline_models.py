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


@dataclass(frozen=True)
class StudySpec:
    """

    Descriptor for a participant study and its associated issue.



    :ivar key: Attribute ``key``.

    :vartype key: str

    :ivar issue: Attribute ``issue``.

    :vartype issue: str

    :ivar label: Attribute ``label``.

    :vartype label: str

    """


    key: str
    issue: str
    label: str

    @property
    def study_slug(self) -> str:
        """

        Return a filesystem-safe slug for the study key.



        :returns: Result produced by ``study_slug``.

        :rtype: str

        """


        return self.key.replace(" ", "_")

    @property
    def issue_slug(self) -> str:
        """

        Return a filesystem-safe slug for the associated issue.



        :returns: Result produced by ``issue_slug``.

        :rtype: str

        """


        return self.issue.replace(" ", "_")

    @property
    def evaluation_slug(self) -> str:
        """

        Return the slug used for evaluation artefacts.



        :returns: Result produced by ``evaluation_slug``.

        :rtype: str

        """


        return f"{self.issue_slug}_{self.study_slug}"
