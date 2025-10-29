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

"""Shared dataclasses representing participant studies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StudySpec:
    """Participant study metadata shared across pipeline packages."""


    key: str
    issue: str
    label: str

    @property
    def study_slug(self) -> str:
        """Return a filesystem-safe slug for the study key."""


        return self.key.replace(" ", "_")

    @property
    def issue_slug(self) -> str:
        """Return a filesystem-safe slug for the associated issue."""


        return self.issue.replace(" ", "_")

    @property
    def evaluation_slug(self) -> str:
        """Return the slug used when saving evaluation artefacts."""


        return f"{self.issue_slug}_{self.study_slug}"
