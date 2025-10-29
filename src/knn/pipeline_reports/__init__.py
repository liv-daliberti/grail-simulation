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

"""Compatibility layer exposing the legacy ``knn.pipeline_reports`` API.

The production modules now live under :mod:`knn.pipeline.reports`.  The tests
and some downstream scripts still import from ``knn.pipeline_reports`` so we
re-export the public entry-points here to avoid churn while the package layout
settles.
"""

from __future__ import annotations

from knn.pipeline.reports import generate_reports  # type: ignore[F401]

__all__ = ["generate_reports"]
