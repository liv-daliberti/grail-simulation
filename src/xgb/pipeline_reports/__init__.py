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

"""Compatibility layer exposing the legacy ``xgb.pipeline_reports`` API.

The production modules now live under :mod:`xgb.pipeline.reports`.  Tests and
downstream scripts still rely on the legacy import paths, so we re-export the
public entry points here to ease the transition to the reorganised package.
"""

from __future__ import annotations

from ..pipeline.reports import *  # type: ignore[F401,F403]
from ..pipeline.reports import __all__ as _REPORTS_ALL  # type: ignore[F401]

__all__ = list(_REPORTS_ALL)
