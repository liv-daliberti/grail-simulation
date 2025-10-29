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

"""Grail Simulation XGBoost baseline package.

Exposes CLI utilities, pipeline orchestration, and reporting helpers for the
slate-ranking and opinion-regression experiments."""

from __future__ import annotations

from common.import_utils import install_package_aliases
from common.package_baseline import BASELINE_PUBLIC_API, build_alias_map

from . import cli, core, pipeline, scripts

__all__ = list(BASELINE_PUBLIC_API)

_ALIAS_MODULES = build_alias_map(
    {
        "model": ".core.model",
        "vectorizers": ".core.vectorizers",
        "_optional": ".core._optional",
    }
)

install_package_aliases(__name__, _ALIAS_MODULES)

del install_package_aliases, _ALIAS_MODULES, BASELINE_PUBLIC_API, build_alias_map
