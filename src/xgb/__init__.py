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

import os

from common.import_utils import install_package_aliases
from common.package_baseline import BASELINE_PUBLIC_API, build_alias_map

# Allow lightweight imports that avoid bringing in CLI/pipeline dependencies
# during unit tests. When XGB_LIGHT_IMPORTS=1 only the minimal package surface is
# initialised and users can import `xgb.core.model` without transitively pulling
# `knn`.
if os.getenv("XGB_LIGHT_IMPORTS") == "1":  # pragma: no cover - import-time switch
    from . import core  # type: ignore
else:  # default behaviour
    from . import cli, core, pipeline, scripts  # type: ignore

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
