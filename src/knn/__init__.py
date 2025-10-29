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

"""Grail Simulation KNN baseline package.

Exposes CLI utilities, pipeline coordination, and reporting helpers for
the slate-ranking and opinion-regression experiments."""

from __future__ import annotations

from common.import_utils import install_package_aliases

from . import cli, core, pipeline, scripts

__all__ = ["cli", "core", "pipeline", "scripts"]

_ALIAS_MODULES = {
    "data": ".core.data",
    "evaluate": ".core.evaluate",
    "features": ".core.features",
    "index": ".core.index",
    "opinion": ".core.opinion",
    "utils": ".core.utils",
    "cli_utils": ".cli.utils",
    "opinion_sweeps": ".pipeline.opinion_sweeps",
    "pipeline_cli": ".pipeline.cli",
    "pipeline_context": ".pipeline.context",
    "pipeline_data": ".pipeline.data",
    "pipeline_evaluate": ".pipeline.evaluate",
    "pipeline_io": ".pipeline.io",
    "pipeline_reports": ".pipeline.reports",
    "pipeline_sweeps": ".pipeline.sweeps",
    "pipeline_utils": ".pipeline.utils",
}

install_package_aliases(__name__, _ALIAS_MODULES)

del install_package_aliases, _ALIAS_MODULES
