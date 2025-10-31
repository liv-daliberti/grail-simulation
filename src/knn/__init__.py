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

# Attempt to import shared helpers; fall back to no-ops in lint-only contexts.
try:
    from common.import_utils import install_package_aliases  # type: ignore  # pylint: disable=import-error
    from common.package_baseline import BASELINE_PUBLIC_API, build_alias_map  # type: ignore  # pylint: disable=import-error
except ImportError:  # pragma: no cover - fallback for environments without src on sys.path
    # Provide lightweight fallbacks so static analysis doesn't fail on imports.
    def install_package_aliases(*_args, **_kwargs):  # type: ignore
        """No-op alias installer used during static analysis or lint-only runs."""
        return None

    BASELINE_PUBLIC_API = ()  # type: ignore

    def build_alias_map(mapping):  # type: ignore
        """Return the input mapping unchanged as a minimal alias map."""
        return mapping

from . import cli, core, pipeline, scripts

__all__ = list(BASELINE_PUBLIC_API)

_ALIAS_MODULES = build_alias_map(
    {
        "index": ".core.index",
        "cli_utils": ".cli.utils",
        "opinion_sweeps": ".pipeline.opinion_sweeps",
        "pipeline_data": ".pipeline.data",
        "pipeline_io": ".pipeline.io",
        "pipeline_reports": ".pipeline.reports",
        "pipeline_utils": ".pipeline.utils",
    }
)

install_package_aliases(__name__, _ALIAS_MODULES)

del install_package_aliases, _ALIAS_MODULES, BASELINE_PUBLIC_API, build_alias_map
