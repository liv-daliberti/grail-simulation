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

"""Shared helpers for baseline package initialisation modules."""

from __future__ import annotations

from typing import Mapping

BASELINE_PUBLIC_API = ("cli", "core", "pipeline", "scripts")

_COMMON_ALIAS_MODULES: Mapping[str, str] = {
    "data": ".core.data",
    "evaluate": ".core.evaluate",
    "features": ".core.features",
    "opinion": ".core.opinion",
    "utils": ".core.utils",
    "pipeline_cli": ".pipeline.cli",
    "pipeline_context": ".pipeline.context",
    "pipeline_evaluate": ".pipeline.evaluate",
    "pipeline_sweeps": ".pipeline.sweeps",
}


def build_alias_map(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """
    Merge the shared alias map with package-specific overrides.

    :param extra: Additional alias definitions unique to the baseline package.
    :returns: Combined alias dictionary safe to mutate by callers.
    """

    merged = dict(_COMMON_ALIAS_MODULES)
    if extra:
        merged.update(extra)
    return merged


__all__ = ["BASELINE_PUBLIC_API", "build_alias_map"]
