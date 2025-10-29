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

"""Convenience exports for pipeline helpers plus compatibility aliases."""

from __future__ import annotations

from common.import_utils import install_package_aliases

from . import executor, formatters, io, models, stage, types, utils

__all__ = [
    "executor",
    "formatters",
    "io",
    "models",
    "stage",
    "types",
    "utils",
]

_ALIAS_MODULES = {
    "pipeline_executor": ".executor",
    "pipeline_formatters": ".formatters",
    "pipeline_io": ".io",
    "pipeline_models": ".models",
    "pipeline_stage": ".stage",
    "pipeline_types": ".types",
    "pipeline_utils": ".utils",
}

install_package_aliases(__name__, _ALIAS_MODULES)

del install_package_aliases, _ALIAS_MODULES
