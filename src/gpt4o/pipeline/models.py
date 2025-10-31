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

"""Compatibility shim re-exporting GPT-4o pipeline models from ``common``.

Implements lazy attribute loading to avoid unused-import suppressions while
preserving the public surface.
"""

from __future__ import annotations

import importlib
from typing import Any, List

__all__: List[str] = [
    "SweepConfig",
    "SweepOutcome",
    "PipelinePaths",
    "coerce_float",
    "parse_config_label",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin re-export
    if name not in __all__:
        raise AttributeError(name)
    module = importlib.import_module("common.pipeline.gpt4o_models")
    return getattr(module, name)


def __dir__() -> List[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
