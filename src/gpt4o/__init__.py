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

"""Public exports for the GPT-4o baseline package.

Names are loaded lazily on first access to avoid import-time side effects.
"""

from __future__ import annotations

from pathlib import Path

import importlib

__all__ = ["run_eval", "pipeline_main"]

_CORE_PATH = Path(__file__).resolve().parent / "core"
if str(_CORE_PATH) not in __path__:
    __path__.append(str(_CORE_PATH))


def __getattr__(name: str):  # pragma: no cover - thin import proxy
    if name == "run_eval":
        return importlib.import_module("gpt4o.core.evaluate").run_eval
    if name == "pipeline_main":
        return importlib.import_module("gpt4o.pipeline").main
    raise AttributeError(name)
