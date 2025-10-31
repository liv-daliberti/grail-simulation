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

"""Public exports for the GPT-4o baseline package."""

from __future__ import annotations

from pathlib import Path

from .core.evaluate import run_eval  # noqa: F401
from .pipeline import main as pipeline_main  # noqa: F401

_CORE_PATH = Path(__file__).resolve().parent / "core"
if str(_CORE_PATH) not in __path__:
    __path__.append(str(_CORE_PATH))

__all__ = ["run_eval", "pipeline_main"]
