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

"""Public exports for the prompt-builder utilities package."""

from __future__ import annotations

import importlib

from .formatters import clean_text
from .parsers import as_list_json, is_nanlike, secs, truthy
from .profiles import render_profile, synthesize_viewer_sentence
from .prompt import build_user_prompt
from .samples import PromptSample, generate_prompt_samples, write_samples_markdown

constants = importlib.import_module(".constants", __name__)
value_maps = importlib.import_module(".value_maps", __name__)

__all__ = [
    "as_list_json",
    "build_user_prompt",
    "clean_text",
    "is_nanlike",
    "generate_prompt_samples",
    "constants",
    "render_profile",
    "secs",
    "value_maps",
    "PromptSample",
    "synthesize_viewer_sentence",
    "truthy",
    "write_samples_markdown",
]
