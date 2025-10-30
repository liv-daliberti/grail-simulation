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

"""Prompt snippets shared across pipeline configurations."""

from __future__ import annotations

STRICT_NUMBERED_ANSWER_GUIDE: str = (
    "\n"
    "Examples of valid <answer>:\n"
    "  <think>\n"
    "  WHY YOU THINK THIS IS THE RIGHT CHOICE\n"
    "  </think>\n"
    "  <answer>\n"
    "  3\n"
    "  </answer>\n"
    "\n"
    "Examples of INVALID <answer> (never do these):\n"
    "  <think></think><answer>3.</answer>                 ← trailing period\n"
    "  <think></think><answer>\"3\"</answer>                ← quoted\n"
    "  <think></think><answer>Option 3</answer>           ← extra words\n"
    "  <think></think><answer>Parkland …</answer>         ← name instead of number\n"
    "  You only have 100 tokens to think and 50 tokens to answer.\n"
)


__all__ = ["STRICT_NUMBERED_ANSWER_GUIDE"]
