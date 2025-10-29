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

"""Thin wrapper that mirrors the legacy prompt statistics entry point.

Historically the prompt reporting lived in this script; during the module
split we kept the file so downstream tooling could continue invoking it.
The implementation now simply delegates to :mod:`clean_data.prompt.cli`.
All usage is covered by the repository's Apache 2.0 license; see LICENSE
for the exact terms.
"""

from __future__ import annotations

from clean_data.prompt import generate_prompt_feature_report, main

__all__ = ["generate_prompt_feature_report", "main"]

if __name__ == "__main__":
    main()
