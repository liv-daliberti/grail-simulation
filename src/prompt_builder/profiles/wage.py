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

"""Minimum wage sentence builders for viewer profiles."""

from __future__ import annotations

from typing import Any, Dict, List

from ..constants import MIN_WAGE_FIELD_LABELS
from ..formatters import clean_text
from ..parsers import format_yes_no, is_nanlike
from ..profile_helpers import sentencize
from ..shared import first_non_nan_value
from ..value_maps import format_field_value


def _wage_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences covering minimum-wage and wage policy attitudes."""

    wage_section: List[str] = []
    for key, label in MIN_WAGE_FIELD_LABELS.items():
        value = first_non_nan_value(ex, selected, key)
        if value is None:
            continue
        text = format_yes_no(value)
        if text is None:
            text = format_field_value(key, value) or clean_text(value, limit=220)
        if text:
            wage_section.append(f"{label}: {text}")
    known_keys = {k.lower() for k in MIN_WAGE_FIELD_LABELS}
    for key in sorted(ex.keys()):
        lower = key.lower()
        if lower in known_keys or not (lower.startswith("minwage") or lower.startswith("mw_")):
            continue
        value = ex.get(key)
        if is_nanlike(value):
            continue
        text = format_field_value(key, value) or clean_text(value, limit=220)
        if not text:
            continue
        label = (
            lower.replace("minwage", "minimum wage")
            .replace("mw_", "minimum wage ")
            .replace("_", " ")
            .strip()
        )
        if label:
            wage_section.append(f"{label.capitalize()}: {text}")
        known_keys.add(lower)
    sentence = sentencize("Minimum wage views include", wage_section)
    return [sentence] if sentence else []
