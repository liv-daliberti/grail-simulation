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

"""Shared helpers for parsing survey metadata in prompt generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Dict, Optional

from .parsers import is_nanlike


def load_selected_survey_row(example: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract and normalise the ``selected_survey_row`` mapping from ``example``.

    :param example: Dataset record potentially containing survey metadata.
    :type example: Mapping[str, Any]
    :returns: Dictionary representation of ``selected_survey_row`` or an empty dict.
    :rtype: Dict[str, Any]
    """

    return normalise_selected_survey_row(example.get("selected_survey_row"))


def normalise_selected_survey_row(raw: Any) -> Dict[str, Any]:
    """
    Convert ``selected_survey_row`` payloads into plain dictionaries.

    Handles inline mappings, JSON-encoded strings, and Arrow scalars that expose
    an ``as_py`` method. Falls back to an empty dictionary when parsing fails.

    :param raw: Raw ``selected_survey_row`` payload.
    :type raw: Any
    :returns: Dictionary representation of ``raw`` or an empty dict on failure.
    :rtype: Dict[str, Any]
    """

    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    as_py = getattr(raw, "as_py", None)
    if callable(as_py):
        try:
            candidate = as_py()
        except (TypeError, ValueError):
            return {}
        if isinstance(candidate, dict):
            return candidate
    return {}


def first_non_nan_value(
    example: Dict[str, Any],
    selected: Dict[str, Any],
    *keys: str,
) -> Optional[Any]:
    """
    Return the first value from ``example`` or ``selected`` that is not NaN-like.

    :param example: Primary dataset row containing viewer or survey metadata.
    :type example: Dict[str, Any]
    :param selected: Secondary mapping representing ``selected_survey_row`` fields.
    :type selected: Dict[str, Any]
    :param keys: Ordered candidate field names to inspect in both mappings.
    :type keys: str
    :returns: The first value that is neither ``None`` nor
        :func:`prompt_builder.parsers.is_nanlike`.
    :rtype: Optional[Any]
    """

    for key in keys:
        if key in example:
            value = example[key]
            if value is not None and not is_nanlike(value):
                return value
        if key in selected:
            value = selected.get(key)
            if value is not None and not is_nanlike(value):
                return value
    return None
