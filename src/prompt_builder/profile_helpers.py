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

"""Helper utilities shared across the profile rendering pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

from .formatters import clean_text
from .parsers import is_nanlike
from .shared import load_selected_survey_row
from .value_maps import format_field_value


def load_selected_row(example: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract the ``selected_survey_row`` payload as a dictionary.

    :param example: Dataset entry containing the optional survey row metadata.
    :type example: Mapping[str, Any]
    :returns: Normalised mapping of the selected survey row, or an empty dict.
    :rtype: Dict[str, Any]
    """

    return load_selected_survey_row(example)


def ensure_sentence(text: str) -> str:
    """
    Ensure that ``text`` terminates with sentence punctuation.

    :param text: Candidate phrase to normalise.
    :type text: str
    :returns: Sentence-terminated text or an empty string when input is empty.
    :rtype: str
    """

    stripped = (text or "").strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


def clean_fragment(text: str) -> str:
    """
    Remove leading/trailing whitespace and trailing periods from ``text``.

    :param text: Fragment to normalise.
    :type text: str
    :returns: Cleaned fragment string.
    :rtype: str
    """

    return (text or "").strip().rstrip(".")


def first_text(
    example: Mapping[str, Any],
    selected: Mapping[str, Any],
    *keys: str,
    limit: Optional[int] = None,
) -> str:
    """
    Return the first non-empty textual value matched across ``keys``.

    :param example: Primary dataset mapping.
    :type example: Mapping[str, Any]
    :param selected: Selected survey-row mapping.
    :type selected: Mapping[str, Any]
    :param keys: Ordered field names to inspect.
    :type keys: str
    :param limit: Optional maximum character count passed to :func:`clean_text`.
    :type limit: int, optional
    :returns: Cleaned text value or an empty string when nothing is found.
    :rtype: str
    """

    for key in keys:
        for dataset in (example, selected):
            if key not in dataset:
                continue
            value = dataset.get(key)
            if value is None or is_nanlike(value):
                continue
            formatted = format_field_value(key, value) or str(value)
            cleaned = clean_text(formatted, limit=limit)
            if cleaned:
                return cleaned
    return ""


def phrases_from_items(items: Sequence[str]) -> List[str]:
    """
    Convert a sequence of ``label: value`` strings into readable phrases.

    :param items: Entries combining labels and values.
    :type items: Sequence[str]
    :returns: List of single phrases suitable for sentence construction.
    :rtype: List[str]
    """

    phrases: List[str] = []
    for item in items:
        fragment = item.strip()
        if not fragment:
            continue
        if ":" not in fragment:
            phrases.append(fragment)
            continue
        label, value = fragment.split(":", 1)
        label_clean = label.strip()
        value_clean = value.strip()
        if not label_clean and not value_clean:
            continue
        if not value_clean:
            phrases.append(label_clean)
            continue
        prefix = label_clean
        if prefix and prefix[1:].lower() == prefix[1:]:
            prefix = prefix[0].lower() + prefix[1:]
        label_lower = label_clean.lower()
        value_lower = value_clean.lower()
        if label_lower and label_lower in value_lower:
            phrases.append(value_clean)
        elif value_lower in {"yes", "no"}:
            phrases.append(f"{prefix} {value_lower}")
        else:
            phrases.append(f"{prefix} is {value_clean}")
    return phrases


def sentencize(prefix: str, items: Sequence[str]) -> str:
    """
    Join ``items`` into a grammatically correct sentence prefixed by ``prefix``.

    :param prefix: Leading phrase preceding the list.
    :type prefix: str
    :param items: Sequence of phrase fragments.
    :type items: Sequence[str]
    :returns: Sentence string or empty string when ``items`` is empty.
    :rtype: str
    """

    phrases = phrases_from_items(items)
    if not phrases:
        return ""
    if len(phrases) == 1:
        return f"{prefix} {phrases[0]}."
    return f"{prefix} {', '.join(phrases[:-1])}, and {phrases[-1]}."


def collect_labeled_fields(
    example: Mapping[str, Any],
    selected: Mapping[str, Any],
    specs: Sequence[Tuple[Sequence[str], str]],
) -> List[str]:
    """
    Collect labelled phrases based on field specifications.

    :param example: Primary dataset mapping.
    :type example: Mapping[str, Any]
    :param selected: Selected survey-row mapping.
    :type selected: Mapping[str, Any]
    :param specs: Sequence of ``(field_names, label)`` pairs.
    :type specs: Sequence[Tuple[Sequence[str], str]]
    :returns: List of ``label: value`` strings where values are present.
    :rtype: List[str]
    """

    entries: List[str] = []
    for keys, label in specs:
        value = first_text(example, selected, *keys)
        if value:
            entries.append(f"{label}: {value}")
    return entries


def first_available_text(
    example: Mapping[str, Any],
    selected: Mapping[str, Any],
    keys: Sequence[str],
    *,
    limit: Optional[int] = 220,
) -> str:
    """
    Return the first non-empty text value across ``keys`` with optional length cap.

    :param example: Primary dataset mapping.
    :type example: Mapping[str, Any]
    :param selected: Selected survey-row mapping.
    :type selected: Mapping[str, Any]
    :param keys: Sequence of candidate field names.
    :type keys: Sequence[str]
    :param limit: Optional maximum character count passed to :func:`clean_text`.
    :type limit: int, optional
    :returns: Cleaned text or an empty string.
    :rtype: str
    """

    for key in keys:
        text = first_text(example, selected, key, limit=limit)
        if text:
            return text
    return ""


__all__ = [
    "clean_fragment",
    "collect_labeled_fields",
    "ensure_sentence",
    "first_available_text",
    "first_text",
    "load_selected_row",
    "phrases_from_items",
    "sentencize",
]
