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

"""Gun policy sentence builders for viewer profiles."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..constants import GUN_FIELD_LABELS
from ..formatters import clean_text
from ..parsers import format_yes_no, is_nanlike
from ..profile_helpers import sentencize
from ..shared import first_non_nan_value
from ..value_maps import format_field_value

GUN_BOOLEAN_PHRASES: Dict[str, Tuple[str, str]] = {
    "right_to_own_importance": (
        "considers the right to own a gun important",
        "considers the right to own a gun unimportant",
    ),
    "assault_ban": (
        "supports an assault weapons ban",
        "opposes an assault weapons ban",
    ),
    "handgun_ban": (
        "supports a handgun ban",
        "opposes a handgun ban",
    ),
    "concealed_safe": (
        "believes concealed carry is safe",
        "believes concealed carry is unsafe",
    ),
    "stricter_laws": (
        "supports stricter gun laws",
        "opposes stricter gun laws",
    ),
    "gun_enthusiasm": (
        "identifies as enthusiastic about guns",
        "does not identify as enthusiastic about guns",
    ),
    "gun_importance": (
        "considers gun policy important",
        "does not consider gun policy important",
    ),
}

GUN_METRIC_LABELS: Dict[str, str] = {
    "gun_index": "a gun index score around {value}",
    "gun_index_2": "an alternate gun index score around {value}",
    "assault_ban": "support for an assault weapons ban is about {value}",
}

GUN_GENERIC_LABELS: Dict[str, str] = {
    "gun_priority": "gun policy priority is {value}",
    "gun_policy": "gun policy stance is {value}",
    "gun_identity": "gun identity is {value}",
}


def _gun_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences summarising gun ownership and policy views."""

    gun_section: List[str] = []
    ownership_entry = _gun_ownership_entry(ex, selected)
    if ownership_entry:
        gun_section.append(ownership_entry)
    labeled_entries, known_keys = _gun_labeled_entries(ex, selected)
    gun_section.extend(labeled_entries)
    gun_section.extend(_gun_additional_entries(ex, known_keys))
    sentence = sentencize("Gun policy views include", gun_section)
    return [sentence] if sentence else []


def _gun_ownership_entry(ex: Dict[str, Any], selected: Dict[str, Any]) -> Optional[str]:
    """Return a phrase describing gun ownership status."""

    ownership = first_non_nan_value(ex, selected, "gun_own", "gunowner", "owns_gun")
    if ownership is None:
        return None
    ownership_flag = format_yes_no(ownership, yes="yes", no="no")
    mapping = {"yes": "owning a gun", "no": "not owning a gun"}
    mapped = mapping.get(ownership_flag or "")
    if mapped:
        return mapped
    ownership_text = format_field_value("gun_own", ownership) or clean_text(ownership) or ""
    lowered = ownership_text.lower()
    if lowered.startswith(("yes", "y")):
        return "owning a gun"
    if lowered.startswith(("no", "n")):
        return "not owning a gun"
    custom = clean_text(ownership)
    return custom if custom else None


def _gun_labeled_entries(
    ex: Dict[str, Any],
    selected: Dict[str, Any],
) -> tuple[List[str], set[str]]:
    """Return labelled gun-policy entries and the keys that were consumed."""

    entries: List[str] = []
    known_keys: set[str] = {"gun_own", "gunowner", "owns_gun"}
    for key, label in GUN_FIELD_LABELS.items():
        value = first_non_nan_value(ex, selected, key)
        if value is None:
            continue
        lower_key = key.lower()
        known_keys.add(lower_key)
        phrase = _render_gun_phrase(lower_key, label, value)
        if phrase:
            entries.append(phrase)
    return entries, known_keys


def _gun_additional_entries(ex: Dict[str, Any], known_keys: set[str]) -> List[str]:
    """Return additional gun-policy entries excluding previously seen keys."""

    entries: List[str] = []
    for key in sorted(ex.keys()):
        lower = key.lower()
        if not lower.startswith("gun_") or lower in known_keys:
            continue
        value = ex.get(key)
        if is_nanlike(value):
            continue
        phrase = _render_gun_phrase(lower, lower[4:].replace("_", " ").strip().title(), value)
        if not phrase:
            continue
        entries.append(phrase)
    return entries


def _render_gun_phrase(key: str, label: str, value: Any) -> Optional[str]:
    """Return a natural-language phrase for a gun-policy field."""

    yes_no = format_yes_no(value, yes="yes", no="no")
    if yes_no in {"yes", "no"}:
        phrase = _render_gun_boolean_phrase(key, yes_no == "yes")
        if phrase:
            return phrase
    formatted = format_field_value(key, value) or clean_text(value, limit=200)
    if not formatted:
        return None
    formatted = formatted.rstrip(".")
    metric_phrase = _render_gun_metric_phrase(key, formatted)
    if metric_phrase:
        return metric_phrase
    generic_phrase = _render_gun_generic_phrase(key, label, formatted)
    if generic_phrase:
        return generic_phrase
    if yes_no:
        return f"{label.lower()} is {yes_no}"
    return f"{label.lower()} is {formatted}"


def _render_gun_boolean_phrase(key: str, affirmative: bool) -> Optional[str]:
    """Return the boolean-oriented phrase for ``key`` if a template exists."""

    phrases = GUN_BOOLEAN_PHRASES.get(key)
    if not phrases:
        return None
    yes_phrase, no_phrase = phrases
    return yes_phrase if affirmative else no_phrase


def _render_gun_metric_phrase(key: str, formatted: str) -> Optional[str]:
    """Return the metric-oriented phrase for ``key`` if a template exists."""

    template = GUN_METRIC_LABELS.get(key)
    if not template:
        return None
    return template.format(value=formatted)


def _render_gun_generic_phrase(key: str, label: str, formatted: str) -> Optional[str]:
    """Fallback renderer for gun-policy fields without specialised templates."""

    template = GUN_GENERIC_LABELS.get(key)
    if template:
        return template.format(value=formatted)
    if key.startswith("gun_"):
        cleaned_label = label.lower()
        if cleaned_label.startswith("gun "):
            cleaned_label = cleaned_label[4:]
        return f"{cleaned_label} is {formatted}"
    return None
