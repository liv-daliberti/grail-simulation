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

"""Demographic sentence builders for viewer profiles."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..formatters import (
    clean_text,
    describe_age_fragment,
    describe_gender_fragment,
    human_join,
    normalize_language_text,
    with_indefinite_article,
)
from ..parsers import format_yes_no
from ..profile_helpers import clean_fragment, ensure_sentence, first_text
from ..shared import first_non_nan_value
from ..value_maps import format_field_value


def _demographic_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Assemble demographic sentences covering identity, location, and status."""

    sentences: List[str] = []
    sentences.extend(_identity_sentences(ex, selected))
    sentences.extend(_location_sentences(ex, selected))
    sentences.extend(_education_sentences(ex, selected))
    sentences.extend(_income_sentences(ex, selected))
    sentences.extend(_employment_sentences(ex, selected))
    sentences.extend(_family_sentences(ex, selected))
    sentences.extend(_religion_sentences(ex, selected))
    sentences.extend(_language_sentences(ex, selected))
    return sentences


def _identity_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Build sentences describing age, gender, and race identity fragments."""

    sentences: List[str] = []
    age_fragment = describe_age_fragment(
        first_non_nan_value(ex, selected, "age", "age_cat", "age_category")
    )
    gender_fragment = describe_gender_fragment(
        first_non_nan_value(ex, selected, "gender", "q26", "gender4", "gender_3_text")
    )
    race_fragment = first_text(ex, selected, "race", "race_ethnicity", "ethnicity", "q29")

    identity_bits: List[str] = []
    if age_fragment:
        lowered = age_fragment.lower()
        if lowered.startswith(("between", "at least", "at most", "under", "over")):
            identity_bits.append(age_fragment)
        else:
            identity_bits.append(with_indefinite_article(age_fragment))
    if gender_fragment:
        gender_clean = clean_fragment(gender_fragment)
        mapped = {"male": "man", "female": "woman"}
        gender_text = mapped.get(gender_clean.lower(), gender_clean)
        if not gender_text.lower().startswith(("a ", "an ", "the ", "someone", "somebody")):
            gender_text = with_indefinite_article(gender_text)
        identity_bits.append(gender_text)
    if identity_bits:
        sentences.append(
            ensure_sentence(f"This viewer is {' and '.join(identity_bits)}.")
        )
    elif gender_fragment:
        sentences.append(
            ensure_sentence(
                f"This viewer identifies as {clean_fragment(gender_fragment)}."
            )
        )
    elif race_fragment:
        sentences.append(
            ensure_sentence(
                f"This viewer identifies as {clean_fragment(race_fragment)}."
            )
        )
    if race_fragment and identity_bits:
        sentences.append(
            ensure_sentence(f"They identify as {clean_fragment(race_fragment)}.")
        )
    return [s for s in sentences if s]


def _location_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences summarising the viewer's location context."""

    sentences: List[str] = []
    city = first_text(ex, selected, "city", "city_name")
    state = first_text(
        ex,
        selected,
        "state",
        "state_residence",
        "state_name",
        "inputstate",
        "thumb_states",
        "state_full",
    )
    county = first_text(ex, selected, "county", "county_name")
    zip_text = first_text(ex, selected, "zip3", "zip", "zip5")
    location = ""
    if city and state:
        location = f"{clean_fragment(city)}, {clean_fragment(state)}"
    elif city:
        location = clean_fragment(city)
    elif state:
        location = clean_fragment(state)
    extras: List[str] = []
    if county:
        extras.append(f"{clean_fragment(county)} County")
    if zip_text:
        extras.append(f"ZIP3 {clean_fragment(zip_text)}")
    if location:
        if extras:
            location = f"{location} ({'; '.join(extras)})"
        sentences.append(ensure_sentence(f"They live in {location}."))
    elif extras:
        sentences.append(
            ensure_sentence(f"They live in {human_join(extras)}.")
        )
    return [s for s in sentences if s]


def _education_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences summarising educational attainment."""

    sentences: List[str] = []
    education = first_text(
        ex,
        selected,
        "education",
        "educ",
        "education_level",
        "college_desc",
        "highest_education",
        "education_text",
    )
    college_flag = format_yes_no(
        first_non_nan_value(ex, selected, "college", "college_grad"),
        yes="yes",
        no="no",
    )
    if education:
        fragment = clean_fragment(education)
        if college_flag == "yes":
            sentences.append(
                ensure_sentence(
                    f"They have {fragment} and are college educated."
                )
            )
        elif college_flag == "no":
            sentences.append(
                ensure_sentence(
                    f"They have {fragment} and are not college educated."
                )
            )
        else:
            sentences.append(ensure_sentence(f"They have {fragment}."))
    elif college_flag == "yes":
        sentences.append(ensure_sentence("They are college educated."))
    elif college_flag == "no":
        sentences.append(ensure_sentence("They are not college educated."))
    return [s for s in sentences if s]


def _income_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences describing household income details."""

    sentences: List[str] = []
    income = first_text(
        ex,
        selected,
        "q31",
        "income",
        "household_income",
        "income_bracket",
        "income_cat",
    )
    if income:
        sentences.append(
            ensure_sentence(
                f"Their household income is reported as {clean_fragment(income)}."
            )
        )
        return sentences
    income_flag = format_yes_no(
        first_non_nan_value(ex, selected, "income_gt50k"),
        yes="above $50k",
        no="not above $50k",
    )
    if income_flag:
        sentences.append(
            ensure_sentence(f"Their household income is {income_flag}.")
        )
    return sentences


def _employment_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences covering employment status and occupation."""

    sentences: List[str] = []
    employment = first_text(
        ex,
        selected,
        "employment_status",
        "employment",
        "labor_force",
        "employ",
    )
    occupation = first_text(ex, selected, "occupation", "occupation_text")
    if employment and occupation:
        sentences.append(
            ensure_sentence(
                "They work as "
                f"{clean_fragment(occupation)} and report being {clean_fragment(employment)}."
            )
        )
    elif occupation:
        sentences.append(
            ensure_sentence(f"They work as {clean_fragment(occupation)}.")
        )
    elif employment:
        sentences.append(
            ensure_sentence(f"They report being {clean_fragment(employment)}.")
        )
    return sentences


def _family_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences capturing family structure details."""

    sentences: List[str] = []
    marital_sentence = _marital_sentence(ex, selected)
    if marital_sentence:
        sentences.append(marital_sentence)
    sentences.extend(_children_sentences(ex, selected))
    household_sentence = _household_sentence(ex, selected)
    if household_sentence:
        sentences.append(household_sentence)
    return sentences


def _marital_sentence(ex: Dict[str, Any], selected: Dict[str, Any]) -> Optional[str]:
    """Return a sentence describing the viewer's marital status."""

    marital = first_text(ex, selected, "marital_status", "married", "marital")
    if not marital:
        return None
    return ensure_sentence(f"They report being {clean_fragment(marital)}.")


def _children_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences describing whether children are present in the household."""

    children_raw = first_non_nan_value(
        ex,
        selected,
        "children_in_house",
        "kids_household",
        "child18",
        "children",
    )
    if children_raw is None:
        return []
    flag = format_yes_no(children_raw, yes="yes", no="no")
    sentence = ""
    if flag == "yes":
        sentence = "They have children in their household."
    elif flag == "no":
        sentence = "They do not have children in their household."
    else:
        formatted = format_field_value("child18", children_raw)
        if formatted:
            lowered = formatted.lower()
            if lowered.startswith("no "):
                sentence = "They do not have children in their household."
            elif "children" in lowered:
                sentence = "They have children in their household."
            else:
                sentence = f"Children in household: {formatted}."
    if not sentence:
        return []
    return [ensure_sentence(sentence)]


def _household_sentence(ex: Dict[str, Any], selected: Dict[str, Any]) -> Optional[str]:
    """Return a sentence summarising household size when available."""

    household = first_text(ex, selected, "household_size", "hh_size")
    if not household:
        return None
    size_clean = clean_fragment(household)
    try:
        size_val = float(size_clean)
    except (TypeError, ValueError):
        size_val = math.nan
    sentence = ""
    if math.isfinite(size_val):
        size_int = int(round(size_val))
        if abs(size_val - size_int) < 1e-6:
            if size_int == 1:
                sentence = "They live alone."
            else:
                sentence = f"Their household has {size_int} people."
    if not sentence:
        sentence = f"Their household size is {size_clean}."
    return ensure_sentence(sentence)


def _religion_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences covering religious identity and related context."""

    sentences: List[str] = []
    identity = _religion_identity_sentence(ex, selected)
    if identity:
        sentences.append(identity)
    attendance = _attendance_sentence(ex, selected)
    if attendance:
        sentences.append(attendance)
    veteran = _veteran_sentence(ex, selected)
    if veteran:
        sentences.append(veteran)
    return sentences


def _religion_identity_sentence(ex: Dict[str, Any], selected: Dict[str, Any]) -> Optional[str]:
    """Return a sentence describing religious affiliation when available."""

    religion = first_text(
        ex,
        selected,
        "religion",
        "relig_affiliation",
        "religious_affiliation",
        "religpew",
        "religion_text",
    )
    if not religion:
        return None
    return ensure_sentence(f"They identify as {clean_fragment(religion)}.")


def _attendance_sentence(ex: Dict[str, Any], selected: Dict[str, Any]) -> Optional[str]:
    """Return a sentence summarising religious attendance or importance."""

    attendance = first_text(
        ex,
        selected,
        "relig_attend",
        "church_attend",
        "service_attendance",
        "pew_religimp",
    )
    if not attendance:
        return None
    attendance_clean = clean_fragment(attendance)
    lowered = attendance_clean.lower()
    if "important" in lowered:
        text = f"They say religion is {attendance_clean} to them."
    else:
        text = f"They report attending services {attendance_clean}."
    return ensure_sentence(text)


def _veteran_sentence(ex: Dict[str, Any], selected: Dict[str, Any]) -> Optional[str]:
    """Return a sentence indicating veteran status when known."""

    veteran_raw = first_non_nan_value(
        ex,
        selected,
        "veteran",
        "military_service",
        "veteran_status",
    )
    if veteran_raw is None:
        return None
    flag = format_yes_no(veteran_raw, yes="yes", no="no")
    if flag == "yes":
        return ensure_sentence("They are a veteran.")
    if flag == "no":
        return ensure_sentence("They are not a veteran.")
    veteran_text = clean_text(veteran_raw)
    if not veteran_text:
        return None
    return ensure_sentence(f"Veteran status: {veteran_text}.")


def _language_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences describing the language used to complete the survey."""

    language = first_non_nan_value(
        ex,
        selected,
        "user_language",
        "user_language_w1",
        "user_language_w2",
        "survey_language",
    )
    normalized = normalize_language_text(language)
    if not normalized:
        return []
    return [ensure_sentence(f"The survey was completed in {normalized}.")]
