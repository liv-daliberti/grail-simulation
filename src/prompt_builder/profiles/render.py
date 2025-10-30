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

"""Core entry points for viewer-centric profile rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..formatters import clean_text, describe_age_fragment, describe_gender_fragment
from ..profile_helpers import ensure_sentence, load_selected_row
from ..value_maps import format_field_value
from .demographics import _demographic_sentences
from .guns import _gun_sentences
from .media import _media_sentences
from .politics import _politics_sentences
from .wage import _wage_sentences


@dataclass
class ProfileRender:
    """
    Container for profile sentences and placeholder metadata.

    :ivar sentences: Ordered sentences describing the viewer's profile.
    :type sentences: List[str]
    :ivar viewer_placeholder: Flag indicating the profile relied on fallback text.
    :type viewer_placeholder: bool
    """

    sentences: List[str]
    viewer_placeholder: bool


def synthesize_viewer_sentence(ex: Dict[str, Any]) -> str:
    """
    Generate a fallback viewer description when an explicit profile is absent.

    :param ex: Data row containing demographic and behavioural fields.
    :type ex: Dict[str, Any]
    :returns: Concise description synthesised from available attributes.
    :rtype: str
    """

    bits: List[str] = []
    age_text = describe_age_fragment(ex.get("age"))
    if age_text:
        bits.append(age_text)
    gender_text = describe_gender_fragment(ex.get("q26") or ex.get("gender"))
    if gender_text:
        bits.append(gender_text)
    race_raw = ex.get("q29") or ex.get("race")
    race_text = format_field_value("race", race_raw)
    if race_text:
        bits.append(race_text)
    pid1 = format_field_value("pid1", ex.get("pid1"))
    ideo1 = format_field_value("ideo1", ex.get("ideo1"))
    if pid1 and pid1.lower() != "nan":
        if ideo1 and ideo1.lower() != "nan":
            bits.append(f"{pid1} {ideo1}".lower())
        else:
            bits.append(pid1)
    elif ideo1 and ideo1.lower() != "nan":
        bits.append(ideo1.lower())
    income = format_field_value("income", ex.get("q31") or ex.get("income"))
    if income:
        bits.append(income)
    college = format_field_value("college", ex.get("college"))
    if college and college.lower() in {"yes", "college educated", "college-educated"}:
        bits.append("college-educated")
    freq = format_field_value("freq_youtube", ex.get("freq_youtube"))
    if freq:
        bits.append(f"watches YouTube {freq}")
    return ", ".join(bits) if bits else "(no profile provided)"


def render_profile(ex: Dict[str, Any]) -> ProfileRender:
    """
    Compose profile sentences describing the viewer associated with ``ex``.

    :param ex: Data row that may contain inline or nested survey information.
    :type ex: Dict[str, Any]
    :returns: Structured profile output including sentences and placeholder state.
    :rtype: ProfileRender
    """

    selected_row = load_selected_row(ex)
    viewer = clean_text(ex.get("viewer_profile_sentence"))
    if not viewer:
        viewer = synthesize_viewer_sentence(ex)
    sentences: List[str] = []
    viewer_placeholder = False
    viewer_sentence = ""
    if viewer:
        normalized = viewer.strip().lower()
        if normalized in {"(no profile provided)", "no profile provided"}:
            viewer_placeholder = True
        viewer_sentence = ensure_sentence(viewer)
        if viewer_placeholder:
            sentences.append(viewer_sentence)

    demographics = _demographic_sentences(ex, selected_row)
    politics = _politics_sentences(ex, selected_row)
    gun = _gun_sentences(ex, selected_row)
    wage = _wage_sentences(ex, selected_row)
    media = _media_sentences(ex, selected_row)

    sentences.extend(demographics)
    sentences.extend(politics)
    sentences.extend(gun)
    sentences.extend(wage)
    sentences.extend(media)

    if viewer_placeholder and len(sentences) > 1:
        sentences = [
            s for s in sentences if "(no profile provided)" not in s.lower()
        ]
    if viewer_placeholder and sentences:
        if all("(no profile provided)" in s.lower() for s in sentences):
            sentences = []
    if not viewer_placeholder and not sentences and viewer_sentence:
        sentences.append(viewer_sentence)

    return ProfileRender(sentences=sentences, viewer_placeholder=viewer_placeholder)
