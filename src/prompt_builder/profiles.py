"""Profile rendering helpers for prompt construction."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .constants import GUN_FIELD_LABELS, MIN_WAGE_FIELD_LABELS, YT_FREQ_MAP
from .formatters import (
    clean_text,
    describe_age_fragment,
    describe_gender_fragment,
    human_join,
    normalize_language_text,
    with_indefinite_article,
)
from .parsers import format_yes_no, is_nanlike


@dataclass
class ProfileRender:
    """Container for profile sentences and placeholder metadata."""

    sentences: List[str]
    viewer_placeholder: bool


def _ensure_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if text[-1] in ".!?":
        return text
    return text + "."


def _load_selected_row(ex: Dict[str, Any]) -> Dict[str, Any]:
    raw = ex.get("selected_survey_row")
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
        return {}
    as_py = getattr(raw, "as_py", None)
    if callable(as_py):
        try:
            candidate = as_py()
            if isinstance(candidate, dict):
                return candidate
        except Exception:
            return {}
    return {}


def _first_raw(
    ex: Dict[str, Any],
    selected_row: Dict[str, Any],
    *keys: str,
) -> Optional[Any]:
    for key in keys:
        if key in ex:
            value = ex[key]
            if value is not None and not is_nanlike(value):
                return value
        if key in selected_row:
            value = selected_row.get(key)
            if value is not None and not is_nanlike(value):
                return value
    return None


def _first_text(
    ex: Dict[str, Any],
    selected_row: Dict[str, Any],
    *keys: str,
    limit: Optional[int] = None,
) -> str:
    value = _first_raw(ex, selected_row, *keys)
    if value is None:
        return ""
    return clean_text(value, limit=limit)


def _clean_fragment(text: str) -> str:
    return (text or "").strip().rstrip(".")


def _phrases_from_items(items: Sequence[str]) -> List[str]:
    phrases: List[str] = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            label, value = item.split(":", 1)
            label_clean = label.strip()
            value_clean = value.strip()
            if not label_clean and not value_clean:
                continue
            if not value_clean:
                phrases.append(label_clean)
                continue
            label_phrase = label_clean
            if label_phrase and label_phrase[1:].lower() == label_phrase[1:]:
                label_phrase = label_phrase[0].lower() + label_phrase[1:]
            label_lower = label_clean.lower()
            value_lower = value_clean.lower()
            if label_lower and label_lower in value_lower:
                phrases.append(value_clean)
            elif value_lower in {"yes", "no"}:
                phrases.append(f"{label_phrase} {value_lower}")
            else:
                phrases.append(f"{label_phrase} {value_clean}")
        else:
            phrases.append(item)
    return phrases


def _sentencize(prefix: str, items: Sequence[str]) -> str:
    phrases = _phrases_from_items(items)
    if not phrases:
        return ""
    if len(phrases) == 1:
        return f"{prefix} {phrases[0]}."
    return f"{prefix} {', '.join(phrases[:-1])}, and {phrases[-1]}."


def synthesize_viewer_sentence(ex: Dict[str, Any]) -> str:
    bits: List[str] = []
    age_text = describe_age_fragment(ex.get("age"))
    if age_text:
        bits.append(age_text)

    gender = describe_gender_fragment(ex.get("q26") or ex.get("gender"))
    if gender:
        bits.append(gender)

    race = clean_text(ex.get("q29") or ex.get("race"))
    if race and not is_nanlike(race):
        bits.append(race)

    pid1 = clean_text(ex.get("pid1"))
    ideo1 = clean_text(ex.get("ideo1"))
    if pid1 and pid1.lower() != "nan":
        if ideo1 and ideo1.lower() != "nan":
            bits.append(f"{pid1} {ideo1}".lower())
        else:
            bits.append(pid1)
    elif ideo1 and ideo1.lower() != "nan":
        bits.append(ideo1.lower())

    income = clean_text(ex.get("q31") or ex.get("income"))
    if income and income.lower() != "nan":
        bits.append(income)

    college = str(ex.get("college") or "").strip().lower()
    if college in {"1", "true", "t", "yes", "y"}:
        bits.append("college-educated")

    freq = str(ex.get("freq_youtube") or "").strip()
    if freq in YT_FREQ_MAP:
        bits.append(f"watches YouTube {YT_FREQ_MAP[freq]}")

    return ", ".join(bits) if bits else "(no profile provided)"


def render_profile(ex: Dict[str, Any]) -> ProfileRender:
    """Return the profile sentences for ``ex``."""

    selected_row = _load_selected_row(ex)
    viewer = clean_text(ex.get("viewer_profile_sentence"))
    if not viewer:
        viewer = synthesize_viewer_sentence(ex)

    sentences: List[str] = []
    viewer_placeholder = False
    if viewer:
        normalized = viewer.strip().lower()
        if normalized in {"(no profile provided)", "no profile provided"}:
            viewer_placeholder = True
        ensured = _ensure_sentence(viewer)
        if ensured:
            sentences.append(ensured)

    demographic_sentences: List[str] = []

    age_fragment = describe_age_fragment(
        _first_raw(ex, selected_row, "age", "age_cat", "age_category")
    )
    gender_fragment = describe_gender_fragment(
        _first_raw(ex, selected_row, "gender", "q26", "gender4", "gender_3_text")
    )
    race_fragment = _first_text(ex, selected_row, "race", "race_ethnicity", "ethnicity", "q29")

    identity_bits: List[str] = []
    if age_fragment:
        lowered_age = age_fragment.lower()
        if lowered_age.startswith(("between", "at least", "at most", "under", "over")):
            identity_bits.append(age_fragment)
        else:
            identity_bits.append(with_indefinite_article(age_fragment))
    if gender_fragment:
        gender_clean = _clean_fragment(gender_fragment)
        lowered_gender = gender_clean.lower()
        if lowered_gender in {"male"}:
            gender_clean = "man"
        elif lowered_gender in {"female"}:
            gender_clean = "woman"
        if not gender_clean.lower().startswith(("a ", "an ", "the ", "someone", "somebody")):
            gender_clean = with_indefinite_article(gender_clean)
        identity_bits.append(gender_clean)
    if identity_bits:
        demographic_sentences.append(
            _ensure_sentence(f"This viewer is {' and '.join(identity_bits)}.")
        )
    elif gender_fragment:
        demographic_sentences.append(
            _ensure_sentence(f"This viewer identifies as { _clean_fragment(gender_fragment)}.")
        )
    elif race_fragment:
        demographic_sentences.append(
            _ensure_sentence(f"This viewer identifies as {_clean_fragment(race_fragment)}.")
        )
    if race_fragment and identity_bits:
        demographic_sentences.append(
            _ensure_sentence(f"They identify as {_clean_fragment(race_fragment)}.")
        )

    city_text = _first_text(ex, selected_row, "city", "city_name")
    state_text = _first_text(
        ex, selected_row, "state", "state_residence", "state_name", "inputstate", "thumb_states", "state_full"
    )
    county_text = _first_text(ex, selected_row, "county", "county_name")
    zip_text = _first_text(ex, selected_row, "zip3", "zip", "zip5")
    location_phrase = ""
    if city_text and state_text:
        location_phrase = f"{_clean_fragment(city_text)}, {_clean_fragment(state_text)}"
    elif city_text:
        location_phrase = _clean_fragment(city_text)
    elif state_text:
        location_phrase = _clean_fragment(state_text)
    extras: List[str] = []
    if county_text:
        extras.append(f"{_clean_fragment(county_text)} County")
    if zip_text:
        extras.append(f"ZIP3 {_clean_fragment(zip_text)}")
    if location_phrase:
        if extras:
            location_phrase = f"{location_phrase} ({'; '.join(extras)})"
        demographic_sentences.append(_ensure_sentence(f"They live in {location_phrase}."))
    elif extras:
        demographic_sentences.append(
            _ensure_sentence(f"They live in {human_join(extras)}.")
        )

    education_text = _first_text(
        ex,
        selected_row,
        "education",
        "educ",
        "education_level",
        "college_desc",
        "highest_education",
        "education_text",
    )
    college_flag = format_yes_no(_first_raw(ex, selected_row, "college", "college_grad"), yes="yes", no="no")
    if education_text:
        cleaned_education = _clean_fragment(education_text)
        if college_flag == "yes":
            demographic_sentences.append(
                _ensure_sentence(f"They have {cleaned_education} and are college educated.")
            )
        elif college_flag == "no":
            demographic_sentences.append(
                _ensure_sentence(f"They have {cleaned_education} and are not college educated.")
            )
        else:
            demographic_sentences.append(_ensure_sentence(f"They have {cleaned_education}."))
    elif college_flag == "yes":
        demographic_sentences.append(_ensure_sentence("They are college educated."))
    elif college_flag == "no":
        demographic_sentences.append(_ensure_sentence("They are not college educated."))

    income_text = _first_text(
        ex,
        selected_row,
        "q31",
        "income",
        "household_income",
        "income_bracket",
        "income_cat",
    )
    if income_text:
        demographic_sentences.append(
            _ensure_sentence(f"Their household income is reported as {_clean_fragment(income_text)}.")
        )
    else:
        income_flag = format_yes_no(
            _first_raw(ex, selected_row, "income_gt50k"), yes="above $50k", no="not above $50k"
        )
        if income_flag:
            demographic_sentences.append(_ensure_sentence(f"Their household income is {income_flag}."))

    employment_text = _first_text(ex, selected_row, "employment_status", "employment", "labor_force", "employ")
    occupation_text = _first_text(ex, selected_row, "occupation", "occupation_text")
    if employment_text and occupation_text:
        demographic_sentences.append(
            _ensure_sentence(
                f"They work as {_clean_fragment(occupation_text)} and report being {_clean_fragment(employment_text)}."
            )
        )
    elif occupation_text:
        demographic_sentences.append(
            _ensure_sentence(f"They work as {_clean_fragment(occupation_text)}.")
        )
    elif employment_text:
        demographic_sentences.append(
            _ensure_sentence(f"They report being {_clean_fragment(employment_text)}.")
        )

    marital_text = _first_text(ex, selected_row, "marital_status", "married", "marital")
    if marital_text:
        demographic_sentences.append(
            _ensure_sentence(f"They report being {_clean_fragment(marital_text)}.")
        )

    children_raw = _first_raw(
        ex,
        selected_row,
        "children_in_house",
        "kids_household",
        "child18",
        "children",
    )
    if children_raw is not None:
        children_flag = format_yes_no(children_raw, yes="yes", no="no")
        if children_flag == "yes":
            demographic_sentences.append(_ensure_sentence("They have children in their household."))
        elif children_flag == "no":
            demographic_sentences.append(_ensure_sentence("They do not have children in their household."))
        else:
            children_clean = clean_text(children_raw)
            if children_clean:
                demographic_sentences.append(
                    _ensure_sentence(f"Children in household: {children_clean}.")
                )

    household_size_text = _first_text(ex, selected_row, "household_size", "hh_size")
    if household_size_text:
        size_clean = _clean_fragment(household_size_text)
        size_sentence = ""
        try:
            size_value = float(size_clean)
            if math.isfinite(size_value):
                size_int = int(round(size_value))
                if abs(size_value - size_int) < 1e-6:
                    if size_int == 1:
                        size_sentence = "They live alone."
                    else:
                        size_sentence = f"Their household has {size_int} people."
        except Exception:
            size_sentence = ""
        if not size_sentence:
            size_sentence = f"Their household size is {size_clean}."
        demographic_sentences.append(_ensure_sentence(size_sentence))

    religion_text = _first_text(
        ex,
        selected_row,
        "religion",
        "relig_affiliation",
        "religious_affiliation",
        "religpew",
        "religion_text",
    )
    if religion_text:
        demographic_sentences.append(
            _ensure_sentence(f"They identify as {_clean_fragment(religion_text)}.")
        )

    attendance_text = _first_text(
        ex,
        selected_row,
        "relig_attend",
        "church_attend",
        "service_attendance",
        "pew_religimp",
    )
    if attendance_text:
        attendance_clean = _clean_fragment(attendance_text)
        lowered = attendance_clean.lower()
        if "important" in lowered:
            demographic_sentences.append(
                _ensure_sentence(f"They say religion is {attendance_clean} to them.")
            )
        else:
            demographic_sentences.append(
                _ensure_sentence(f"They report attending services {attendance_clean}.")
            )

    veteran_raw = _first_raw(ex, selected_row, "veteran", "military_service", "veteran_status")
    if veteran_raw is not None:
        veteran_flag = format_yes_no(veteran_raw, yes="yes", no="no")
        if veteran_flag == "yes":
            demographic_sentences.append(_ensure_sentence("They are a veteran."))
        elif veteran_flag == "no":
            demographic_sentences.append(_ensure_sentence("They are not a veteran."))
        else:
            veteran_clean = clean_text(veteran_raw)
            if veteran_clean:
                demographic_sentences.append(
                    _ensure_sentence(f"Veteran status: {veteran_clean}.")
                )

    language_text = _first_raw(
        ex,
        selected_row,
        "user_language",
        "user_language_w1",
        "user_language_w2",
        "survey_language",
    )
    language_clean = normalize_language_text(language_text)
    if language_clean:
        demographic_sentences.append(
            _ensure_sentence(f"The survey was completed in {language_clean}.")
        )

    for sentence in demographic_sentences:
        if sentence:
            sentences.append(sentence)

    politics: List[str] = []
    party_text = _first_text(ex, selected_row, "pid1", "party_id", "party_registration", "partyid")
    if party_text:
        politics.append(f"Party identification: {party_text}")
    party_lean_text = _first_text(ex, selected_row, "pid2", "party_id_lean", "party_lean")
    if party_lean_text:
        politics.append(f"Party lean: {party_lean_text}")
    ideology_text = _first_text(ex, selected_row, "ideo1", "ideo2", "ideology", "ideology_text")
    if ideology_text:
        politics.append(f"Ideology: {ideology_text}")
    pol_interest = _first_text(ex, selected_row, "pol_interest", "interest_politics", "political_interest")
    if pol_interest:
        politics.append(f"Political interest: {pol_interest}")
    vote_2016 = _first_text(ex, selected_row, "vote_2016", "presvote16post")
    if vote_2016:
        politics.append(f"Voted in 2016: {vote_2016}")
    vote_2020 = _first_text(ex, selected_row, "vote_2020", "presvote20post")
    if vote_2020:
        politics.append(f"Voted in 2020: {vote_2020}")
    vote_2024 = _first_text(ex, selected_row, "vote_2024", "vote_intent_2024", "vote_2024_intention")
    if vote_2024:
        politics.append(f"Vote intention 2024: {vote_2024}")
    trump_approve = _first_text(
        ex,
        selected_row,
        "trump_approve",
        "trump_job_approval",
        "q5_2",
        "Q5_a",
        "Q5_a_W2",
        "political_lead_feels_2",
    )
    if trump_approve:
        politics.append(f"Trump approval: {trump_approve}")
    biden_approve = _first_text(
        ex,
        selected_row,
        "biden_approve",
        "biden_job_approval",
        "q5_5",
        "Q5_b",
        "Q5_b_W2",
        "political_lead_feels_5",
    )
    if biden_approve:
        politics.append(f"Biden approval: {biden_approve}")
    civic_engagement = _first_text(ex, selected_row, "civic_participation", "volunteering", "civic_activity")
    if civic_engagement:
        politics.append(f"Civic engagement: {civic_engagement}")

    politics_sentence = _sentencize("Politics include", politics)
    if politics_sentence:
        sentences.append(politics_sentence)

    gun_section: List[str] = []
    gun_own_val = _first_raw(ex, selected_row, "gun_own", "gunowner", "owns_gun")
    if gun_own_val is not None:
        gun_own_text = format_yes_no(gun_own_val, yes="owns a gun", no="does not own a gun")
        if gun_own_text:
            gun_section.append(f"Gun ownership: {gun_own_text}")
        else:
            custom = clean_text(gun_own_val)
            if custom:
                gun_section.append(f"Gun ownership: {custom}")

    known_gun_keys = {"gun_own", "gunowner", "owns_gun"}
    for key, label in GUN_FIELD_LABELS.items():
        value = _first_raw(ex, selected_row, key)
        if value is None:
            continue
        known_gun_keys.add(key.lower())
        text = format_yes_no(value)
        if text is None:
            text = clean_text(value, limit=200)
        if text:
            gun_section.append(f"{label}: {text}")

    for key in sorted(ex.keys()):
        low = key.lower()
        if not low.startswith("gun_"):
            continue
        if low in known_gun_keys:
            continue
        value = ex.get(key)
        if is_nanlike(value):
            continue
        text = clean_text(value, limit=200)
        if not text:
            continue
        label = low[4:].replace("_", " ").strip().capitalize()
        if not label:
            continue
        gun_section.append(f"{label}: {text}")
        known_gun_keys.add(low)

    gun_sentence = _sentencize("Gun policy views include", gun_section)
    if gun_sentence:
        sentences.append(gun_sentence)

    wage_section: List[str] = []
    for key, label in MIN_WAGE_FIELD_LABELS.items():
        value = _first_raw(ex, selected_row, key)
        if value is None:
            continue
        text = format_yes_no(value)
        if text is None:
            text = clean_text(value, limit=220)
        if text:
            wage_section.append(f"{label}: {text}")

    known_wage_keys = {k.lower() for k in MIN_WAGE_FIELD_LABELS}
    for key in sorted(ex.keys()):
        low = key.lower()
        if not (low.startswith("minwage") or low.startswith("mw_")):
            continue
        if low in known_wage_keys:
            continue
        value = ex.get(key)
        if is_nanlike(value):
            continue
        text = clean_text(value, limit=220)
        if not text:
            continue
        label = (
            low.replace("minwage", "minimum wage")
            .replace("mw_", "minimum wage ")
            .replace("_", " ")
            .strip()
        )
        if not label:
            continue
        wage_section.append(f"{label.capitalize()}: {text}")
        known_wage_keys.add(low)

    wage_sentence = _sentencize("Minimum wage views include", wage_section)
    if wage_sentence:
        sentences.append(wage_sentence)

    media_section: List[str] = []
    fy_raw = _first_raw(ex, selected_row, "freq_youtube", "q77", "Q77", "youtube_freq", "youtube_freq_v2")
    if fy_raw is not None:
        code = str(fy_raw).strip()
        freq = YT_FREQ_MAP.get(code)
        if freq:
            media_section.append(f"YouTube frequency: {freq}")
        else:
            freq_text = clean_text(fy_raw)
            if freq_text:
                media_section.append(f"YouTube frequency: {freq_text}")

    binge_raw = _first_raw(ex, selected_row, "binge_youtube", "youtube_time")
    binge_text = format_yes_no(binge_raw, yes="yes", no="no")
    if binge_text:
        media_section.append(f"Binge watches YouTube: {binge_text}")
    elif binge_raw is not None:
        binge_clean = clean_text(binge_raw)
        if binge_clean:
            media_section.append(f"YouTube time reported: {binge_clean}")

    media_labels: List[Tuple[Tuple[str, ...], str]] = [
        (("q8", "fav_channels"), "Favorite channels"),
        (("q78", "popular_channels"), "Popular channels followed"),
        (("media_diet",), "Media diet"),
        (("news_consumption",), "News consumption"),
        (("news_sources",), "News sources"),
        (("news_sources_top",), "Top news sources"),
        (("news_frequency", "newsint"), "News frequency"),
        (("platform_use",), "Platform usage"),
        (("social_media_use",), "Social media use"),
        (
            (
                "news_trust",
                "trust_majornews_w1",
                "trust_localnews_w1",
                "trust_majornews_w2",
                "trust_localnews_w2",
                "trust_majornews_w3",
                "trust_localnews_w3",
            ),
            "News trust",
        ),
    ]
    seen_media_labels: set[str] = set()
    for keys, label in media_labels:
        text = ""
        for key in keys:
            text = _first_text(ex, selected_row, key, limit=220)
            if text:
                break
        if not text:
            continue
        if label in seen_media_labels and label in {"Favorite channels"}:
            continue
        media_section.append(f"{label}: {text}")
        seen_media_labels.add(label)

    media_sentence = _sentencize("Media habits include", media_section)
    if media_sentence:
        sentences.append(media_sentence)

    if viewer_placeholder and len(sentences) > 1:
        sentences = [s for s in sentences if "(no profile provided)" not in s.lower()]
    if viewer_placeholder and sentences and all("(no profile provided)" in s.lower() for s in sentences):
        sentences = []

    return ProfileRender(sentences=sentences, viewer_placeholder=viewer_placeholder)


__all__ = [
    "ProfileRender",
    "render_profile",
    "synthesize_viewer_sentence",
]
