"""Profile rendering helpers for prompt construction."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

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


def synthesize_viewer_sentence(ex: Dict[str, Any]) -> str:
    """Fallback sentence when an explicit profile is missing."""

    bits: List[str] = []
    age_text = describe_age_fragment(ex.get("age"))
    if age_text:
        bits.append(age_text)
    gender_text = describe_gender_fragment(ex.get("q26") or ex.get("gender"))
    if gender_text:
        bits.append(gender_text)
    race_text = clean_text(ex.get("q29") or ex.get("race"))
    if race_text and not is_nanlike(race_text):
        bits.append(race_text)
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
    """Compose the viewer profile sentences for an example row."""

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
        sentence = _ensure_sentence(viewer)
        if sentence:
            sentences.append(sentence)

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

    return ProfileRender(sentences=sentences, viewer_placeholder=viewer_placeholder)


def _load_selected_row(ex: Dict[str, Any]) -> Dict[str, Any]:
    raw = ex.get("selected_survey_row")
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


def _ensure_sentence(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


def _first_raw(ex: Dict[str, Any], selected: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        if key in ex:
            value = ex[key]
            if value is not None and not is_nanlike(value):
                return value
        if key in selected:
            value = selected.get(key)
            if value is not None and not is_nanlike(value):
                return value
    return None


def _first_text(
    ex: Dict[str, Any],
    selected: Dict[str, Any],
    *keys: str,
    limit: Optional[int] = None,
) -> str:
    value = _first_raw(ex, selected, *keys)
    if value is None:
        return ""
    return clean_text(value, limit=limit)


def _clean_fragment(text: str) -> str:
    return (text or "").strip().rstrip(".")


def _phrases_from_items(items: Sequence[str]) -> List[str]:
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
            phrases.append(f"{prefix} {value_clean}")
    return phrases


def _sentencize(prefix: str, items: Sequence[str]) -> str:
    phrases = _phrases_from_items(items)
    if not phrases:
        return ""
    if len(phrases) == 1:
        return f"{prefix} {phrases[0]}."
    return f"{prefix} {', '.join(phrases[:-1])}, and {phrases[-1]}."


def _demographic_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
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
    sentences: List[str] = []
    age_fragment = describe_age_fragment(
        _first_raw(ex, selected, "age", "age_cat", "age_category")
    )
    gender_fragment = describe_gender_fragment(
        _first_raw(ex, selected, "gender", "q26", "gender4", "gender_3_text")
    )
    race_fragment = _first_text(ex, selected, "race", "race_ethnicity", "ethnicity", "q29")

    identity_bits: List[str] = []
    if age_fragment:
        lowered = age_fragment.lower()
        if lowered.startswith(("between", "at least", "at most", "under", "over")):
            identity_bits.append(age_fragment)
        else:
            identity_bits.append(with_indefinite_article(age_fragment))
    if gender_fragment:
        gender_clean = _clean_fragment(gender_fragment)
        mapped = {"male": "man", "female": "woman"}
        gender_text = mapped.get(gender_clean.lower(), gender_clean)
        if not gender_text.lower().startswith(("a ", "an ", "the ", "someone", "somebody")):
            gender_text = with_indefinite_article(gender_text)
        identity_bits.append(gender_text)
    if identity_bits:
        sentences.append(
            _ensure_sentence(f"This viewer is {' and '.join(identity_bits)}.")
        )
    elif gender_fragment:
        sentences.append(
            _ensure_sentence(
                f"This viewer identifies as {_clean_fragment(gender_fragment)}."
            )
        )
    elif race_fragment:
        sentences.append(
            _ensure_sentence(
                f"This viewer identifies as {_clean_fragment(race_fragment)}."
            )
        )
    if race_fragment and identity_bits:
        sentences.append(
            _ensure_sentence(f"They identify as {_clean_fragment(race_fragment)}.")
        )
    return [s for s in sentences if s]


def _location_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    sentences: List[str] = []
    city = _first_text(ex, selected, "city", "city_name")
    state = _first_text(
        ex,
        selected,
        "state",
        "state_residence",
        "state_name",
        "inputstate",
        "thumb_states",
        "state_full",
    )
    county = _first_text(ex, selected, "county", "county_name")
    zip_text = _first_text(ex, selected, "zip3", "zip", "zip5")
    location = ""
    if city and state:
        location = f"{_clean_fragment(city)}, {_clean_fragment(state)}"
    elif city:
        location = _clean_fragment(city)
    elif state:
        location = _clean_fragment(state)
    extras: List[str] = []
    if county:
        extras.append(f"{_clean_fragment(county)} County")
    if zip_text:
        extras.append(f"ZIP3 {_clean_fragment(zip_text)}")
    if location:
        if extras:
            location = f"{location} ({'; '.join(extras)})"
        sentences.append(_ensure_sentence(f"They live in {location}."))
    elif extras:
        sentences.append(
            _ensure_sentence(f"They live in {human_join(extras)}.")
        )
    return [s for s in sentences if s]


def _education_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    sentences: List[str] = []
    education = _first_text(
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
        _first_raw(ex, selected, "college", "college_grad"),
        yes="yes",
        no="no",
    )
    if education:
        fragment = _clean_fragment(education)
        if college_flag == "yes":
            sentences.append(
                _ensure_sentence(
                    f"They have {fragment} and are college educated."
                )
            )
        elif college_flag == "no":
            sentences.append(
                _ensure_sentence(
                    f"They have {fragment} and are not college educated."
                )
            )
        else:
            sentences.append(_ensure_sentence(f"They have {fragment}."))
    elif college_flag == "yes":
        sentences.append(_ensure_sentence("They are college educated."))
    elif college_flag == "no":
        sentences.append(_ensure_sentence("They are not college educated."))
    return [s for s in sentences if s]


def _income_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    sentences: List[str] = []
    income = _first_text(
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
            _ensure_sentence(
                f"Their household income is reported as {_clean_fragment(income)}."
            )
        )
        return sentences
    income_flag = format_yes_no(
        _first_raw(ex, selected, "income_gt50k"),
        yes="above $50k",
        no="not above $50k",
    )
    if income_flag:
        sentences.append(
            _ensure_sentence(f"Their household income is {income_flag}.")
        )
    return sentences


def _employment_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    sentences: List[str] = []
    employment = _first_text(
        ex,
        selected,
        "employment_status",
        "employment",
        "labor_force",
        "employ",
    )
    occupation = _first_text(ex, selected, "occupation", "occupation_text")
    if employment and occupation:
        sentences.append(
            _ensure_sentence(
                "They work as "
                f"{_clean_fragment(occupation)} and report being {_clean_fragment(employment)}."
            )
        )
    elif occupation:
        sentences.append(
            _ensure_sentence(f"They work as {_clean_fragment(occupation)}.")
        )
    elif employment:
        sentences.append(
            _ensure_sentence(f"They report being {_clean_fragment(employment)}.")
        )
    return sentences


def _family_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    sentences: List[str] = []
    marital = _first_text(ex, selected, "marital_status", "married", "marital")
    if marital:
        sentences.append(
            _ensure_sentence(f"They report being {_clean_fragment(marital)}.")
        )

    children_raw = _first_raw(
        ex,
        selected,
        "children_in_house",
        "kids_household",
        "child18",
        "children",
    )
    if children_raw is not None:
        children_flag = format_yes_no(children_raw, yes="yes", no="no")
        if children_flag == "yes":
            sentences.append(_ensure_sentence("They have children in their household."))
        elif children_flag == "no":
            sentences.append(
                _ensure_sentence("They do not have children in their household.")
            )
        else:
            children_text = clean_text(children_raw)
            if children_text:
                sentences.append(
                    _ensure_sentence(
                        f"Children in household: {children_text}."
                    )
                )

    household = _first_text(ex, selected, "household_size", "hh_size")
    if household:
        size_clean = _clean_fragment(household)
        sentence = ""
        try:
            size_val = float(size_clean)
        except (TypeError, ValueError):
            size_val = math.nan
        if math.isfinite(size_val):
            size_int = int(round(size_val))
            if abs(size_val - size_int) < 1e-6:
                if size_int == 1:
                    sentence = "They live alone."
                else:
                    sentence = f"Their household has {size_int} people."
        if not sentence:
            sentence = f"Their household size is {size_clean}."
        sentences.append(_ensure_sentence(sentence))
    return sentences


def _religion_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    sentences: List[str] = []
    religion = _first_text(
        ex,
        selected,
        "religion",
        "relig_affiliation",
        "religious_affiliation",
        "religpew",
        "religion_text",
    )
    if religion:
        sentences.append(
            _ensure_sentence(f"They identify as {_clean_fragment(religion)}.")
        )
    attendance = _first_text(
        ex,
        selected,
        "relig_attend",
        "church_attend",
        "service_attendance",
        "pew_religimp",
    )
    if attendance:
        attendance_clean = _clean_fragment(attendance)
        lowered = attendance_clean.lower()
        if "important" in lowered:
            sentences.append(
                _ensure_sentence(
                    f"They say religion is {attendance_clean} to them."
                )
            )
        else:
            sentences.append(
                _ensure_sentence(
                    f"They report attending services {attendance_clean}."
                )
            )
    veteran_raw = _first_raw(
        ex,
        selected,
        "veteran",
        "military_service",
        "veteran_status",
    )
    if veteran_raw is not None:
        veteran_flag = format_yes_no(veteran_raw, yes="yes", no="no")
        if veteran_flag == "yes":
            sentences.append(_ensure_sentence("They are a veteran."))
        elif veteran_flag == "no":
            sentences.append(_ensure_sentence("They are not a veteran."))
        else:
            veteran_text = clean_text(veteran_raw)
            if veteran_text:
                sentences.append(
                    _ensure_sentence(f"Veteran status: {veteran_text}.")
                )
    return sentences


def _language_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    language = _first_raw(
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
    return [_ensure_sentence(f"The survey was completed in {normalized}.")]


def _politics_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    politics: List[str] = []
    party = _first_text(ex, selected, "pid1", "party_id", "party_registration", "partyid")
    if party:
        politics.append(f"Party identification: {party}")
    party_lean = _first_text(ex, selected, "pid2", "party_id_lean", "party_lean")
    if party_lean:
        politics.append(f"Party lean: {party_lean}")
    ideology = _first_text(ex, selected, "ideo1", "ideo2", "ideology", "ideology_text")
    if ideology:
        politics.append(f"Ideology: {ideology}")
    interest = _first_text(ex, selected, "pol_interest", "interest_politics", "political_interest")
    if interest:
        politics.append(f"Political interest: {interest}")
    vote_2016 = _first_text(ex, selected, "vote_2016", "presvote16post")
    if vote_2016:
        politics.append(f"Voted in 2016: {vote_2016}")
    vote_2020 = _first_text(ex, selected, "vote_2020", "presvote20post")
    if vote_2020:
        politics.append(f"Voted in 2020: {vote_2020}")
    vote_2024 = _first_text(ex, selected, "vote_2024", "vote_intent_2024", "vote_2024_intention")
    if vote_2024:
        politics.append(f"Vote intention 2024: {vote_2024}")
    trump = _first_text(
        ex,
        selected,
        "trump_approve",
        "trump_job_approval",
        "q5_2",
        "Q5_a",
        "Q5_a_W2",
        "political_lead_feels_2",
    )
    if trump:
        politics.append(f"Trump approval: {trump}")
    biden = _first_text(
        ex,
        selected,
        "biden_approve",
        "biden_job_approval",
        "q5_5",
        "Q5_b",
        "Q5_b_W2",
        "political_lead_feels_5",
    )
    if biden:
        politics.append(f"Biden approval: {biden}")
    civic = _first_text(ex, selected, "civic_participation", "volunteering", "civic_activity")
    if civic:
        politics.append(f"Civic engagement: {civic}")
    sentence = _sentencize("Politics include", politics)
    return [sentence] if sentence else []


def _gun_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    gun_section: List[str] = []
    ownership = _first_raw(ex, selected, "gun_own", "gunowner", "owns_gun")
    if ownership is not None:
        ownership_text = format_yes_no(ownership, yes="owns a gun", no="does not own a gun")
        if ownership_text:
            gun_section.append(f"Gun ownership: {ownership_text}")
        else:
            custom = clean_text(ownership)
            if custom:
                gun_section.append(f"Gun ownership: {custom}")
    known_keys = {"gun_own", "gunowner", "owns_gun"}
    for key, label in GUN_FIELD_LABELS.items():
        value = _first_raw(ex, selected, key)
        if value is None:
            continue
        known_keys.add(key.lower())
        text = format_yes_no(value)
        if text is None:
            text = clean_text(value, limit=200)
        if text:
            gun_section.append(f"{label}: {text}")
    for key in sorted(ex.keys()):
        lower = key.lower()
        if not lower.startswith("gun_") or lower in known_keys:
            continue
        value = ex.get(key)
        if is_nanlike(value):
            continue
        text = clean_text(value, limit=200)
        if not text:
            continue
        label = lower[4:].replace("_", " ").strip().capitalize()
        if label:
            gun_section.append(f"{label}: {text}")
    sentence = _sentencize("Gun policy views include", gun_section)
    return [sentence] if sentence else []


def _wage_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    wage_section: List[str] = []
    for key, label in MIN_WAGE_FIELD_LABELS.items():
        value = _first_raw(ex, selected, key)
        if value is None:
            continue
        text = format_yes_no(value)
        if text is None:
            text = clean_text(value, limit=220)
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
        text = clean_text(value, limit=220)
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
    sentence = _sentencize("Minimum wage views include", wage_section)
    return [sentence] if sentence else []


def _media_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    media_section: List[str] = []
    freq_raw = _first_raw(
        ex,
        selected,
        "freq_youtube",
        "q77",
        "Q77",
        "youtube_freq",
        "youtube_freq_v2",
    )
    if freq_raw is not None:
        code = str(freq_raw).strip()
        mapped = YT_FREQ_MAP.get(code)
        if mapped:
            media_section.append(f"YouTube frequency: {mapped}")
        else:
            freq_text = clean_text(freq_raw)
            if freq_text:
                media_section.append(f"YouTube frequency: {freq_text}")
    binge_raw = _first_raw(ex, selected, "binge_youtube", "youtube_time")
    binge_text = format_yes_no(binge_raw, yes="yes", no="no")
    if binge_text:
        media_section.append(f"Binge watches YouTube: {binge_text}")
    elif binge_raw is not None:
        binge_clean = clean_text(binge_raw)
        if binge_clean:
            media_section.append(f"YouTube time reported: {binge_clean}")
    media_fields: List[Sequence[str]] = [
        ("q8", "fav_channels"),
        ("q78", "popular_channels"),
        ("media_diet",),
        ("news_consumption",),
        ("news_sources",),
        ("news_sources_top",),
        ("news_frequency", "newsint"),
        ("platform_use",),
        ("social_media_use",),
        (
            "news_trust",
            "trust_majornews_w1",
            "trust_localnews_w1",
            "trust_majornews_w2",
            "trust_localnews_w2",
            "trust_majornews_w3",
            "trust_localnews_w3",
        ),
    ]
    labels = [
        "Favorite channels",
        "Popular channels followed",
        "Media diet",
        "News consumption",
        "News sources",
        "Top news sources",
        "News frequency",
        "Platform usage",
        "Social media use",
        "News trust",
    ]
    seen: set[str] = set()
    for sources, label in zip(media_fields, labels):
        text = ""
        for key in sources:
            text = _first_text(ex, selected, key, limit=220)
            if text:
                break
        if not text:
            continue
        if label == "Favorite channels" and label in seen:
            continue
        media_section.append(f"{label}: {text}")
        seen.add(label)
    sentence = _sentencize("Media habits include", media_section)
    return [sentence] if sentence else []


__all__ = [
    "ProfileRender",
    "render_profile",
    "synthesize_viewer_sentence",
]
