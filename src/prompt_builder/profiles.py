"""Profile rendering helpers for prompt construction."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

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
from .value_maps import format_field_value

MEDIA_SOURCES: Sequence[tuple[Sequence[str], str, bool]] = (
    (("q8", "fav_channels"), "Favorite channels", True),
    (("q78", "popular_channels"), "Popular channels followed", False),
    (("media_diet",), "Media diet", False),
    (("news_consumption",), "News consumption", False),
    (("news_sources",), "News sources", False),
    (("news_sources_top",), "Top news sources", False),
    (("news_frequency", "newsint"), "News frequency", False),
    (("platform_use",), "Platform usage", False),
    (("social_media_use",), "Social media use", False),
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
        False,
    ),
)

POLITICS_FIELD_SPECS: Sequence[tuple[Sequence[str], str]] = (
    (("pid1", "party_id", "party_registration", "partyid"), "Party identification"),
    (("pid2", "party_id_lean", "party_lean"), "Party lean"),
    (("ideo1", "ideo2", "ideology", "ideology_text"), "Ideology"),
    (("pol_interest", "interest_politics", "political_interest"), "Political interest"),
    (("vote_2016", "presvote16post"), "Voted in 2016"),
    (("vote_2020", "presvote20post"), "Voted in 2020"),
    (("vote_2024", "vote_intent_2024", "vote_2024_intention"), "Vote intention 2024"),
    (
        (
            "trump_approve",
            "trump_job_approval",
            "q5_2",
            "Q5_a",
            "Q5_a_W2",
            "political_lead_feels_2",
        ),
        "Trump approval",
    ),
    (
        (
            "biden_approve",
            "biden_job_approval",
            "q5_5",
            "Q5_b",
            "Q5_b_W2",
            "political_lead_feels_5",
        ),
        "Biden approval",
    ),
    (("civic_participation", "volunteering", "civic_activity"), "Civic engagement"),
)


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

    selected_row = _load_selected_row(ex)
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
        viewer_sentence = _ensure_sentence(viewer)
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


def _load_selected_row(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the ``selected_survey_row`` field as a plain dictionary.

    :param ex: Dataset example containing optional survey-row metadata.
    :returns: Mapping of selected survey responses or an empty dictionary.
    """
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
    """Ensure that ``text`` ends with sentence punctuation.

    :param text: Candidate fragment to normalise.
    :returns: Sentence-terminated string or an empty string when missing.
    """
    stripped = (text or "").strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


def _first_raw(ex: Dict[str, Any], selected: Dict[str, Any], *keys: str) -> Optional[Any]:
    """Return the first non-null value among ``keys`` from the example metadata.

    :param ex: Primary dataset example.
    :param selected: Selected survey-row mapping.
    :param keys: Candidate field names searched in order.
    :returns: First value that is not NaN-like, or ``None``.
    """
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
    """Return the first textual value among ``keys`` with optional length limit.

    :param ex: Primary dataset example.
    :param selected: Selected survey-row mapping.
    :param keys: Candidate field names searched in order.
    :param limit: Optional maximum length passed to :func:`clean_text`.
    :returns: Cleaned text fragment or an empty string.
    """
    for key in keys:
        for dataset in (ex, selected):
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


def _clean_fragment(text: str) -> str:
    """Return a trimmed fragment without trailing periods.

    :param text: Text fragment to normalise.
    :returns: Stripped fragment.
    """
    return (text or "").strip().rstrip(".")


def _phrases_from_items(items: Sequence[str]) -> List[str]:
    """Convert labeled item strings into a list of readable phrases.

    :param items: Iterable of ``label: value`` strings.
    :returns: List of single phrases suitable for inclusion in sentences.
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
            phrases.append(f"{prefix} {value_clean}")
    return phrases


def _sentencize(prefix: str, items: Sequence[str]) -> str:
    """Join ``items`` into a grammatically correct sentence prefixed by text.

    :param prefix: Leading phrase preceding the list.
    :param items: Sequence of phrase fragments.
    :returns: Sentence string or empty string when ``items`` is empty.
    """
    phrases = _phrases_from_items(items)
    if not phrases:
        return ""
    if len(phrases) == 1:
        return f"{prefix} {phrases[0]}."
    return f"{prefix} {', '.join(phrases[:-1])}, and {phrases[-1]}."


def _demographic_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Assemble demographic sentences covering identity, location, and status.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of demographic sentences (may contain empties).
    """
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
    """Build sentences describing age, gender, and race identity fragments.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of identity-related sentences.
    """
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
    """Return sentences summarising the viewer's location context.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of location-specific sentences.
    """
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
    """Return sentences summarising educational attainment.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of education-focused sentences.
    """
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
    """Return sentences describing household income details.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of income-related sentences.
    """
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
    """Return sentences covering employment status and occupation.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of employment-related sentences.
    """
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
    """Return sentences capturing family structure details.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of family-oriented sentences.
    """
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
    marital = _first_text(ex, selected, "marital_status", "married", "marital")
    if not marital:
        return None
    return _ensure_sentence(f"They report being {_clean_fragment(marital)}.")


def _children_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences describing whether children are present in the household."""
    children_raw = _first_raw(
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
    return [_ensure_sentence(sentence)]


def _household_sentence(ex: Dict[str, Any], selected: Dict[str, Any]) -> Optional[str]:
    """Return a sentence summarising household size when available."""
    household = _first_text(ex, selected, "household_size", "hh_size")
    if not household:
        return None
    size_clean = _clean_fragment(household)
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
    return _ensure_sentence(sentence)


def _religion_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences covering religious identity and related context.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of religion-related sentences.
    """
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
    religion = _first_text(
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
    return _ensure_sentence(f"They identify as {_clean_fragment(religion)}.")


def _attendance_sentence(ex: Dict[str, Any], selected: Dict[str, Any]) -> Optional[str]:
    """Return a sentence summarising religious attendance or importance."""
    attendance = _first_text(
        ex,
        selected,
        "relig_attend",
        "church_attend",
        "service_attendance",
        "pew_religimp",
    )
    if not attendance:
        return None
    attendance_clean = _clean_fragment(attendance)
    lowered = attendance_clean.lower()
    if "important" in lowered:
        text = f"They say religion is {attendance_clean} to them."
    else:
        text = f"They report attending services {attendance_clean}."
    return _ensure_sentence(text)


def _veteran_sentence(ex: Dict[str, Any], selected: Dict[str, Any]) -> Optional[str]:
    """Return a sentence indicating veteran status when known."""
    veteran_raw = _first_raw(
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
        return _ensure_sentence("They are a veteran.")
    if flag == "no":
        return _ensure_sentence("They are not a veteran.")
    veteran_text = clean_text(veteran_raw)
    if not veteran_text:
        return None
    return _ensure_sentence(f"Veteran status: {veteran_text}.")


def _language_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences describing the language used to complete the survey."""
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


def _collect_labeled_fields(
    ex: Dict[str, Any],
    selected: Dict[str, Any],
    specs: Sequence[tuple[Sequence[str], str]],
) -> List[str]:
    """Collect labelled phrases from the example based on field specifications.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :param specs: Sequence of ``(field_names, label)`` pairs.
    :returns: List of ``label: value`` strings where values are present.
    """
    entries: List[str] = []
    for keys, label in specs:
        value = _first_text(ex, selected, *keys)
        if value:
            entries.append(f"{label}: {value}")
    return entries


def _politics_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences covering political views and affiliations.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of politics-related sentences.
    """
    politics = _collect_labeled_fields(ex, selected, POLITICS_FIELD_SPECS)
    sentence = _sentencize("Politics include", politics)
    return [sentence] if sentence else []


def _gun_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences summarising gun ownership and policy views.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of gun-policy sentences.
    """
    gun_section: List[str] = []
    ownership_entry = _gun_ownership_entry(ex, selected)
    if ownership_entry:
        gun_section.append(ownership_entry)
    labeled_entries, known_keys = _gun_labeled_entries(ex, selected)
    gun_section.extend(labeled_entries)
    gun_section.extend(_gun_additional_entries(ex, known_keys))
    sentence = _sentencize("Gun policy views include", gun_section)
    return [sentence] if sentence else []


def _gun_ownership_entry(ex: Dict[str, Any], selected: Dict[str, Any]) -> Optional[str]:
    """Return a phrase describing gun ownership status."""
    ownership = _first_raw(ex, selected, "gun_own", "gunowner", "owns_gun")
    if ownership is None:
        return None
    ownership_text = format_yes_no(ownership, yes="owns a gun", no="does not own a gun")
    if ownership_text:
        return f"Gun ownership: {ownership_text}"
    custom = clean_text(ownership)
    if custom:
        return f"Gun ownership: {custom}"
    return None


def _gun_labeled_entries(
    ex: Dict[str, Any],
    selected: Dict[str, Any],
) -> tuple[List[str], set[str]]:
    """Return labelled gun-policy entries and the keys that were consumed.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: Tuple of ``(entries, known_keys)``.
    """
    entries: List[str] = []
    known_keys: set[str] = {"gun_own", "gunowner", "owns_gun"}
    for key, label in GUN_FIELD_LABELS.items():
        value = _first_raw(ex, selected, key)
        if value is None:
            continue
        known_keys.add(key.lower())
        text = format_yes_no(value)
        if text is None:
            text = clean_text(value, limit=200)
        if text:
            entries.append(f"{label}: {text}")
    return entries, known_keys


def _gun_additional_entries(ex: Dict[str, Any], known_keys: set[str]) -> List[str]:
    """Return additional gun-policy entries excluding previously seen keys.

    :param ex: Dataset example containing viewer metadata.
    :param known_keys: Lowercased keys already consumed.
    :returns: List of ``label: value`` strings for additional fields.
    """
    entries: List[str] = []
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
            entries.append(f"{label}: {text}")
    return entries


def _wage_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences covering minimum-wage and wage policy attitudes.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of minimum-wage sentences.
    """
    wage_section: List[str] = []
    for key, label in MIN_WAGE_FIELD_LABELS.items():
        value = _first_raw(ex, selected, key)
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
    sentence = _sentencize("Minimum wage views include", wage_section)
    return [sentence] if sentence else []


def _media_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences describing overall media consumption habits.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of media-related sentences.
    """
    media_section: List[str] = []
    media_section.extend(_youtube_frequency_sentences(ex, selected))
    media_section.extend(_youtube_binge_sentences(ex, selected))
    seen: set[str] = set()
    for sources, label, skip_duplicate in MEDIA_SOURCES:
        text = _first_available_text(ex, selected, sources)
        if not text:
            continue
        if skip_duplicate and label in seen:
            continue
        media_section.append(f"{label}: {text}")
        seen.add(label)
    sentence = _sentencize("Media habits include", media_section)
    return [sentence] if sentence else []


def _youtube_frequency_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return phrases describing YouTube watch frequency.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of frequency phrases (0 or 1 entry).
    """
    freq_raw = _first_raw(
        ex,
        selected,
        "freq_youtube",
        "q77",
        "Q77",
        "youtube_freq",
        "youtube_freq_v2",
    )
    if freq_raw is None:
        return []
    code = str(freq_raw).strip()
    mapped = YT_FREQ_MAP.get(code)
    if mapped:
        return [f"YouTube frequency: {mapped}"]
    freq_text = clean_text(freq_raw)
    if not freq_text:
        return []
    return [f"YouTube frequency: {freq_text}"]


def _youtube_binge_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return phrases describing binge behaviour and reported watch time.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :returns: List of binge-related phrases (0 or 1 entry).
    """
    binge_raw = _first_raw(ex, selected, "binge_youtube", "youtube_time")
    if binge_raw is None:
        return []
    binge_text = format_yes_no(binge_raw, yes="yes", no="no")
    if binge_text:
        return [f"Binge watches YouTube: {binge_text}"]
    binge_clean = clean_text(binge_raw)
    if not binge_clean:
        return []
    return [f"YouTube time reported: {binge_clean}"]


def _first_available_text(
    ex: Dict[str, Any], selected: Dict[str, Any], keys: Sequence[str]
) -> str:
    """Return the first non-empty text value across ``keys``.

    :param ex: Dataset example containing viewer metadata.
    :param selected: Selected survey-row mapping.
    :param keys: Sequence of candidate field names.
    :returns: Cleaned text or an empty string.
    """
    for key in keys:
        text = _first_text(ex, selected, key, limit=220)
        if text:
            return text
    return ""


__all__ = [
    "ProfileRender",
    "render_profile",
    "synthesize_viewer_sentence",
]
