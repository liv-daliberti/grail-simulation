"""Enumerated value mappings and formatting helpers for prompt text."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional

from .constants import YT_FREQ_MAP
from .parsers import is_nanlike

RACE_CODE_MAP = {
    "1": "White (non-Hispanic)",
    "2": "Black or African-American",
    "3": "Hispanic or Latino",
    "4": "Asian or Pacific Islander",
    "5": "American Indian or Alaska Native",
    "6": "Multiracial",
    "7": "Other race",
    "8": "Prefer not to answer",
    "9": "Prefer not to answer",
}

EDUCATION_CODE_MAP = {
    "1": "less than high school",
    "2": "high school graduate",
    "3": "some college",
    "4": "associate degree",
    "5": "college graduate",
    "6": "postgraduate degree",
}

EMPLOYMENT_CODE_MAP = {
    "1": "working full time",
    "2": "working part time",
    "3": "self-employed",
    "4": "temporarily not working",
    "5": "retired",
    "6": "student",
    "7": "homemaker",
    "8": "unable to work",
    "9": "unemployed",
}

RELIGION_CODE_MAP = {
    "1": "Protestant",
    "2": "Catholic",
    "3": "Mormon",
    "4": "Orthodox Christian",
    "5": "Jewish",
    "6": "Muslim",
    "7": "Buddhist",
    "8": "Hindu",
    "9": "Atheist",
    "10": "Agnostic",
    "11": "Nothing in particular",
    "12": "Something else",
    "13": "Prefer not to answer",
}

RELIG_IMPORTANCE_MAP = {
    "1": "very important",
    "2": "somewhat important",
    "3": "not too important",
    "4": "not at all important",
}

VOTE_2016_MAP = {
    "1": "voted for Hillary Clinton",
    "2": "voted for Donald Trump",
    "3": "voted for Gary Johnson",
    "4": "voted for Jill Stein",
    "5": "voted for another candidate",
    "6": "did not vote",
    "7": "did not vote",
}

VOTE_2020_MAP = {
    "1": "voted for Joe Biden",
    "2": "voted for Donald Trump",
    "3": "voted for Jo Jorgensen",
    "4": "voted for Howie Hawkins",
    "5": "voted for another candidate",
    "6": "did not vote",
}

POLITICAL_INTEREST_MAP = {
    "1": "very interested in politics",
    "2": "somewhat interested in politics",
    "3": "not very interested in politics",
    "4": "not at all interested in politics",
    "5": "unsure about political interest",
}

POLITICAL_INTEREST_FIELDS = {"pol_interest", "interest_politics", "political_interest"}

NEWS_INTEREST_MAP = {
    "1": "pays attention to news most of the time",
    "2": "follows the news some of the time",
    "3": "checks the news occasionally",
    "4": "rarely follows the news",
    "5": "almost never follows the news",
    "6": "prefers not to say",
    "7": "not sure",
}

GUN_IMPORTANCE_MAP = {
    "1": "not at all important",
    "2": "not too important",
    "3": "somewhat important",
    "4": "very important",
    "5": "extremely important",
}

CHILDREN_MAP = {
    "0": "no children in the household",
    "1": "children in the household",
    "2": "no children in the household",
}

STUDY_LABEL_MAP = {
    "study1": "Study 1 (gun control MTurk)",
    "study2": "Study 2 (minimum wage MTurk)",
    "study3": "Study 3 (minimum wage YouGov)",
    "study4": "Study 4 (YouTube Shorts follow-up)",
    "unknown": "Study label unknown",
}

SLATE_SOURCE_MAP = {
    "display_orders": "experimental display order slate",
    "tree_metadata": "recommendation tree slate",
}

STATE_FIPS_MAP = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
    "60": "American Samoa",
    "66": "Guam",
    "69": "Northern Mariana Islands",
    "72": "Puerto Rico",
    "78": "U.S. Virgin Islands",
}


def _normalize_state_code(raw: str) -> Optional[str]:
    """Normalise a state FIPS code to a two-digit string."""

    text = raw.strip()
    if not text:
        return None
    if len(text) == 1:
        return f"0{text}"
    if len(text) == 2:
        return text
    try:
        number = int(float(text))
    except (TypeError, ValueError):
        return None
    return f"{number:02d}"


def _format_percentage(value: float, *, precision: int = 0) -> str:
    """Return a percentage string with bounded input and optional precision."""

    percent = max(0.0, min(100.0, value * 100.0))
    fmt = f"{{percent:.{precision}f}}%".format(percent=percent)
    if fmt.endswith(".0%"):
        fmt = fmt.replace(".0%", "%")
    return fmt


def _format_scale_percentage(raw: float) -> str:
    """Render a 0-100 scale value as an integer percentage string."""

    clamped = max(0.0, min(100.0, raw))
    formatted = f"{clamped:.0f}%"
    return formatted


def _format_currency(raw: float) -> str:
    """Return a currency string with thousands separators."""

    if math.isnan(raw):
        return ""
    if raw.is_integer():
        return f"${int(raw):,}"
    return f"${raw:,.2f}"


FIELD_VALUE_MAPS: Dict[str, Dict[str, str]] = {
    "race": RACE_CODE_MAP,
    "race_ethnicity": RACE_CODE_MAP,
    "ethnicity": RACE_CODE_MAP,
    "q29": RACE_CODE_MAP,
    "educ": EDUCATION_CODE_MAP,
    "education": EDUCATION_CODE_MAP,
    "education_level": EDUCATION_CODE_MAP,
    "college_desc": EDUCATION_CODE_MAP,
    "highest_education": EDUCATION_CODE_MAP,
    "education_text": EDUCATION_CODE_MAP,
    "employment": EMPLOYMENT_CODE_MAP,
    "employment_status": EMPLOYMENT_CODE_MAP,
    "labor_force": EMPLOYMENT_CODE_MAP,
    "employ": EMPLOYMENT_CODE_MAP,
    "religpew": RELIGION_CODE_MAP,
    "religion": RELIGION_CODE_MAP,
    "religious_affiliation": RELIGION_CODE_MAP,
    "religion_text": RELIGION_CODE_MAP,
    "pew_religimp": RELIG_IMPORTANCE_MAP,
    "freq_youtube": YT_FREQ_MAP,
    "youtube_freq": YT_FREQ_MAP,
    "youtube_freq_v2": YT_FREQ_MAP,
    "pol_interest": POLITICAL_INTEREST_MAP,
    "interest_politics": POLITICAL_INTEREST_MAP,
    "political_interest": POLITICAL_INTEREST_MAP,
    "newsint": NEWS_INTEREST_MAP,
    "child18": CHILDREN_MAP,
    "kids_household": CHILDREN_MAP,
    "children_in_house": CHILDREN_MAP,
    "college": {"1": "yes", "0": "no"},
    "college_grad": {"1": "yes", "0": "no"},
    "presvote16post": VOTE_2016_MAP,
    "vote_2016": VOTE_2016_MAP,
    "vote_2020": VOTE_2020_MAP,
    "presvote20post": VOTE_2020_MAP,
    "participant_study": STUDY_LABEL_MAP,
    "slate_source": SLATE_SOURCE_MAP,
}

STATE_FIELD_NAMES = {
    "state",
    "state_residence",
    "state_name",
    "state_full",
    "state_text",
    "inputstate",
    "state_fips",
}

PERCENTAGE_FIELDS = {
    "mw_support_w1",
    "mw_support_w2",
    "minwage15_w1",
    "minwage15_w2",
    "mw_index_w1",
    "mw_index_w2",
    "mw_support",
    "gun_index",
    "gun_index_2",
    "assault_ban",
    "affpol_ft",
    "affpol_ft_w2",
    "affpol_comfort",
    "affpol_comfort_w2",
    "affpol_smart",
    "affpol_smart_w2",
    "news_trust",
    "trust_majornews_w1",
    "trust_localnews_w1",
    "trust_majornews_w2",
    "trust_localnews_w2",
    "trust_majornews_w3",
    "trust_localnews_w3",
}

NEWS_TRUST_FIELDS = {
    "news_trust",
    "trust_majornews_w1",
    "trust_localnews_w1",
    "trust_majornews_w2",
    "trust_localnews_w2",
    "trust_majornews_w3",
    "trust_localnews_w3",
}

CURRENCY_FIELDS = {
    "minwage_text_w1",
    "minwage_text_w2",
    "minwage_text_r_w1",
    "minwage_text_r_w2",
    "minwage_text_r_w3",
    "minwage_priority",
    "minwage_importance",
}

SCALE_0_100_FIELDS = {
    "q5_2",
    "q5_a",
    "q5_a_w2",
    "political_lead_feels_2",
    "q5_5",
    "q5_b",
    "q5_b_w2",
    "political_lead_feels_5",
    "biden_approve",
    "trump_approve",
}

FEELING_THERMOMETER_FIELDS = {
    "trump_approve",
    "trump_job_approval",
    "q5_2",
    "q5_a",
    "q5_a_w2",
    "political_lead_feels_2",
    "biden_approve",
    "biden_job_approval",
    "q5_5",
    "q5_b",
    "q5_b_w2",
    "political_lead_feels_5",
}

IDEOLOGY_FIELDS = {"ideo1", "ideo2", "ideo3", "ideo4", "ideo5", "ideology", "ideology_text"}

GUN_IMPORTANCE_FIELDS = {"gun_importance", "gun_priority"}


def _as_float(value: Any) -> Optional[float]:
    """Best-effort conversion of ``value`` to ``float``."""

    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _format_ideology(raw: Any) -> Optional[str]:
    """Return a qualitative ideology label from numeric or textual inputs."""

    numeric = _as_float(raw)
    if numeric is None:
        text = str(raw).strip()
        return text or None
    label = "extremely conservative"
    if numeric <= 1.5:
        label = "extremely liberal"
    elif numeric <= 2.5:
        label = "liberal"
    elif numeric <= 3.5:
        label = "slightly liberal"
    elif numeric <= 4.5:
        label = "moderate"
    elif numeric <= 5.5:
        label = "slightly conservative"
    elif numeric <= 6.5:
        label = "conservative"
    return label


def _format_news_minutes(raw: Any) -> Optional[str]:
    """Normalise self-reported news minutes/hours into a descriptive string."""

    numeric = _as_float(raw)
    if numeric is None:
        text = str(raw).strip()
        return text or None
    if numeric <= 0:
        result = "0 minutes"
    elif numeric < 1:
        minutes = numeric * 60
        result = f"{minutes:.0f} minutes"
    elif numeric <= 6:
        result = f"{numeric:.1f} hours"
    else:
        result = f"{numeric:.0f} hours"
    return result


CUSTOM_FIELD_RENDERERS: Dict[str, Callable[[Any], Optional[str]]] = {
    "ideo1": _format_ideology,
    "ideo2": _format_ideology,
    "ideology": _format_ideology,
    "ideology_text": _format_ideology,
    "youtube_time": _format_news_minutes,
}


def _format_percentage_field(value: Any, precision: int = 0) -> Optional[str]:
    """Format a percentage-like field while handling 0-1 and 0-100 scales."""

    numeric = _as_float(value)
    if numeric is None:
        return None
    if abs(numeric) < 1.0:
        return _format_percentage(numeric, precision=precision)
    return _format_scale_percentage(numeric)


def _format_currency_field(value: Any) -> Optional[str]:
    """Format a currency-like field preserving existing dollar strings."""

    numeric = _as_float(value)
    if numeric is None:
        text = str(value).strip()
        if text.startswith("$"):
            return text
        return text or None
    return _format_currency(numeric)


def _format_state(field: str, value: Any) -> Optional[str]:
    """Map a state code to its canonical label when possible."""

    text = str(value).strip()
    if not text:
        return None
    normalized = _normalize_state_code(text)
    if normalized and normalized in STATE_FIPS_MAP:
        return STATE_FIPS_MAP[normalized]
    return text


def _normalize_mapping_lookup(mapping: Dict[str, str], key: str) -> Optional[str]:
    """Return the best-effort value from ``mapping`` handling numeric variants."""

    if key in mapping:
        return mapping[key]
    simplified = key.lstrip("0")
    if simplified in mapping:
        return mapping[simplified]
    if key.endswith(".0"):
        trimmed = key[:-2]
        if trimmed in mapping:
            return mapping[trimmed]
    try:
        numeric = float(key)
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric) and numeric.is_integer():
        candidate = str(int(numeric))
        if candidate in mapping:
            return mapping[candidate]
    formatted = f"{numeric:g}"
    if formatted in mapping:
        return mapping[formatted]
    return None


def _format_feeling_thermometer(raw: Any) -> Optional[str]:
    """Convert 0–100 feeling thermometer scores into descriptive phrases."""

    numeric = _as_float(raw)
    if numeric is None:
        text = str(raw).strip()
        return text or None
    clamped = max(0.0, min(100.0, numeric))
    score = int(round(clamped))
    if clamped <= 20:
        descriptor = "strongly unfavorable"
    elif clamped <= 40:
        descriptor = "unfavorable"
    elif clamped < 60:
        descriptor = "neutral"
    elif clamped < 80:
        descriptor = "favorable"
    else:
        descriptor = "strongly favorable"
    return f"{descriptor} ({score} out of 100)"


def _format_news_trust(raw: Any) -> Optional[str]:
    """Render fractional news-trust scores as qualitative descriptions."""

    numeric = _as_float(raw)
    if numeric is None:
        text = str(raw).strip()
        return text or None
    percent = numeric
    if abs(percent) <= 1.0:
        percent *= 100.0
    percent = max(0.0, min(100.0, percent))
    score = int(round(percent))
    if percent < 20:
        descriptor = "very low"
    elif percent < 40:
        descriptor = "low"
    elif percent < 60:
        descriptor = "moderate"
    elif percent < 80:
        descriptor = "high"
    else:
        descriptor = "very high"
    return f"{descriptor} (about {score}%)"


def format_field_value(field: str, value: Any) -> str:
    """
    Convert dataset field ``value`` into human-readable text based on ``field``.

    :param field: Field name from the dataset row.
    :param value: Raw value associated with ``field``.
    :returns: Human-friendly string or an empty string when nothing meaningful remains.
    """

    if is_nanlike(value):
        return ""
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        parts = [format_field_value(field, item) for item in value]
        parts = [part for part in parts if part]
        return ", ".join(parts)

    text = str(value).strip()
    if not text:
        return ""

    field_lower = field.lower()
    result: Optional[str] = None

    if field_lower in STATE_FIELD_NAMES:
        result = _format_state(field_lower, value)
    else:
        mapping = FIELD_VALUE_MAPS.get(field_lower)
        if mapping:
            key = text
            result = _normalize_mapping_lookup(mapping, key)
        elif field_lower in YT_FREQ_MAP and text in YT_FREQ_MAP:
            result = YT_FREQ_MAP[text]

    if result:
        return result

    if field_lower in POLITICAL_INTEREST_FIELDS:
        result = _format_percentage_field(value, precision=1)
    if not result and field_lower in FEELING_THERMOMETER_FIELDS:
        result = _format_feeling_thermometer(value)
    if not result and field_lower in NEWS_TRUST_FIELDS:
        result = _format_news_trust(value)
    if not result and field_lower in PERCENTAGE_FIELDS:
        result = _format_percentage_field(value)
    if not result and field_lower in CURRENCY_FIELDS:
        result = _format_currency_field(value)
    if not result and field_lower in SCALE_0_100_FIELDS:
        result = _format_percentage_field(value)
    if not result and field_lower in GUN_IMPORTANCE_FIELDS:
        key = text.rstrip(".0")
        result = GUN_IMPORTANCE_MAP.get(key)
    if not result:
        renderer = CUSTOM_FIELD_RENDERERS.get(field_lower)
        if renderer:
            result = renderer(value)

    return result or text


__all__ = [
    "format_field_value",
    "RACE_CODE_MAP",
    "EDUCATION_CODE_MAP",
    "EMPLOYMENT_CODE_MAP",
    "RELIGION_CODE_MAP",
    "STATE_FIPS_MAP",
]
