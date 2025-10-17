from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

TRUE_STRINGS = {"1", "true", "t", "yes", "y"}
FALSE_STRINGS = {"0", "false", "f", "no", "n"}

YT_FREQ_MAP = {
    "0": "rarely",
    "1": "occasionally",
    "2": "a few times a month",
    "3": "weekly",
    "4": "several times a week",
    "5": "daily",
}

GUN_FIELD_LABELS: Dict[str, str] = {
    "right_to_own_importance": "Right-to-own importance",
    "assault_ban": "Supports assault weapons ban",
    "handgun_ban": "Supports handgun ban",
    "concealed_safe": "Believes concealed carry is safe",
    "stricter_laws": "Supports stricter gun laws",
    "gun_index": "Gun index",
    "gun_index_2": "Gun index (alt)",
    "gun_enthusiasm": "Gun enthusiasm",
    "gun_importance": "Gun importance",
    "gun_priority": "Gun policy priority",
    "gun_policy": "Gun policy stance",
    "gun_identity": "Gun identity",
}

MIN_WAGE_FIELD_LABELS: Dict[str, str] = {
    "minwage_text_r_w1": "Minimum wage stance (wave 1, inferred)",
    "minwage_text_r_w2": "Minimum wage stance (wave 2, inferred)",
    "minwage_text_r_w3": "Minimum wage stance (wave 3, inferred)",
    "minwage_text_w1": "Minimum wage stance (wave 1, survey)",
    "minwage_text_w2": "Minimum wage stance (wave 2, survey)",
    "mw_index_w1": "Minimum wage support index (wave 1)",
    "mw_index_w2": "Minimum wage support index (wave 2)",
    "minwage15_w1": "$15 minimum wage support (wave 1)",
    "minwage15_w2": "$15 minimum wage support (wave 2)",
    "mw_support_w1": "Supports wage increase (wave 1)",
    "mw_support_w2": "Supports wage increase (wave 2)",
    "minwage_importance": "Minimum wage importance",
    "minwage_priority": "Minimum wage priority",
}


def as_list_json(x: Any, default: str = "[]") -> List[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            value = json.loads(x or default)
            return value if isinstance(value, list) else []
        except Exception:
            return []
    try:
        import pyarrow as pa  # type: ignore

        if isinstance(x, pa.Array):
            return x.to_pylist()
    except Exception:
        pass
    return []


def secs(x: Any) -> str:
    try:
        return f"{int(round(float(x)))}s"
    except Exception:
        return "?"


def _is_nanlike(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float):
        if math.isnan(x):
            return True
    try:
        import pandas as pd  # type: ignore

        if isinstance(x, pd._libs.missing.NAType):  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    s = str(x).strip().lower()
    return s in {"", "nan", "none", "null", "n/a", "na"}


def is_nanlike(value: Any) -> bool:
    return _is_nanlike(value)


def _truncate_text(text: str, limit: int = 160) -> str:
    text = text.strip()
    if limit and limit > 3 and len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def clean_text(value: Any, *, limit: Optional[int] = None) -> str:
    if _is_nanlike(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [clean_text(v) for v in value]
        parts = [p for p in parts if p]
        if not parts:
            return ""
        text = "; ".join(parts)
        return _truncate_text(text, limit or len(text))
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            value = int(value)
    text = str(value).strip()
    if not text:
        return ""
    if limit:
        text = _truncate_text(text, limit)
    return text


def truthy(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False
    return None


def _format_yes_no(value: Any, *, yes: str = "yes", no: str = "no") -> Optional[str]:
    verdict = truthy(value)
    if verdict is True:
        return yes
    if verdict is False:
        return no
    return None


def _format_age(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        age = int(float(str(value).strip()))
        if age > 0:
            return str(age)
    except Exception:
        pass
    text = clean_text(value)
    return text or None


def synthesize_viewer_sentence(ex: Dict[str, Any]) -> str:
    bits: List[str] = []
    age_text = _format_age(ex.get("age"))
    if age_text:
        bits.append(f"{age_text}-year-old")

    gender = str(ex.get("q26") or ex.get("gender") or "").strip().lower()
    if gender in {"man", "male"}:
        bits.append("man")
    elif gender in {"woman", "female"}:
        bits.append("woman")
    elif gender:
        bits.append(gender.title())

    race = str(ex.get("q29") or ex.get("race") or "").strip()
    if race and not _is_nanlike(race):
        bits.append(race)

    pid1 = str(ex.get("pid1") or "").strip()
    ideo1 = str(ex.get("ideo1") or "").strip()
    if pid1 and pid1.lower() != "nan":
        if ideo1 and ideo1.lower() != "nan":
            bits.append(f"{pid1} {ideo1}".lower())
        else:
            bits.append(pid1)
    elif ideo1 and ideo1.lower() != "nan":
        bits.append(ideo1.lower())

    income = str(ex.get("q31") or ex.get("income") or "").strip()
    if income and income.lower() != "nan":
        bits.append(income)

    college = str(ex.get("college") or "").strip().lower()
    if college in TRUE_STRINGS:
        bits.append("college-educated")

    freq = str(ex.get("freq_youtube") or "").strip()
    if freq in YT_FREQ_MAP:
        bits.append(f"watches YouTube {YT_FREQ_MAP[freq]}")

    return ", ".join(bits) if bits else "(no profile provided)"


def build_user_prompt(ex: Dict[str, Any], max_hist: int = 12) -> str:
    show_ids = os.getenv("GRAIL_SHOW_IDS", "0") == "1"
    lines: List[str] = []

    viewer = clean_text(ex.get("viewer_profile_sentence"))
    if not viewer:
        viewer = synthesize_viewer_sentence(ex)
    lines.append("PROFILE:")
    lines.append(viewer or "(no profile provided)")

    def _append_section(title: str, entries: List[str]) -> None:
        if not entries:
            return
        lines.append("")
        lines.append(f"{title}:")
        for entry in entries:
            lines.append(f"- {entry}")

    def _first_raw(*keys: str):
        for key in keys:
            if key in ex and not _is_nanlike(ex[key]):
                return ex[key]
        return None

    def _first_text(*keys: str, limit: Optional[int] = None) -> str:
        value = _first_raw(*keys)
        if value is None:
            return ""
        return clean_text(value, limit=limit)

    demographics: List[str] = []
    age_text = _format_age(_first_raw("age"))
    if age_text:
        demographics.append(f"Age: {age_text}")
    gender_text = _first_text("gender", "q26")
    if gender_text:
        demographics.append(f"Gender: {gender_text}")
    race_text = _first_text("race", "ethnicity", "q29")
    if race_text:
        demographics.append(f"Race/ethnicity: {race_text}")

    location_parts: List[str] = []
    city_text = _first_text("city", "city_name")
    if city_text:
        location_parts.append(city_text)
    state_text = _first_text("state", "state_residence", "state_name")
    if state_text:
        location_parts.append(state_text)
    county_text = _first_text("county", "county_name")
    if county_text:
        location_parts.append(f"{county_text} County")
    zip_text = _first_text("zip3")
    if zip_text:
        location_parts.append(f"ZIP3 {zip_text}")
    if location_parts:
        demographics.append("Location: " + ", ".join(location_parts))

    educ_text = _first_text("education", "educ", "education_level", "college_desc")
    if educ_text:
        demographics.append(f"Education: {educ_text}")
    college_text = _format_yes_no(_first_raw("college"), yes="college educated", no="not college educated")
    if college_text:
        demographics.append(f"College: {college_text}")

    income_text = _first_text("q31", "income", "household_income")
    if income_text:
        demographics.append(f"Household income: {income_text}")

    employment_text = _first_text("employment_status", "employment", "labor_force")
    if employment_text:
        demographics.append(f"Employment: {employment_text}")
    occupation_text = _first_text("occupation")
    if occupation_text:
        demographics.append(f"Occupation: {occupation_text}")

    marital_text = _first_text("marital_status", "married")
    if marital_text:
        demographics.append(f"Marital status: {marital_text}")

    children_text = _format_yes_no(_first_raw("children_in_house", "kids_household"), yes="yes", no="no")
    if children_text:
        demographics.append(f"Children in household: {children_text}")
    household_size = _first_text("household_size")
    if household_size:
        demographics.append(f"Household size: {household_size}")

    religion_text = _first_text("religion", "relig_affiliation", "religious_affiliation")
    if religion_text:
        demographics.append(f"Religion: {religion_text}")
    attendance_text = _first_text("relig_attend", "church_attend", "service_attendance")
    if attendance_text:
        demographics.append(f"Religious attendance: {attendance_text}")

    veteran_text = _format_yes_no(_first_raw("veteran", "military_service"), yes="yes", no="no")
    if veteran_text:
        demographics.append(f"Veteran: {veteran_text}")

    _append_section("DEMOGRAPHICS", demographics)

    politics: List[str] = []
    party_text = _first_text("pid1", "party_id", "party_registration")
    if party_text:
        politics.append(f"Party identification: {party_text}")
    party_lean_text = _first_text("pid2", "party_id_lean", "party_lean")
    if party_lean_text:
        politics.append(f"Party lean: {party_lean_text}")
    ideology_text = _first_text("ideo1", "ideo2", "ideology")
    if ideology_text:
        politics.append(f"Ideology: {ideology_text}")
    pol_interest = _first_text("pol_interest", "interest_politics", "political_interest")
    if pol_interest:
        politics.append(f"Political interest: {pol_interest}")
    vote_2016 = _first_text("vote_2016")
    if vote_2016:
        politics.append(f"Voted in 2016: {vote_2016}")
    vote_2020 = _first_text("vote_2020")
    if vote_2020:
        politics.append(f"Voted in 2020: {vote_2020}")
    vote_2024 = _first_text("vote_2024", "vote_intent_2024", "vote_2024_intention")
    if vote_2024:
        politics.append(f"Vote intention 2024: {vote_2024}")
    trump_approve = _first_text("trump_approve", "trump_job_approval")
    if trump_approve:
        politics.append(f"Trump approval: {trump_approve}")
    biden_approve = _first_text("biden_approve", "biden_job_approval")
    if biden_approve:
        politics.append(f"Biden approval: {biden_approve}")
    civic_engagement = _first_text("civic_participation", "volunteering", "civic_activity")
    if civic_engagement:
        politics.append(f"Civic engagement: {civic_engagement}")

    _append_section("POLITICS", politics)

    gun_section: List[str] = []
    gun_own_val = _first_raw("gun_own", "gunowner", "owns_gun")
    if gun_own_val is not None:
        gun_own_text = _format_yes_no(gun_own_val, yes="owns a gun", no="does not own a gun")
        if gun_own_text:
            gun_section.append(f"Gun ownership: {gun_own_text}")
        else:
            custom = clean_text(gun_own_val)
            if custom:
                gun_section.append(f"Gun ownership: {custom}")

    known_gun_keys = {"gun_own", "gunowner", "owns_gun"}
    for key, label in GUN_FIELD_LABELS.items():
        value = _first_raw(key)
        if value is None:
            continue
        known_gun_keys.add(key.lower())
        text = _format_yes_no(value)
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
        if _is_nanlike(value):
            continue
        text = clean_text(value, limit=200)
        if not text:
            continue
        label = low[4:].replace("_", " ").strip().capitalize()
        if not label:
            continue
        gun_section.append(f"{label}: {text}")
        known_gun_keys.add(low)

    _append_section("GUN POLICY", gun_section)

    wage_section: List[str] = []
    for key, label in MIN_WAGE_FIELD_LABELS.items():
        value = _first_raw(key)
        if value is None:
            continue
        text = _format_yes_no(value)
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
        if _is_nanlike(value):
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

    _append_section("MINIMUM WAGE", wage_section)

    media_section: List[str] = []
    fy_raw = _first_raw("freq_youtube")
    if fy_raw is not None:
        code = str(fy_raw).strip()
        freq = YT_FREQ_MAP.get(code)
        if freq:
            media_section.append(f"YouTube frequency: {freq}")
        else:
            freq_text = clean_text(fy_raw)
            if freq_text:
                media_section.append(f"YouTube frequency: {freq_text}")

    binge_text = _format_yes_no(_first_raw("binge_youtube"), yes="yes", no="no")
    if binge_text:
        media_section.append(f"Binge watches YouTube: {binge_text}")

    media_labels = [
        ("q8", "Favorite channels"),
        ("fav_channels", "Favorite channels"),
        ("q78", "Popular channels followed"),
        ("media_diet", "Media diet"),
        ("news_consumption", "News consumption"),
        ("news_sources", "News sources"),
        ("news_sources_top", "Top news sources"),
        ("news_frequency", "News frequency"),
        ("platform_use", "Platform usage"),
        ("social_media_use", "Social media use"),
        ("news_trust", "News trust"),
    ]
    seen_media_labels: set[str] = set()
    for key, label in media_labels:
        text = _first_text(key, limit=220)
        if not text:
            continue
        if label in seen_media_labels and label in {"Favorite channels"}:
            continue
        media_section.append(f"{label}: {text}")
        seen_media_labels.add(label)

    _append_section("MEDIA HABITS", media_section)

    session_section: List[str] = []
    issue_text = _first_text("issue")
    if issue_text:
        session_section.append(f"Issue: {issue_text}")
    issue_detail = _first_text("issue_detail")
    if issue_detail:
        session_section.append(f"Issue detail: {issue_detail}")
    issue_source = _first_text("issue_source")
    if issue_source:
        session_section.append(f"Issue source: {issue_source}")
    session_id_text = _first_text("session_id")
    if session_id_text:
        session_section.append(f"Session ID: {session_id_text}")
    step_text = _first_text("display_step", "step_index")
    if step_text:
        session_section.append(f"Step: {step_text}")
    slate_source = _first_text("slate_source")
    if slate_source:
        session_section.append(f"Slate source: {slate_source}")
    n_options_text = _first_text("n_options")
    if n_options_text:
        session_section.append(f"Options in slate: {n_options_text}")

    _append_section("SESSION CONTEXT", session_section)

    current_title = clean_text(ex.get("current_video_title"), limit=160)
    current_id = clean_text(ex.get("current_video_id"))
    current_channel = clean_text(ex.get("current_video_channel") or ex.get("current_video_channel_title"))
    current_line_parts: List[str] = []
    if current_title:
        current_line_parts.append(current_title)
    if current_channel:
        current_line_parts.append(f"channel: {current_channel}")
    if show_ids and current_id:
        current_line_parts.append(f"id: {current_id}")
    elif not current_title and current_id:
        current_line_parts.append(f"id: {current_id}")
    if current_line_parts:
        lines.append("")
        lines.append("CURRENT VIDEO:")
        lines.append(" — ".join(current_line_parts))

    vids = as_list_json(ex.get("watched_vids_json"))
    det = as_list_json(ex.get("watched_detailed_json"))

    def _last_index(xs, val):
        if not isinstance(xs, list) or val is None:
            return None
        idx = None
        for i, v in enumerate(xs):
            if v == val:
                idx = i
        return idx

    cur_idx = None
    if current_id:
        cur_idx = _last_index(vids, current_id)
        if cur_idx is None and isinstance(det, list):
            for j in range(len(det) - 1, -1, -1):
                try:
                    if isinstance(det[j], dict) and clean_text(det[j].get("id")) == current_id:
                        cur_idx = j
                        break
                except Exception:
                    continue
    if cur_idx is None and isinstance(vids, list) and vids:
        cur_idx = len(vids) - 1

    prior: List[dict] = []
    if isinstance(det, list) and cur_idx is not None and cur_idx > 0:
        prior = det[:cur_idx]

    if prior:
        lines.append("")
        lines.append("HISTORY (most recent first):")
        limit = max_hist if max_hist and max_hist > 0 else len(prior)
        recent = list(reversed(prior))[:limit]
        for idx, r in enumerate(recent, 1):
            if not isinstance(r, dict):
                continue
            title = clean_text(r.get("title") or r.get("name") or r.get("video_title"), limit=160)
            rid = clean_text(r.get("id"))
            channel = clean_text(r.get("channel_title") or r.get("channel"))
            ws = secs(r.get("watch_seconds"))
            tl = secs(r.get("total_length"))
            descriptor = f"[{ws}/{tl}] {title or '(untitled)'}"
            extras: List[str] = []
            if channel:
                extras.append(f"channel: {channel}")
            if show_ids and rid:
                extras.append(f"id: {rid}")
            if extras:
                descriptor = f"{descriptor} — {', '.join(extras)}"
            lines.append(f"{idx}. {descriptor}")

    items = ex.get("slate_items_json")
    items = as_list_json(items)
    lines.append("")
    lines.append("OPTIONS:")
    if items:
        for i, it in enumerate(items, 1):
            if not isinstance(it, dict):
                continue
            title = clean_text(it.get("title"), limit=160)
            option_id = clean_text(it.get("id"))
            if not title and option_id:
                title = option_id
            channel = clean_text(it.get("channel_title") or it.get("channel") or it.get("channel_name"))
            duration_raw = it.get("length_seconds") or it.get("duration_seconds") or it.get("duration")
            duration_text = ""
            try:
                if duration_raw is not None and str(duration_raw).strip():
                    duration_val = float(duration_raw)
                    if duration_val > 0:
                        duration_text = f"{int(round(duration_val))}s"
            except Exception:
                duration_text = ""
            parts = [title or "(untitled)"]
            if channel:
                parts.append(f"channel: {channel}")
            if duration_text:
                parts.append(f"duration: {duration_text}")
            if show_ids and option_id:
                parts.append(f"id: {option_id}")
            elif not title and option_id:
                parts.append(f"id: {option_id}")
            lines.append(f"{i}. {' — '.join(parts)}")
    else:
        lines.append("(no options provided)")

    return "\n".join(lines)


__all__ = [
    "as_list_json",
    "build_user_prompt",
    "clean_text",
    "is_nanlike",
    "secs",
    "synthesize_viewer_sentence",
    "truthy",
]
