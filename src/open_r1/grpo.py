#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO for GRAIL (columns-only) + OPTIONAL Discriminator (same style as your prior working file)

What this script does (columns-only; no extra parsing/lookups):
  • Builds a USER prompt with:
      - PROFILE: viewer_profile_sentence (or synthesized from age/q26/q29/pid1/ideo1/q31/freq_youtube/college)
      - CURRENTLY WATCHING: current_video_title + current_video_id
      - HISTORY (prior only, most recent first): from watched_detailed_json with [watched/total] seconds
      - OPTIONS: slate_items_json (names preferred; fallback to ids)
  • The model must output ONLY the numeric option index (1..N) inside <answer>…</answer>.
  • The gold target is the index of the *next* chosen video (after current) within the slate.
  • Keeps only examples where:
      - slate_items_json is non-empty
      - the gold "next id" exists in that slate (so there is a ground-truth index 1..N)

Rewards:
  • Uses your YAML-defined rewards (e.g., pure_accuracy_reward).
  • If GAIL_USE != "0", appends a discriminator reward ("gail_reward") implemented the SAME WAY
    as your prior working script (simple classifier prob on the chosen action; no online training).

Env (optional):
  GRAIL_MAX_HISTORY (default 12)
  GAIL_USE=1/0 (default 1)
  GAIL_DISC_MODEL=distilbert-base-uncased
  GAIL_DEVICE=cuda:0|cpu (default auto; CPU if ZeRO/DeepSpeed detected)
  GAIL_LR=2e-5
  GAIL_ALPHA=1.0
  GAIL_WEIGHT=0.5   (used only if your YAML provided fewer weights than rewards)
"""

from __future__ import annotations
import os, sys, re, json, logging, math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import datasets
import transformers
from transformers import set_seed
from trl import TrlParser, ModelConfig, get_peft_config
from trl.trainer.grpo_trainer import GRPOTrainer

# --- Open-R1 plumbing ---
_TREES = "/n/fs/similarity/trees/src"
if _TREES not in sys.path:
    sys.path.insert(0, _TREES)

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.rewards import get_reward_funcs

logger = logging.getLogger(__name__)

# ---------- tiny helpers ----------
ANS_RE   = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
IDX_ONLY = re.compile(r'^\s*(?:option\s*)?(\d+)\s*$', re.I)

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

def _completion_text(x: Any) -> str:
    if isinstance(x, str): return x
    if isinstance(x, dict): return str(x.get("content", "")).strip()
    if isinstance(x, list) and x:
        for m in reversed(x):
            if isinstance(m, dict) and "content" in m:
                c = str(m.get("content","")).strip()
                if c: return c
        try: return " ".join(str(m.get("content","")).strip() for m in x if isinstance(m, dict))
        except Exception: pass
    return str(x)

def _parse_index_from_answer_block(text: str) -> Optional[int]:
    m = ANS_RE.search(text or "")
    s = (m.group(1).strip() if m else (text or "").strip())
    m2 = IDX_ONLY.match(s)
    if not m2: return None
    try: return int(m2.group(1))
    except Exception: return None

def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower().strip())


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


def _truncate_text(text: str, limit: int = 160) -> str:
    text = text.strip()
    if limit and limit > 3 and len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def _clean_text(value: Any, *, limit: Optional[int] = None) -> str:
    if _is_nanlike(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [_clean_text(v) for v in value]
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


def _truthy(value: Any) -> Optional[bool]:
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
    truth = _truthy(value)
    if truth is True:
        return yes
    if truth is False:
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
    text = _clean_text(value)
    return text or None

def _as_list_json(x: Any, default="[]") -> list:
    if isinstance(x, list): return x
    if isinstance(x, str):
        try: return json.loads(x or default)
        except Exception: return []
    return []

def _load_slate_items(ex: dict) -> List[dict]:
    arr = _as_list_json(ex.get("slate_items_json"))
    keep_keys = {
        "title",
        "id",
        "raw_id",
        "video_id",
        "video_title",
        "channel",
        "channel_title",
        "channel_name",
        "channel_id",
        "length_seconds",
        "duration_seconds",
        "duration",
        "watch_seconds",
        "score",
        "rank",
        "position",
        "reason",
        "source",
    }
    out: List[dict] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        cleaned: Dict[str, Any] = {}
        for key, value in it.items():
            if key in keep_keys and not _is_nanlike(value):
                cleaned[key] = value
        title = _clean_text(
            it.get("title") or it.get("name") or it.get("video_title") or cleaned.get("title"),
            limit=160,
        )
        vid = _clean_text(it.get("id") or it.get("raw_id") or it.get("video_id") or cleaned.get("id"))
        channel = _clean_text(
            it.get("channel_title") or it.get("channel_name") or it.get("channel") or cleaned.get("channel_title"),
            limit=120,
        )
        if title:
            cleaned["title"] = title
        if vid:
            cleaned["id"] = vid
        if channel:
            cleaned["channel_title"] = channel
        if cleaned.get("id") or cleaned.get("title"):
            out.append(cleaned)
    return out

def _gold_index_from_items(gold: str, items: List[dict]) -> int:
    gold = (gold or "").strip()
    if not gold or not items: return -1
    for i, it in enumerate(items, 1):
        if gold == it.get("id", ""): return i
    gc = _canon(gold)
    if gc:
        for i, it in enumerate(items, 1):
            if gc == _canon(it.get("title", "")): return i
    return -1

# ---------- synthesize a viewer one-liner if missing ----------
def _synthesize_viewer_sentence(ex: dict) -> str:
    bits: List[str] = []
    # age
    age = ex.get("age")
    try: age_i = int(age) if age not in (None, "", "nan") else None
    except Exception: age_i = None
    if isinstance(age_i, int) and age_i > 0:
        bits.append(f"{age_i}-year-old")
    # gender (q26)
    gender = str(ex.get("q26") or "").strip().lower()
    if   gender in {"man", "male"}: bits.append("man")
    elif gender in {"woman", "female"}: bits.append("woman")
    elif gender: bits.append(gender.title())
    # race (q29)
    race = str(ex.get("q29") or "").strip()
    if race and race.lower() != "nan": bits.append(race)
    # party/ideo (pid1 / ideo1 preferred)
    pid1  = str(ex.get("pid1") or "").strip()
    ideo1 = str(ex.get("ideo1") or "").strip()
    if pid1 and pid1.lower() != "nan":
        if ideo1 and ideo1.lower() != "nan": bits.append(f"{pid1} {ideo1}".lower())
        else: bits.append(pid1)
    elif ideo1 and ideo1.lower() != "nan":
        bits.append(ideo1.lower())
    # income (q31)
    inc = str(ex.get("q31") or "").strip()
    if inc and inc.lower() != "nan": bits.append(inc)
    # education (college -> True)
    college = str(ex.get("college") or "").strip().lower()
    if college in {"true","1","yes","y"}: bits.append("college-educated")
    # youtube frequency
    fy = str(ex.get("freq_youtube") or "").strip()
    if fy in {"0","1","2","3","4","5"}:
        fmap = {"0":"rarely","1":"occasionally","2":"a few times a month","3":"weekly","4":"several times a week","5":"daily"}
        bits.append(f"watches YouTube {fmap[fy]}")
    s = ", ".join(b for b in bits if b)
    return s if s else "(no profile provided)"

# ---------- build the user prompt purely from columns ----------
def _secs(x: Any) -> str:
    try: return f"{int(round(float(x)))}s"
    except Exception: return "?"

def _build_user_prompt_from_columns(ex: dict, max_hist: int = 12) -> str:
    """
    Build a structured, LLM-friendly prompt from dataset columns alone.

    Sections:
      • PROFILE: natural one-liner (uses viewer_profile_sentence or a synthetic fallback)
      • DEMOGRAPHICS / POLITICS / ISSUE STANCES / MEDIA HABITS / SESSION CONTEXT: rich key/value bullets
      • CURRENT VIDEO: title (+ channel / id when available)
      • HISTORY: prior videos, most recent first (respecting GRAIL_MAX_HISTORY)
      • OPTIONS: enumerated slate entries (titles with optional metadata)
    """
    show_ids = os.getenv("GRAIL_SHOW_IDS", "0") == "1"
    lines: List[str] = []

    # --------- helpers scoped to this builder ---------
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
        return _clean_text(value, limit=limit)

    # --------- PROFILE ---------
    viewer = _clean_text(ex.get("viewer_profile_sentence"))
    if not viewer:
        viewer = _synthesize_viewer_sentence(ex)
    lines.append("PROFILE:")
    lines.append(viewer or "(no profile provided)")

    # --------- DEMOGRAPHICS ---------
    demographics: List[str] = []

    age_val = _first_raw("age")
    age_text = _format_age(age_val)
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

    # --------- POLITICS ---------
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

    # --------- GUN POLICY ---------
    gun_section: List[str] = []

    gun_own_val = _first_raw("gun_own", "gunowner", "owns_gun")
    if gun_own_val is not None:
        gun_own_text = _format_yes_no(gun_own_val, yes="owns a gun", no="does not own a gun")
        if gun_own_text:
            gun_section.append(f"Gun ownership: {gun_own_text}")
        else:
            custom = _clean_text(gun_own_val)
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
            text = _clean_text(value, limit=200)
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
        text = _clean_text(value, limit=200)
        if not text:
            continue
        label = low[4:].replace("_", " ").strip().capitalize()
        if not label:
            continue
        gun_section.append(f"{label}: {text}")
        known_gun_keys.add(low)

    _append_section("GUN POLICY", gun_section)

    # --------- MINIMUM WAGE / ECONOMICS ---------
    wage_section: List[str] = []

    for key, label in MIN_WAGE_FIELD_LABELS.items():
        value = _first_raw(key)
        if value is None:
            continue
        text = _format_yes_no(value)
        if text is None:
            text = _clean_text(value, limit=220)
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
        text = _clean_text(value, limit=220)
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

    # --------- MEDIA HABITS ---------
    media_section: List[str] = []

    fy_raw = _first_raw("freq_youtube")
    if fy_raw is not None:
        code = str(fy_raw).strip()
        freq = YT_FREQ_MAP.get(code)
        if freq:
            media_section.append(f"YouTube frequency: {freq}")
        else:
            freq_text = _clean_text(fy_raw)
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

    # --------- SESSION CONTEXT ---------
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

    # --------- CURRENT VIDEO ---------
    current_title = _clean_text(ex.get("current_video_title"), limit=160)
    current_id = _clean_text(ex.get("current_video_id"))
    current_channel = _clean_text(ex.get("current_video_channel") or ex.get("current_video_channel_title"))
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

    # --------- HISTORY (prior only, most recent first) ---------
    vids = _as_list_json(ex.get("watched_vids_json"))
    det = _as_list_json(ex.get("watched_detailed_json"))

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
                    if isinstance(det[j], dict) and _clean_text(det[j].get("id")) == current_id:
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
            title = _clean_text(r.get("title") or r.get("name") or r.get("video_title"), limit=160)
            rid = _clean_text(r.get("id"))
            channel = _clean_text(r.get("channel_title") or r.get("channel"))
            ws = _secs(r.get("watch_seconds"))
            tl = _secs(r.get("total_length"))
            descriptor = f"[{ws}/{tl}] {title or '(untitled)'}"
            extras: List[str] = []
            if channel:
                extras.append(f"channel: {channel}")
            if show_ids and rid:
                extras.append(f"id: {rid}")
            if extras:
                descriptor = f"{descriptor} — {', '.join(extras)}"
            lines.append(f"{idx}. {descriptor}")

    # --------- OPTIONS (next-step slate) ---------
    items = _load_slate_items(ex)
    lines.append("")
    lines.append("OPTIONS:")
    if items:
        for i, it in enumerate(items, 1):
            if not isinstance(it, dict):
                continue
            title = _clean_text(it.get("title"), limit=160)
            option_id = _clean_text(it.get("id"))
            if not title and option_id:
                title = option_id
            channel = _clean_text(it.get("channel_title") or it.get("channel") or it.get("channel_name"))
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

# ----------------- Online discriminator (GPU + DS-safe) -----------------
class OnlineDiscriminator:
    def __init__(self, model_name: str, device: torch.device, lr: float = 2e-5):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
        self._model_name = model_name
        self._lr = lr
        self.device = device

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        cfg = AutoConfig.from_pretrained(model_name, num_labels=2)

        # IMPORTANT: keep the disc out of meta/zero-init paths
        #  - low_cpu_mem_usage=False prevents init_empty_weights/meta
        #  - device_map=None avoids any accelerate/DS device mapping
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=cfg,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,
        )
        # lock to the chosen CUDA device for this rank
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.model.to(self.device).train()

        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self._sanity_check_embeddings()

    def _sanity_check_embeddings(self):
        try:
            W = self.model.get_input_embeddings().weight
            if W is None or W.dim() != 2 or W.is_meta:
                raise RuntimeError("disc embedding not 2-D/materialized")
        except Exception:
            self._reload_clean()

    def _reload_clean(self):
        from transformers import AutoModelForSequenceClassification
        m = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.model = m.to(self.device).train()
        self.opt = optim.AdamW(self.model.parameters(), lr=self._lr)

    @torch.no_grad()
    def prob_positive(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0,), dtype=np.float32)
        texts = [t if isinstance(t, str) and t.strip() else "[PAD]" for t in texts]

        # guard device context on each call (some launchers change current device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self.model.eval()
        try:
            batch = self.tok(
                texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)

            W = self.model.get_input_embeddings().weight
            if W is None or W.dim() != 2 or W.is_meta:
                self._reload_clean()
                batch = {k: v.to(self.device) for k, v in batch.items()}

            logits = self.model(**batch).logits
            return logits.softmax(dim=-1)[:, 1].detach().cpu().numpy()
        except Exception:
            # fail-quiet: shape problems → zero shaping instead of crashing GRPO
            return np.zeros((len(texts),), dtype=np.float32)
        finally:
            self.model.train()

    def train_batch(self, texts: List[str], labels: List[int]) -> Optional[float]:
        if not texts:
            return None
        texts = [t if isinstance(t, str) and t.strip() else "[PAD]" for t in texts]
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        batch = self.tok(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)
        out = self.model(**batch, labels=y)
        loss = out.loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return float(loss.detach().cpu().item())


def _render_disc_text(viewer: str, state_text: str,
                      slate_items: List[dict], action_surface: str, action_id: Optional[str]) -> str:
    show_ids = os.getenv("GRAIL_DISC_SHOW_IDS", "0") == "1"

    names = [f"{i}. {(it.get('title') or (it.get('id') if show_ids else '') or '(untitled)')}"
             for i, it in enumerate(slate_items, 1)]

    parts = [
        f"VIEWER: {viewer or '(none)'}",
        "STATE:",
        state_text or "(none)",
        "SLATE (names):",
        *(names if names else ["(none)"]),
        # only include id lines if explicitly requested
        *([] if not show_ids else [
            "SLATE_IDS:", *( [f"{i}. {(it.get('id') or '(none)')}" for i, it in enumerate(slate_items, 1)] or ["(none)"] ),
            f"ACTION_ID: {action_id or '(none)'}",
        ]),
        f"ACTION_NAME: {action_surface or '(none)'}",
    ]
    return "\n".join(parts)

def make_gail_reward_fn(disc: Optional[OnlineDiscriminator], alpha: float = 1.0):
    """
    Train the discriminator online during TRAIN, and only score during EVAL.
    Train is gated by env GAIL_TRAIN=1 and not GAIL_EVAL_MODE=1.
    """
    def _reward(completions, answer, **kw):
        if disc is None:
            return [0.0] * len(completions)

        # Are we allowed to train right now?
        train_on = (os.getenv("GAIL_TRAIN", "1") == "1") and (os.getenv("GAIL_EVAL_MODE", "0") != "1")

        # Batch-ify fields we need
        def _aslist(x, n): return x if isinstance(x, list) else [x]*n
        n        = len(completions)
        viewerL  = _aslist(kw.get("viewer_profile") or "", n)
        stateL   = _aslist(kw.get("state_text") or "", n)
        itemsL   = _aslist(kw.get("slate_items") or [], n)
        goldIdL  = _aslist(kw.get("gold_id") or "", n)
        goldIdxL = _aslist(kw.get("gold_index") or -1, n)

        # Build policy texts + validity mask
        policy_texts, valid_mask, chosen_idx = [], [], []
        for comp, v, s, its in zip(completions, viewerL, stateL, itemsL):
            mi = _parse_index_from_answer_block(_completion_text(comp))
            if isinstance(mi, int) and 1 <= mi <= len(its):
                choice  = its[mi-1]
                surface = (choice.get("title") or choice.get("id") or "")
                policy_texts.append(_render_disc_text(v or "", s or "", its or [], surface, choice.get("id")))
                valid_mask.append(True)
                chosen_idx.append(mi)
            else:
                policy_texts.append("[PAD]")
                valid_mask.append(False)
                chosen_idx.append(-1)

        # Inference: P(expert|state, choice)
        probs = disc.prob_positive(policy_texts)

        # Optional online training
        if train_on:
            pos_texts, pos_labels = [], []
            neg_texts, neg_labels = [], []

            for v, s, its, g_id, g_idx, pol_text, ok, mi in zip(
                viewerL, stateL, itemsL, goldIdL, goldIdxL, policy_texts, valid_mask, chosen_idx
            ):
                # Build a positive "expert" text if gold is in slate
                g_id = (g_id or "").strip()
                if isinstance(its, list) and its and g_id:
                    surface_pos = None
                    for it in its:
                        if g_id == (it.get("id") or ""):
                            surface_pos = (it.get("title") or it.get("id") or "")
                            break
                    if surface_pos is not None:
                        pos_texts.append(_render_disc_text(v or "", s or "", its or [], surface_pos, g_id))
                        pos_labels.append(1)

                # Use policy choice as a negative if it was valid and != gold index
                if ok and isinstance(g_idx, int) and g_idx >= 1 and mi != g_idx:
                    neg_texts.append(pol_text)
                    neg_labels.append(0)

            train_texts  = pos_texts + neg_texts
            train_labels = pos_labels + neg_labels
            if train_texts:
                try:
                    disc.train_batch(train_texts, train_labels)
                except Exception as _e:
                    # swallow and keep going; disc is auxiliary
                    pass

        # Final rewards: zero for invalid indices, else alpha * prob
        out = []
        for pr, ok in zip(probs, valid_mask):
            out.append(float(alpha * pr) if ok else 0.0)
        return out

    return _reward

# ---------- find "next" gold id from history ----------
def _derive_next_from_history(ex: dict, current_id: str) -> str:
    vids = _as_list_json(ex.get("watched_vids_json"))
    if current_id and isinstance(vids, list) and vids:
        try:
            i = vids.index(current_id)
            if i + 1 < len(vids):
                nxt = vids[i + 1]
                if isinstance(nxt, str) and nxt.strip():
                    return nxt.strip()
        except ValueError:
            pass
    det = _as_list_json(ex.get("watched_detailed_json"))
    if current_id and isinstance(det, list) and det:
        for j, r in enumerate(det):
            if isinstance(r, dict) and (r.get("id") or "").strip() == current_id:
                if j + 1 < len(det):
                    nxt = (det[j + 1].get("id") or "").strip()
                    if nxt:
                        return nxt
                break
    return ""

def _get_gold_next_id(ex: dict, sol_key: Optional[str]) -> str:
    # Allow explicit label if present (and not current id)
    if sol_key and sol_key not in {"current_video_id", "current_id"}:
        v = ex.get(sol_key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ["next_video_id", "clicked_id", "video_id", "label", "answer"]:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    cur = (ex.get("current_video_id") or "").strip()
    return _derive_next_from_history(ex, cur)

# ---------- map one row -> conversation dict ----------
def _row_to_example(ex: dict, system_prompt: Optional[str], sol_key: Optional[str], max_hist: int = 12) -> Optional[dict]:
    items = _load_slate_items(ex)
    if not items: return None

    # GOLD = the "next" id (after current) and its index in the slate
    gold_id = _get_gold_next_id(ex, sol_key)
    gidx    = _gold_index_from_items(gold_id, items)
    if gidx < 1: return None

    user_msg = _build_user_prompt_from_columns(ex, max_hist=max_hist)
    sys_msg  = system_prompt or (
        "You are choosing EXACTLY ONE item from a short slate for a specific viewer.\n"
        "Think briefly in <think>…</think>, then output ONLY the option NUMBER (1..N) inside <answer>…</answer>.\n"
        "Format (STRICT): <think>…</think><answer>3</answer>"
    )
    nopts = int(ex.get("n_options") or len(items) or 0)

    return {
        "prompt": [{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
        "answer": str(gidx),                 # GOLD = numeric index as string (so pure_accuracy can compare)
        "gold_index": gidx,                  # int for analysis
        "gold_id": gold_id,                  # id for analysis
        "n_options": nopts,
        "viewer_profile": str(ex.get("viewer_profile_sentence") or _synthesize_viewer_sentence(ex)),
        "state_text": user_msg,
        "slate_items": items,
        "slate_text":  str(ex.get("slate_text") or ""),
        # keep histories for reference (not needed by disc in this "same style" version)
        "watched_detailed_json": _as_list_json(ex.get("watched_detailed_json")),
        "watched_vids_json":     _as_list_json(ex.get("watched_vids_json")),
        "current_video_id":      str(ex.get("current_video_id") or ""),
        "current_video_title":   str(ex.get("current_video_title") or ""),
        "task": "GRAIL",
        "is_replay": False, "accuracy": 0.0, "mix_group_id": -1, "mix_copy_idx": -1,
    }

# ---------- main ----------
def main(script_args, training_args, model_args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(training_args.seed)
    training_args.steps_per_generation = 3
    training_args.num_iterations       = 3

    raw = get_dataset(script_args)
    tok = get_tokenizer(model_args, training_args)

    solution_key = getattr(script_args, "dataset_solution_column", None)
    max_hist = int(os.environ.get("GRAIL_MAX_HISTORY", "12") or "12")

    # Keep rows with a usable slate AND a gold index in that slate
    def _ok(ex):
        items = _load_slate_items(ex)
        if not items:
            return False
        gold = _get_gold_next_id(ex, solution_key)
        if not gold:
            return False
        return _gold_index_from_items(gold, items) >= 1

    raw = raw.filter(_ok)

    # Build conversation from columns only
    ds = raw.map(lambda ex: _row_to_example(ex, training_args.system_prompt, solution_key, max_hist=max_hist),
                 load_from_cache_file=False)

    # Drop any failed rows (should be none; map returns dict)
    if "__drop__" in ds[script_args.dataset_train_split].column_names:
        for split in list(ds.keys()):
            mask = [not b for b in ds[split]["__drop__"]]
            ds[split] = ds[split].select([i for i, keep in enumerate(mask) if keep])

    # Keep only what trainer/rewards need (+ a few for logs/analysis)
    keep_cols = {
        "prompt","answer","gold_index","gold_id","n_options",
        "viewer_profile","state_text","slate_text","slate_items",
        "watched_detailed_json","watched_vids_json",
        "current_video_id","current_video_title",
        "task","is_replay","accuracy","mix_group_id","mix_copy_idx"
    }
    for split in list(ds.keys()):
        drop = [c for c in ds[split].column_names if c not in keep_cols]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    # ===== Rewards =====
    reward_fns = []
    try:
        reward_fns = get_reward_funcs(script_args, ref_model=None, tokenizer=tok)
    except Exception as e:
        logger.warning("[rewards] get_reward_funcs failed: %s", e)

    # Optional GAIL shaping
    use_gail = os.environ.get("GAIL_USE", "1") != "0"
    if use_gail:
        def _pick_disc_device() -> torch.device:
            # Respect explicit override; map "cuda" → local rank
            dev_str = os.getenv("GAIL_DEVICE", "")
            if dev_str.strip().lower() == "cuda":
                lrk = os.getenv("LOCAL_RANK")
                if lrk is not None:
                    dev_str = f"cuda:{int(lrk)}"
            if dev_str:
                return torch.device(dev_str)

            # Default: per-rank CUDA if available, else CPU
            if torch.cuda.is_available():
                lrk = os.getenv("LOCAL_RANK")
                return torch.device(f"cuda:{int(lrk)}") if lrk is not None else torch.device("cuda:0")
            return torch.device("cpu")

        dev        = _pick_disc_device()
        disc_model = os.environ.get("GAIL_DISC_MODEL", "distilbert-base-uncased")
        disc_lr    = float(os.environ.get("GAIL_LR", "2e-5"))
        gail_alpha = float(os.environ.get("GAIL_ALPHA", "1.0"))

        disc = OnlineDiscriminator(disc_model, dev, lr=disc_lr)
        gail_fn = make_gail_reward_fn(disc, alpha=gail_alpha)
        gail_fn.__name__ = "gail_reward"
        reward_fns.append(gail_fn)
        logger.info("GAIL shaping ENABLED (alpha=%.3f, model=%s, device=%s)", gail_alpha, disc_model, str(dev))
    else:
        logger.info("GAIL shaping DISABLED")

    # Weights: respect YAML; if you appended GAIL and lengths mismatch, auto-extend with GAIL_WEIGHT
    weights = getattr(training_args, "reward_weights", None)
    if weights is None:
        if use_gail and len(reward_fns) == 2:
            gail_w = float(os.environ.get("GAIL_WEIGHT", "0.5"))
            training_args.reward_weights = [1.0, gail_w]
        else:
            training_args.reward_weights = [1.0] * len(reward_fns)
    elif len(weights) != len(reward_fns):
        if use_gail and len(weights) == (len(reward_fns) - 1):
            gail_w = float(os.environ.get("GAIL_WEIGHT", "0.5"))
            training_args.reward_weights = list(weights) + [gail_w]
        else:
            raise ValueError(
                f"reward_weights length ({len(weights)}) != number of rewards ({len(reward_fns)}). "
                "Update YAML or set $GAIL_WEIGHT to auto-extend."
            )

    # Normalize non-negative weights
    ws = [max(0.0, float(w)) for w in training_args.reward_weights]
    s  = sum(ws) or 1.0
    training_args.reward_weights = [w / s for w in ws]
    logger.info("[grpo] rewards=%s weights=%s",
                [getattr(f, "__name__", f.__class__.__name__) for f in reward_fns],
                training_args.reward_weights)

    # Model + trainer
    model = get_model(model_args, training_args)
    # Let YAML control sampling; ensure generate returns dicts
    model.generation_config.return_dict_in_generate = True
    model.config.return_dict_in_generate = True

    train_split = script_args.dataset_train_split
    eval_ds = None
    if getattr(training_args, "do_eval", False) and script_args.dataset_test_split in ds:
        full = ds[script_args.dataset_test_split]
        n_keep = max(1, int(0.1 * len(full)))
        eval_ds = full.shuffle(seed=training_args.seed).select(range(n_keep))

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fns,
        train_dataset=ds[train_split],
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
        processing_class=tok,
    )

    # Train/Eval/Save
    from transformers.trainer_utils import get_last_checkpoint
    last = (training_args.resume_from_checkpoint
            or (get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None))
    train_result = trainer.train(resume_from_checkpoint=last)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    if getattr(training_args, "do_eval", False) and eval_ds is not None:
        os.environ["GAIL_EVAL_MODE"] = "1"          # ← freeze disc: score only
        metrics = trainer.evaluate()
        os.environ["GAIL_EVAL_MODE"] = "0"          # ← optional reset
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if getattr(training_args, "push_to_hub", False):
        trainer.push_to_hub(dataset_name=script_args.dataset_name, tags=["open-r1"])

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
