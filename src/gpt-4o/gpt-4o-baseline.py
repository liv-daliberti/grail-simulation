#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate GPT-4o on next-video choice using the EXACT SAME prompt shape as GRPO.
- No GRPO, no discriminator, no YAML.
- Channels OFF by default (env toggles preserved).
- Falls back to HF streaming if local disk space is insufficient.
- Saves per-example JSONL and a metrics JSON (overall + by sequence position).

Edit DATASET_* if needed. Set your Azure Sandbox creds below.
"""

from __future__ import annotations

import os, re, sys, json, time, argparse, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from itertools import islice

import numpy as np
from datasets import load_dataset, DownloadConfig
from openai import AzureOpenAI

# ───────────────────────── Azure OpenAI (Princeton Sandbox) ────────────────────
SANDBOX_API_KEY  = "1e30d0e4d7564ba984e8adff48053009"   # <<< set here
SANDBOX_ENDPOINT = "https://api-ai-sandbox.princeton.edu/"
SANDBOX_API_VER  = "2025-03-01-preview"
DEPLOYMENT_NAME  = "gpt-4o"

# Export into env (as requested)
os.environ["SANDBOX_API_KEY"]  = SANDBOX_API_KEY
os.environ["SANDBOX_ENDPOINT"] = SANDBOX_ENDPOINT
os.environ["SANDBOX_API_VER"]  = SANDBOX_API_VER
os.environ["DEPLOYMENT_NAME"]  = DEPLOYMENT_NAME

_client = AzureOpenAI(
    api_key        = SANDBOX_API_KEY,
    azure_endpoint = SANDBOX_ENDPOINT,
    api_version    = SANDBOX_API_VER,
)

def ds_call(messages: List[Dict[str,str]], max_tokens: int, temperature: float) -> str:
    resp = _client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

# ─────────────────────────── Data config (no YAML) ─────────────────────────────
DATASET_NAME    = "od2961/grail-interactions"  # HF dataset repo id
TRAIN_SPLIT     = "train"
EVAL_SPLIT      = "validation"
PROMPT_COLUMN   = "state_text"                 # optional; becomes CONTEXT:
SOLUTION_COLUMN = "video_id"                   # gold “current” video id for this row

# ─────────────────────────── System prompt (EXACT) ─────────────────────────────
SYSTEM_PROMPT = """You are choosing EXACTLY ONE item from a short slate for a specific viewer.

Input you will see:
  • Viewer profile and optional context/history
  • An "OPTIONS:" list with items numbered 1..N
    – Each item is shown as either a title (preferred) or an id

Your job:
  • Think briefly in <think>…</think> using the viewer’s profile, context, and options.
  • Compare the top 2–3 candidates, then choose the single best option.
  • Never invent new items; choose only from the given OPTIONS list.

Output format (STRICT):
  • First output your hidden reasoning in <think>…</think>.
    – In your thinking, reference candidates by their numbers and names (or ids) to justify the choice.
  • Then output ONLY the chosen option’s NUMBER inside <answer>…</answer>.
    – Do NOT output the name, id, or any extra text—ONLY the number.
    – Do NOT include punctuation, quotes, or a period after the number.

Examples of valid <answer>:
  <answer>
  3
  </answer>

Examples of INVALID <answer> (never do these):
  <answer>3.</answer>                 ← trailing period
  <answer>"3"</answer>                ← quoted
  <answer>Option 3</answer>           ← extra words
  <answer>Parkland …</answer>         ← name instead of number
  You only have 100 tokens to think and 50 tokens to answer.
"""

# ───────────────────────────── Helpers / canon ─────────────────────────────────
ANS_TAG   = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
INDEX_ONLY= re.compile(r'^\s*(?:option\s*)?(\d+)\s*$', re.I)
YTID_RE   = re.compile(r'([A-Za-z0-9_-]{11})')
CANON_RE  = re.compile(r"[^a-z0-9]+")
# ---- Default title sources (used even if no env vars are set) ----
DEFAULT_TITLE_DIRS = [
    "/n/fs/similarity/trees/data/results/capsule-5416997-data/recommendation trees/trees_gun",
    "/n/fs/similarity/trees/data/results/capsule-5416997-data/recommendation trees/trees_wage",
]
def _bin_nopts(n: int) -> str:
    if n <= 1: return "1"
    if n == 2: return "2"
    if n == 3: return "3"
    if n == 4: return "4"
    return "5+"

def _canon(s: str) -> str:
    return CANON_RE.sub("", (s or "").lower().strip())

def _canon_vid(vid: str) -> str:
    if not isinstance(vid, str): return ""
    m = YTID_RE.search(vid)
    return m.group(1) if m else vid.strip()

def _is_nanlike(x: Optional[str]) -> bool:
    if x is None: return True
    s = str(x).strip().lower()
    return s in {"", "nan", "none", "null", "na", "n/a"}

def _pick_ci(d: dict, *alts: str) -> Optional[str]:
    if not isinstance(d, dict): return None
    lower = {k.lower(): k for k in d.keys()}
    for a in alts:
        k = lower.get(a.lower())
        if k is not None:
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None

# ───────────────────── Full profile extractors (EXACT as in GRPO) ─────────────
def _truthy(x) -> bool:
    if x is None: return False
    if isinstance(x, (int, float)): return x != 0
    s = str(x).strip().lower()
    return s in {"1","true","t","yes","y"}

_INCOME_HINT_KEYS = ["income", "income_bracket", "q30", "q31", "q32", "q34"]
_INCOME_PAT = re.compile(r"\$\s?\d{1,3}(?:,\d{3})?(?:\s*-\s*\$\s?\d{1,3}(?:,\d{3})?)?")

def _extract_income(ex: dict) -> Optional[str]:
    for k in _INCOME_HINT_KEYS:
        v = ex.get(k)
        if isinstance(v, str) and not _is_nanlike(v):
            s = v.strip()
            if _INCOME_PAT.search(s) or "income" in s.lower():
                return s
    try:
        for k, v in ex.items():
            if isinstance(v, str) and not _is_nanlike(v) and _INCOME_PAT.search(v):
                return v.strip()
    except Exception:
        pass
    if "income_gt50k" in ex:
        return ">$50k household income" if bool(ex.get("income_gt50k")) else "≤$50k household income"
    return None

def _extract_party(ex: dict) -> Optional[str]:
    pid_txt = ex.get("pid")
    if isinstance(pid_txt, str) and not _is_nanlike(pid_txt):
        s = pid_txt.strip().lower()
        if "dem" in s: return "Democratic"
        if "rep" in s or "gop" in s: return "Republican"
        if "independent" in s: return "Independent"
        if "libertarian" in s: return "Libertarian"
        if "green" in s: return "Green"
        if "closer to the democratic" in s: return "Democratic-leaning"
        if "closer to the republican" in s: return "Republican-leaning"
        return pid_txt.strip()
    vals = []
    for k in ("pid1","pid2","pid3","pid4"):
        v = ex.get(k)
        try:
            if v is not None and str(v).strip() != "":
                vals.append(float(v))
        except Exception:
            pass
    if vals:
        m = sum(vals)/len(vals)
        if m <= 3.0:  return "Democratic-leaning"
        if m >= 5.0:  return "Republican-leaning"
        return "Independent/Other"
    return None

def _format_ideology(ideo: Any) -> Optional[str]:
    if ideo is None: return None
    s = str(ideo).strip()
    if _is_nanlike(s): return None
    try:
        x = float(s)
        if x <= 1.5: return "Extremely liberal"
        if x <= 2.5: return "Liberal"
        if x <= 3.5: return "Slightly liberal"
        if x <= 4.5: return "Moderate"
        if x <= 5.5: return "Slightly conservative"
        if x <= 6.5: return "Conservative"
        return "Extremely conservative"
    except Exception:
        pass
    s_l = s.lower()
    if "extreme" in s_l and "lib" in s_l: return "Extremely liberal"
    if "lib" in s_l: return "Liberal"
    if "moderate" in s_l or "centrist" in s_l: return "Moderate"
    if "conservative" in s_l and "extreme" in s_l: return "Extremely conservative"
    if "conservative" in s_l: return "Conservative"
    return s

_MARITAL_KEYS = ["marital", "marital_status", "q18", "q56", "q77"]
def _extract_marital(ex: dict) -> Optional[str]:
    for k in _MARITAL_KEYS:
        v = ex.get(k)
        if isinstance(v, str) and not _is_nanlike(v):
            s = v.strip().lower()
            if "married" in s: return "Married"
            if "partner" in s or "cohabit" in s: return "Living with partner"
            if "single" in s: return "Single"
            if "divorced" in s: return "Divorced"
            if "widow" in s: return "Widowed"
            if "separated" in s: return "Separated"
            return s
    return None

_RACE_TEXT_FIELDS = ["q26","q27","q28","q26_3_text","q33_14_text","race","ethnicity"]
_RACE_MAP = {
    "white": "White", "caucasian": "White", "caucasian/white": "White",
    "black": "Black", "africanamerican": "Black", "african american": "Black",
    "asian": "Asian",
    "hispanic": "Hispanic/Latino", "latino": "Hispanic/Latino", "latina": "Hispanic/Latino", "latinx": "Hispanic/Latino",
    "nativeamerican": "Native American", "americanindian": "Native American",
    "pacificislander": "Pacific Islander",
    "middleeastern": "Middle Eastern",
    "other": "Other", "mixed": "Multiracial", "twoormore": "Multiracial",
    "prefernottoanswer": "Unspecified", "unknown": "Unspecified",
}
def _normalize_race_token(s: str) -> Optional[str]:
    if not s: return None
    key = _canon(s)
    if key in _RACE_MAP: return _RACE_MAP[key]
    for k, v in _RACE_MAP.items():
        if k in key:
            return v
    return s.strip()

def _extract_race(ex: dict) -> Optional[str]:
    try:
        if _truthy(ex.get("white")) and _truthy(ex.get("black")): return "Multiracial"
        if _truthy(ex.get("white")): return "White"
        if _truthy(ex.get("black")): return "Black"
    except Exception:
        pass
    for f in _RACE_TEXT_FIELDS:
        val = ex.get(f)
        if isinstance(val, str) and not _is_nanlike(val):
            norm = _normalize_race_token(val)
            if norm and not _is_nanlike(norm):
                return norm
    return None

def _humanize_profile(ex: dict) -> str:
    """EXACT humanized sentence used by GRPO."""
    bits: List[str] = []
    age = ex.get("age", None)
    if isinstance(age, (int, float)) and age > 0:
        bits.append(f"{int(age)}-year-old")

    race = _extract_race(ex)
    female = _truthy(ex.get("female"))
    male   = _truthy(ex.get("male"))
    gender = "woman" if (female and not male) else ("man" if (male and not female) else None)

    if race and gender: bits.append(f"{race} {gender}")
    else:
        if gender: bits.append(gender)
        if race and not gender: bits.append(race)

    marital = _extract_marital(ex)
    if marital and not _is_nanlike(marital):
        bits.append(marital.lower() if marital in {"Married","Single","Divorced","Widowed","Separated"} else marital)

    party = _extract_party(ex)
    ideo  = _format_ideology(ex.get("ideo"))
    if party and ideo: bits.append(f"{party.lower()} {ideo.lower()}")
    elif party: bits.append(party)
    elif ideo: bits.append(ideo.lower())

    income = _extract_income(ex)
    if income and not _is_nanlike(income): bits.append(income)

    if _truthy(ex.get("college")) and not any("degree" in (income or "").lower() for _ in [0]):
        bits.append("college-educated")

    yt = str(ex.get("freq_youtube", "")).strip()
    if yt in {"0","1","2","3","4","5"}:
        freq_map = {"0":"rarely","1":"occasionally","2":"a few times a month","3":"weekly","4":"several times a week","5":"daily"}
        bits.append(f"watches YouTube {freq_map.get(yt, 'regularly')}")

    sent = ", ".join([b for b in bits if b and not _is_nanlike(b)])
    return sent if sent else "(no profile provided)"

# (Structured profile text used only for discriminator in GRPO — kept for parity)
def _build_profile_text(ex: dict) -> str:
    lines: List[str] = []
    race    = _extract_race(ex);    marital = _extract_marital(ex)
    party   = _extract_party(ex);   ideo    = _format_ideology(ex.get("ideo"))
    income  = _extract_income(ex)
    if race:    lines.append(f"race: {race}")
    if marital: lines.append(f"marital: {marital}")
    if party:   lines.append(f"party: {party}")
    if ideo:    lines.append(f"ideology: {ideo}")
    if income:  lines.append(f"income: {income}")
    default_cols = [
        "age","gender","female","male","college","income_gt50k","pid","ideo",
        "pol_interest","freq_youtube","fav_channels","popular_channels",
        "gun_index","gun_index_2","minwage_text_r_w1"
    ]
    cols = os.environ.get("GRAIL_PROFILE_COLS", ",".join(default_cols))
    wanted = [c.strip() for c in cols.split(",") if c.strip()]
    for col in wanted:
        if col in ex and ex[col] is not None:
            val = str(ex[col]).strip()
            if not _is_nanlike(val): lines.append(f"{col}: {val}")
    if len(lines) > 24: lines = lines[:24] + ["..."]
    return "\n".join(lines) if lines else "(none)"

# ───────────────────────────── Title index (optional) ──────────────────────────
def _split_env_list(s: str | None) -> list[str]:
    if not s: return []
    return [p for chunk in re.split(r'[:,\s]+', s) if (p:=chunk.strip())]

def _iter_csv_files_from_env() -> list[str]:
    files: list[str] = []

    # Env-provided directories (crawl for any *.csv)
    for d in _split_env_list(os.environ.get("GRAIL_TITLE_DIRS")):
        if os.path.isdir(d):
            for root, _, fnames in os.walk(d):
                for f in fnames:
                    if f.lower().endswith(".csv"):
                        files.append(os.path.join(root, f))

    # Env-provided globs
    for pat in _split_env_list(os.environ.get("GRAIL_TITLE_GLOB")):
        try:
            from glob import glob
            for p in glob(pat):
                if os.path.isfile(p) and p.lower().endswith(".csv"):
                    files.append(p)
        except Exception:
            pass

    # Env-provided explicit CSV list
    for p in _split_env_list(os.environ.get("GRAIL_TITLE_CSVS")):
        if os.path.isfile(p) and p.lower().endswith(".csv"):
            files.append(p)

    # ALWAYS include defaults (crawl both trees)
    for d in DEFAULT_TITLE_DIRS:
        if os.path.isdir(d):
            for root, _, fnames in os.walk(d):
                for f in fnames:
                    if f.lower().endswith(".csv"):
                        files.append(os.path.join(root, f))

    # de-dup and keep order
    seen, out = set(), []
    for p in files:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def _guess_cols(header: List[str]) -> Tuple[Optional[str], Optional[str]]:
    cand_ids = ['originId','ytid','video_id','youtube_id','videoId','origin_id','id']
    cand_tit = ['originTitle','title','video_title','name']
    lower = {c.lower(): c for c in header}
    id_col = next((lower[c] for c in map(str.lower, cand_ids) if c in lower), None)
    ti_col = next((lower[c] for c in map(str.lower, cand_tit) if c in lower), None)
    return id_col, ti_col

_TITLE_INDEX: Optional[Dict[str,str]] = None
def _build_title_index() -> Dict[str,str]:
    import csv
    idx: Dict[str,str] = {}
    for path in _iter_csv_files_from_env():
        try:
            with open(path, 'r', encoding='utf-8', newline='') as f:
                rd = csv.DictReader(f)
                if not rd.fieldnames: continue
                id_col, ti_col = _guess_cols(rd.fieldnames)
                if not id_col or not ti_col: continue
                for row in rd:
                    vid = _canon_vid(row.get(id_col, "") or "")
                    tit = (row.get(ti_col, "") or "").strip()
                    if vid and tit and vid not in idx: idx[vid] = tit
        except Exception:
            continue
    return idx

def _title_for(vid: str) -> Optional[str]:
    global _TITLE_INDEX
    if _TITLE_INDEX is None:
        _TITLE_INDEX = _build_title_index()
        print(f"[title-index] loaded {len(_TITLE_INDEX)} titles from CSV")
    return _TITLE_INDEX.get(_canon_vid(vid))

# ───────────────────────── CURRENTLY WATCHING / HISTORY / SLATE ───────────────
def _extract_now_watching(ex: dict) -> Tuple[str,str] | None:
    vid = _pick_ci(ex, "video_id","videoId")
    if vid and not _is_nanlike(vid):
        tit = (_pick_ci(ex, "current_video_title","now_playing_title","watching_title",
                           "currentVideoTitle","nowPlayingTitle","watchingTitle",
                           "now_title","current_title","meta_originTitle") or _title_for(vid) or "")
        return ((tit or "(untitled)"), vid)
    title = _pick_ci(ex, "current_video_title","now_playing_title","watching_title",
                        "currentVideoTitle","nowPlayingTitle","watchingTitle","now_title","current_title")
    vid   = _pick_ci(ex, "current_video_id","now_playing_id","watching_id",
                        "currentVideoId","nowPlayingId","watchingId",
                        "now_id","current_id","originId","video_id","videoId")
    if (title and not _is_nanlike(title)) or (vid and not _is_nanlike(vid)):
        if _is_nanlike(title) and vid: title = _title_for(vid) or ""
        return ((title or "(untitled)"), (vid or ""))
    tj = ex.get("trajectory_json"); obj = None
    if isinstance(tj, str) and tj.strip():
        try:
            obj = json.loads(tj)
            for key in ("current","now","active","playing","nowPlaying","currentVideo","watching"):
                cur = obj.get(key)
                if isinstance(cur, dict):
                    vid2 = (_pick_ci(cur, "video_id","id","videoId") or "").strip()
                    tit2 = (_pick_ci(cur, "title","video_title","name","videoTitle") or "").strip()
                    if vid2 or tit2:
                        if _is_nanlike(tit2) and vid2: tit2 = _title_for(vid2) or ""
                        return ((tit2 or "(untitled)"), (vid2 or ""))
        except Exception:
            obj = None
    if obj:
        for key in ("history","actions","events","steps","log","recent"):
            arr = obj.get(key)
            if isinstance(arr, list) and arr:
                for it in reversed(arr):
                    if not isinstance(it, dict): continue
                    typ = str(it.get("type") or it.get("action") or "").lower()
                    if typ and typ not in {"open","click","play","watch","select"}: continue
                    vid3 = (_pick_ci(it, "video_id","id","videoId") or "").strip()
                    tit3 = (_pick_ci(it, "title","video_title","name","videoTitle") or "").strip()
                    if vid3 or tit3:
                        if _is_nanlike(tit3) and vid3: tit3 = _title_for(vid3) or ""
                        return ((tit3 or "(untitled)"), (vid3 or ""))
        order = obj.get("order")
        if isinstance(order, list) and order:
            def _end_ms(x):
                try: return float(x.get("end_ms") or -1)
                except: return -1.0
            cand = max(order, key=_end_ms)
            vid4 = (_pick_ci(cand, "video_id","id","videoId") or "").strip()
            tit4 = (_pick_ci(cand, "title","video_title","name","videoTitle") or "").strip()
            if vid4 or tit4:
                if _is_nanlike(tit4) and vid4: tit4 = _title_for(vid4) or ""
                return ((tit4 or "(untitled)"), (vid4 or ""))
    return None

def _get_current_pointer_from_order(ex: dict) -> Tuple[Optional[int], Optional[float]]:
    vi = ex.get("video_index")
    try:
        if vi is not None and str(vi).strip() != "": return int(vi), None
    except Exception: pass
    cur_idx, cur_end = None, None
    now = _extract_now_watching(ex); now_title = now[0] if now else ""; now_id = now[1] if now else ""
    tj = ex.get("trajectory_json")
    if isinstance(tj, str) and tj.strip():
        try:
            obj = json.loads(tj); order = obj.get("order")
            if isinstance(order, list) and order:
                key_id, key_title = _canon_vid(now_id or ""), _canon(now_title or "")
                for it in order:
                    vid = str(_pick_ci(it, "video_id","id","videoId") or "")
                    tit = str(_pick_ci(it, "title","video_title","name","videoTitle") or "")
                    if (key_id and _canon_vid(vid)==key_id) or (key_title and _canon(tit)==key_title):
                        cur_idx = it.get("idx")
                        try: cur_end = float(it.get("end_ms") or -1)
                        except: cur_end = None
                        break
                if cur_idx is None and cur_end is None:
                    it = max(order, key=lambda x: float(x.get("end_ms") or -1))
                    cur_idx = it.get("idx")
                    try: cur_end = float(it.get("end_ms") or -1)
                    except: cur_end = None
        except Exception:
            pass
    return cur_idx, cur_end

def _extract_full_history_from_order(ex: dict, up_to_idx=None, up_to_end_ms=None, include_current=False) -> List[dict]:
    out: List[dict] = []
    tj = ex.get("trajectory_json")
    if not (isinstance(tj, str) and tj.strip()): return out
    try:
        obj = json.loads(tj); order = obj.get("order")
        if not isinstance(order, list): return out
        for it in order:
            if not isinstance(it, dict): continue
            vid   = _pick_ci(it, "video_id","id","videoId") or ""
            title = _pick_ci(it, "title","video_title","name","videoTitle") or ""
            if _is_nanlike(title) and vid: title = _title_for(vid) or ""
            rec = {
                "idx": it.get("idx"),
                "video_id": vid,
                "title": title,
                "watch_seconds": it.get("watch_seconds"),
                "total_length": it.get("total_length"),
                "start_ms": it.get("start_ms"),
                "end_ms": it.get("end_ms"),
            }
            out.append(rec)
        def _key(r):
            idx = r.get("idx")
            if isinstance(idx, int): return (0, idx)
            try: ems = float(r.get("end_ms") or -1)
            except: ems = -1.0
            return (1, ems)
        out.sort(key=_key)
        if up_to_idx is not None:
            out = [r for r in out if (r.get("idx") is None) or (r["idx"] < up_to_idx or (include_current and r["idx"] == up_to_idx))]
        elif up_to_end_ms is not None:
            try:
                thr = float(up_to_end_ms)
                out = [r for r in out if (r.get("end_ms") is None) or (float(r["end_ms"]) < thr or (include_current and float(r["end_ms"]) <= thr))]
            except Exception:
                pass
    except Exception:
        pass
    return out

def _extract_slate_items(ex: dict) -> List[Tuple[str,str]]:
    """
    Return a slate as [(title, video_id)].
    If slate_text lines are opaque IDs, resolve titles via _title_for when possible.
    Falls back to trajectory_json['order'] with the same title mapping.
    """
    items: List[Tuple[str,str]] = []

    st = ex.get("slate_text")
    if isinstance(st, str) and st.strip():
        for line in st.splitlines():
            s = line.strip()
            if not s or s == "-":
                continue
            m = re.match(r"^\s*(?:-|\d+\s*[\.\)])\s*(.+)$", s)
            surface = m.group(1).strip() if m else s

            vid = _canon_vid(surface)
            if len(vid) == 11:
                title = _title_for(vid) or ""
                items.append((title, vid))
            else:
                items.append((surface, ""))

    if not items:
        tj = ex.get("trajectory_json")
        if isinstance(tj, str) and tj.strip():
            try:
                obj = json.loads(tj)
                order = obj.get("order")
                if isinstance(order, list):
                    for it in order:
                        if not isinstance(it, dict):
                            continue
                        raw_id = str(_pick_ci(it, "video_id","id","videoId") or "").strip()
                        title  = str(_pick_ci(it, "title","video_title","name","videoTitle") or "").strip()
                        if _is_nanlike(title) and raw_id:
                            title = _title_for(raw_id) or ""
                        if raw_id or title:
                            items.append((title or "(untitled)", raw_id))
            except Exception:
                pass

    seen, out = set(), []
    for t, vid in items:
        key = _canon(vid) or _canon(t)
        if key and key not in seen:
            seen.add(key)
            out.append((t, vid))
    return out


def make_conversation_record(ex: dict) -> Dict[str, Any]:
    raw_context = str(ex.get(PROMPT_COLUMN, "") or "").strip()
    if _is_nanlike(raw_context): raw_context = ""

    viewer_profile_sentence = _humanize_profile(ex)
    profile_block = _build_profile_text(ex)  # parity only

    now = _extract_now_watching(ex)
    if now:
        now_title, now_id = now
        now_line_user  = (now_title or now_id or "(untitled)")
        now_line_state = f"{now_title or '(untitled)'}{(' — id: '+now_id) if now_id else ''}"
    else:
        now_title = now_id = ""
        now_line_user  = "(none)"
        now_line_state = "(none)"

    if str(now_line_user).strip().lower() == "nan":
        now_line_user = "(none)"

    cur_idx, cur_end = _get_current_pointer_from_order(ex)
    prior_seq = _extract_full_history_from_order(ex, up_to_idx=cur_idx, up_to_end_ms=cur_end, include_current=False)

    now_id_c    = _canon_vid((now_id if isinstance(now_id, str) else "") or "")
    now_title_c = _canon((now_title if isinstance(now_title, str) else "") or "")
    def _is_now_row(r):
        rid = _canon_vid((r.get("video_id") or "")); rti = _canon((r.get("title") or ""))
        return (now_id_c and rid == now_id_c) or (now_title_c and rti == now_title_c)
    prior_seq = [r for r in prior_seq if not _is_now_row(r)]

    def _fmt_sec(x) -> str:
        try: return f"{int(round(float(x)))}s"
        except Exception: return "?"
    def _format_hist_lines(seq: List[dict]) -> List[str]:
        lines: List[str] = []
        for r in seq:
            idx = r.get("idx"); ws = r.get("watch_seconds"); tl = r.get("total_length")
            tit = (r.get("title") or "").strip() or "(untitled)"
            left = []
            if idx is not None: left.append(str(idx))
            if ws is not None or tl is not None: left.append(f"{_fmt_sec(ws)}/{_fmt_sec(tl)}")
            prefix = f"[{' • '.join(left)}] " if left else ""
            lines.append(f"- {prefix}{tit}")
        return lines

    max_hist = int(os.environ.get("GRAIL_MAX_HISTORY", "8"))
    nonfull_hist_lines = list(reversed(_format_hist_lines(prior_seq)))[:max_hist] if prior_seq else []

    slate_pairs = _extract_slate_items(ex)
    options_lines = []
    for i, (title, vid) in enumerate(slate_pairs, start=1):
        surf = title if (title and title != "(untitled)") else (vid or "(untitled)")
        options_lines.append(f"{i}. {surf}")

    # gold index (1-based)
    gold_raw = str(ex.get(SOLUTION_COLUMN, "")).strip()
    gold_idx = -1
    if slate_pairs:
        for i, (t, vid) in enumerate(slate_pairs, start=1):
            if gold_raw and (gold_raw == vid or _canon(gold_raw) == _canon(t)):
                gold_idx = i; break

    user_lines: List[str] = []
    user_lines.append(f"Viewer: {viewer_profile_sentence}.")
    if raw_context: user_lines.append(f"CONTEXT: {raw_context}")
    user_lines.append("\nCURRENTLY WATCHING:")
    user_lines.append(now_line_user)

    show_full = (str(os.environ.get("GRAIL_HISTORY_FULL","0")).lower() in {"1","true","t","yes","y"} or
                 str(os.environ.get("GRAIL_HISTORY_MODE_FULL","0")).lower() in {"1","true","t","yes","y"} or
                 int(os.environ.get("GRAIL_MAX_HISTORY","8")) <= 0)

    full_lines: List[str] = []
    if show_full:
        full_seq = _extract_full_history_from_order(ex, include_current=(str(os.environ.get("GRAIL_HISTORY_INCLUDES_CURRENT","0")).lower() in {"1","true","t","yes","y"}))
        full_lines = _format_hist_lines(full_seq)
        if full_lines:
            user_lines.append("\nHISTORY (full sequence):")
            user_lines.extend(full_lines)
    else:
        if nonfull_hist_lines:
            user_lines.append("\nHISTORY (most recent first):")
            user_lines.extend(nonfull_hist_lines)

    user_lines.append("\nOPTIONS:")
    user_lines.extend(options_lines if options_lines else ["(no options provided)"])
    user_lines.append("\nAfter thinking in <think>, choose exactly one candidate from OPTIONS and return ONLY its NUMBER in <answer>.")
    user_msg = "\n".join(user_lines)

    # position index (0-based if present)
    pos_idx = ex.get("video_index")
    try:
        pos_idx = int(pos_idx) if pos_idx is not None else -1
    except Exception:
        pos_idx = -1

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "gold_index": gold_idx,
        "n_options": len(slate_pairs),
        "position_index": pos_idx,
    }

# ─────────────────────────────── Eval loop ────────────────────────────────────
def _bucket_from_pos(pos_idx: int) -> str:
    if pos_idx < 0: return "unknown"
    if pos_idx == 0: return "1"
    if pos_idx == 1: return "2"
    if pos_idx == 2: return "3"
    if pos_idx == 3: return "4"
    return "5+"

def _extract_answer_text(raw: str) -> str:
    m = ANS_TAG.search(raw)
    return (m.group(1).strip() if m else raw.strip())

def parse_index_from_output(raw: str) -> Optional[int]:
    m = ANS_TAG.search(raw)
    if m:
        s = m.group(1).strip()
        n = INDEX_ONLY.match(s)
        return int(n.group(1)) if n else None
    tail = "\n".join(raw.strip().splitlines()[-4:])
    for line in reversed(tail.splitlines()):
        line = line.strip()
        n = INDEX_ONLY.match(line)
        if n:
            return int(n.group(1))
    return None

def run_eval(args):
    """
    Evaluate GPT-4o on next-video choice with richer diagnostics:
      • By-position accuracy (existing)
      • By n_options (1,2,3,4,5+) histogram + accuracy + parsed/format rate
      • Single vs Multi (n_options==1 vs >1) accuracy + parsed/format rate on multi only
      • Gold index distribution + "most-frequent gold index" baseline
      • Expected random baseline (mean 1/n_options over eligible rows)
    """
    import logging, os, json, time
    from pathlib import Path
    from itertools import islice
    import numpy as np
    from datasets import load_dataset, DownloadConfig

    logging.info("Loading dataset %s", DATASET_NAME)

    # Make sure HF caches go somewhere you control
    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os.environ.setdefault("HF_HOME", args.cache_dir)

    # Try normal (cached) mode first; fall back to streaming on disk error
    use_streaming = False
    try:
        ds = load_dataset(
            DATASET_NAME,
            cache_dir=args.cache_dir,
            download_config=DownloadConfig(resume_download=True, max_retries=2),
        )
    except Exception as e:
        msg = str(e)
        if "Not enough disk space" in msg or "Insufficient space" in msg:
            logging.warning("Low disk space detected; falling back to streaming mode.")
            use_streaming = True
        else:
            raise

    # pick eval split
    if use_streaming:
        eval_split = EVAL_SPLIT
        try:
            data_iter = load_dataset(DATASET_NAME, split=eval_split, streaming=True)
        except Exception:
            for alt in ("validation", "eval", "test"):
                try:
                    data_iter = load_dataset(DATASET_NAME, split=alt, streaming=True)
                    eval_split = alt
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError("No eval split available for streaming.")
        if args.eval_max and args.eval_max > 0:
            data_iter = islice(data_iter, args.eval_max)
        n_eval_target = None
    else:
        if EVAL_SPLIT in ds:
            eval_split = EVAL_SPLIT
        else:
            for alt in ("validation", "eval", "test"):
                if alt in ds:
                    eval_split = alt
                    break
            else:
                raise ValueError(f"No eval split found in {list(ds.keys())}")
        full = ds[eval_split]
        if args.eval_max and args.eval_max > 0:
            data = full.select(range(min(args.eval_max, len(full))))
        else:
            data = full
        data_iter = iter(data)
        n_eval_target = len(data)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl    = out_dir / f"gpt4o_eval_{eval_split}.jsonl"
    metrics_json = out_dir / f"gpt4o_eval_{eval_split}_metrics.json"
    if out_jsonl.exists() and not args.overwrite:
        print(f"[SKIP] Output exists (use --overwrite): {out_jsonl}")
    w = open(out_jsonl, "w", encoding="utf-8")

    # -------------------- counters --------------------
    correct_overall = 0
    eligible_overall = 0
    parsed_ok = 0
    format_ok = 0

    # position buckets (existing)
    buckets = ["1","2","3","4","5+","unknown"]
    seen_b = {b: 0 for b in buckets}
    elig_b = {b: 0 for b in buckets}
    corr_b = {b: 0 for b in buckets}

    # n_options buckets + single/multi split
    def _bin_nopts(n: int) -> str:
        if n <= 1: return "1"
        if n == 2: return "2"
        if n == 3: return "3"
        if n == 4: return "4"
        return "5+"

    opts_buckets   = ["1","2","3","4","5+"]
    seen_opts_b    = {b: 0 for b in opts_buckets}
    elig_opts_b    = {b: 0 for b in opts_buckets}
    corr_opts_b    = {b: 0 for b in opts_buckets}
    parsed_opts_b  = {b: 0 for b in opts_buckets}
    format_opts_b  = {b: 0 for b in opts_buckets}

    seen_single = seen_multi = 0
    elig_single = elig_multi = 0
    corr_single = corr_multi = 0
    format_ok_multi = parsed_ok_multi = 0

    # gold index distribution / baselines
    gold_hist = {}            # {gold_index: count} over eligible rows
    all_gold_indices = []     # list of gold indexes (eligible rows)
    all_n_options    = []     # list of n_options (eligible rows)

    # local parser fallback: parse <answer>NUMBER</answer>, else tail lines
    def _parse_index_from_output(raw: str) -> Optional[int]:
        m = ANS_TAG.search(raw)
        if m:
            s = m.group(1).strip()
            mm = INDEX_ONLY.match(s)
            if mm:
                try: return int(mm.group(1))
                except: return None
        tail = "\n".join(raw.strip().splitlines()[-4:])
        for line in reversed(tail.splitlines()):
            mm2 = INDEX_ONLY.match(line.strip())
            if mm2:
                try: return int(mm2.group(1))
                except: return None
        return None

    def safe_div(n, d): return (n / d) if d else 0.0

    # -------------------- loop --------------------
    t0 = time.time()
    n_seen = 0
    for ex in data_iter:
        n_seen += 1
        rec = make_conversation_record(ex)  # -> {"prompt", "gold_index", "n_options", "position_index"}
        messages = rec["prompt"]
        gold_idx = int(rec.get("gold_index", -1))
        nopts    = int(rec.get("n_options", 0))
        pos_idx  = int(rec.get("position_index", -1))
        pbucket  = _bucket_from_pos(pos_idx)
        seen_b[pbucket] += 1

        # call model
        try:
            raw = ds_call(messages, max_tokens=args.max_tokens, temperature=args.temperature)
        except Exception as e:
            raw = f"(error: {e})"

        # parse/format status
        is_formatted = bool(ANS_TAG.search(raw))
        if is_formatted:
            format_ok += 1

        idx = _parse_index_from_output(raw)
        if idx is not None:
            parsed_ok += 1

        # n_options bucket tallies
        nbucket = _bin_nopts(nopts)
        seen_opts_b[nbucket] += 1
        if is_formatted:  format_opts_b[nbucket] += 1
        if idx is not None: parsed_opts_b[nbucket] += 1

        # single vs multi tallies
        if nopts == 1:
            seen_single += 1
        else:
            seen_multi += 1
            if is_formatted:    format_ok_multi += 1
            if idx is not None: parsed_ok_multi += 1

        # eligibility & correctness
        eligible = (gold_idx > 0 and nopts > 0)
        if eligible:
            eligible_overall += 1
            elig_b[pbucket] += 1
            elig_opts_b[nbucket] += 1

            gold_hist[gold_idx] = gold_hist.get(gold_idx, 0) + 1
            all_gold_indices.append(gold_idx)
            all_n_options.append(nopts)

        is_correct = eligible and (idx is not None) and (idx == gold_idx)
        if is_correct:
            correct_overall += 1
            corr_b[pbucket] += 1
            corr_opts_b[nbucket] += 1

        if eligible:
            if nopts == 1:
                elig_single += 1
                if is_correct: corr_single += 1
            else:
                elig_multi += 1
                if is_correct: corr_multi += 1

        # write row
        out_row = {
            "messages": messages,
            "gpt_output": raw,
            "parsed_index": idx,
            "gold_index": gold_idx,
            "n_options": nopts,
            "correct": bool(is_correct),
            "eligible": bool(eligible),
            "position_index": pos_idx,
            "position_bucket": pbucket,
        }
        w.write(json.dumps(out_row, ensure_ascii=False) + "\n")

        # progress
        if n_seen % 25 == 0:
            elapsed = time.time() - t0
            acc = safe_div(correct_overall, eligible_overall)
            denom = n_eval_target if n_eval_target is not None else n_seen
            print(f"[eval] {n_seen}/{denom}  acc={acc:.3f}  parsed={safe_div(parsed_ok, n_seen):.3f}  fmt={safe_div(format_ok, n_seen):.3f}  {elapsed:.1f}s")

    w.close()

    # -------------------- aggregate metrics --------------------
    n_eval_final = n_seen
    overall_acc  = safe_div(correct_overall, eligible_overall)
    fmt_rate     = safe_div(format_ok, n_eval_final)
    prs_rate     = safe_div(parsed_ok, n_eval_final)

    # by position (existing)
    pos_stats = {
        b: {
            "n_seen": int(seen_b[b]),
            "n_eligible": int(elig_b[b]),
            "correct": int(corr_b[b]),
            "accuracy": safe_div(corr_b[b], elig_b[b]),
        }
        for b in buckets
    }

    # by n_options bucket (new)
    by_n_options = {
        "hist_seen":        {b: int(seen_opts_b[b])   for b in opts_buckets},
        "hist_eligible":    {b: int(elig_opts_b[b])   for b in opts_buckets},
        "hist_correct":     {b: int(corr_opts_b[b])   for b in opts_buckets},
        "accuracy":         {b: safe_div(corr_opts_b[b], elig_opts_b[b]) for b in opts_buckets},
        "parsed_rate":      {b: safe_div(parsed_opts_b[b], seen_opts_b[b]) for b in opts_buckets},
        "format_rate":      {b: safe_div(format_opts_b[b], seen_opts_b[b]) for b in opts_buckets},
    }

    # single vs multi (new) incl. parsed/format on multi only
    split_single_vs_multi = {
        "n_single": int(seen_single),
        "n_multi": int(seen_multi),
        "eligible_single": int(elig_single),
        "eligible_multi": int(elig_multi),
        "accuracy_single": safe_div(corr_single, elig_single),
        "accuracy_multi":  safe_div(corr_multi,  elig_multi),
        "parsed_rate_multi": safe_div(parsed_ok_multi, max(1, seen_multi)),
        "format_rate_multi": safe_div(format_ok_multi, max(1, seen_multi)),
    }

    # gold index distribution + most-frequent-index baseline (new)
    gold_index_distribution = {str(k): int(v) for k, v in sorted(gold_hist.items())}
    if gold_hist:
        top_idx = max(gold_hist.items(), key=lambda kv: kv[1])[0]
        baseline_correct = sum(1 for gi in all_gold_indices if gi == top_idx)
        baseline_acc = safe_div(baseline_correct, eligible_overall)
    else:
        top_idx = None
        baseline_acc = 0.0

    baseline_most_frequent_gold_index = {
        "top_index": top_idx,
        "count": int(gold_hist.get(top_idx, 0) if top_idx is not None else 0),
        "accuracy": baseline_acc,
    }

    # expected random baseline (mean 1/n_options over eligible rows) (new)
    random_baseline_expected_accuracy = float(np.mean([1.0/n for n in all_n_options])) if all_n_options else 0.0

    metrics = {
        "model": DEPLOYMENT_NAME,
        "dataset": DATASET_NAME,
        "split": eval_split,
        "n_total": int(n_eval_final),
        "n_eligible": int(eligible_overall),
        "accuracy_overall": overall_acc,
        "parsed_rate": prs_rate,
        "format_rate": fmt_rate,
        "position_stats": pos_stats,
        # new sections:
        "by_n_options": by_n_options,
        "split_single_vs_multi": split_single_vs_multi,
        "gold_index_distribution": gold_index_distribution,
        "baseline_most_frequent_gold_index": baseline_most_frequent_gold_index,
        "random_baseline_expected_accuracy": random_baseline_expected_accuracy,
        "notes": "Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.",
    }

    with open(metrics_json, "w", encoding="utf-8") as mj:
        json.dump(metrics, mj, ensure_ascii=False, indent=2)

    print(f"[DONE] split={eval_split}  n={n_eval_final}  eligible={eligible_overall} "
          f"accuracy={overall_acc:.4f}  parsed_ok={prs_rate:.3f}  format_rate={fmt_rate:.3f}")
    print(f"[WROTE] per-example: {out_jsonl}")
    print(f"[WROTE] metrics:     {metrics_json}")

# ─────────────────────────────────── CLI ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_max", type=int, default=0, help="Limit eval examples (0=all).")
    ap.add_argument("--out_dir", default="gpt4o_eval", help="Output directory.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    ap.add_argument("--max_tokens", type=int, default=32, help="Max tokens for the answer.")
    ap.add_argument("--cache_dir", default=os.path.join(os.getcwd(), "hf_cache"), help="HF datasets cache dir.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    run_eval(args)

if __name__ == "__main__":
    main()
