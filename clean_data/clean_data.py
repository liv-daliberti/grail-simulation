#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean & prepare GRAIL rows for GRPO (+optional GAIL).

Outputs (per row):
  - prompt:      chat-format messages [{role, content}, ...]
  - answer:      str gold index (1..N)
  - gold_index:  int (1..N)
  - gold_id:     str (next chosen id)
  - n_options:   int
  - viewer_profile: str (one-liner)
  - state_text:      str (LM-facing tidy state shown in prompt)
  - state_disc_text: str (disc-facing, richer state: full history + prior slates)
  - slate_items:     list[{"title","id"}]
  - slate_text:      enumerated names (optional; built if missing)
  - watched_detailed_json, watched_vids_json (passthrough if present)
  - current_video_id, current_video_title
  - task/is_replay/accuracy/mix_group_id/mix_copy_idx defaults

Only rows with:
  - non-empty slate_items_json
  - resolvable gold "next id" that exists in the slate
are kept.

CLI:
  python clean_grail.py \
    --dataset-name <hf-hub-id | /path/to/load_from_disk> \
    --train-split train --test-split validation \
    --output-dir /path/to/cleaned

Env (optional):
  GRAIL_MAX_HISTORY (default 12)
  GRAIL_SHOW_IDS=0/1 (affects LM state rendering only)
"""

from __future__ import annotations
import os, sys, re, json, logging, argparse, random
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
from datasets import DatasetDict
import pandas as pd

# ---------- logging ----------
log = logging.getLogger("clean_grail")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ---------- helpers ----------
ANS_RE   = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
IDX_ONLY = re.compile(r'^\s*(?:option\s*)?(\d+)\s*$', re.I)
YTID_RE  = re.compile(r'([A-Za-z0-9_-]{11})')

TOPIC_TO_ISSUE = {
    "min_wage": "minimum_wage",
    "gun_control": "gun_control",
}

LABEL_OPTIONS: Dict[str, List[Dict[str, str]]] = {
    "minimum_wage": [
        {"id": "min_wage_raise", "title": "WANTS to raise the minimum wage"},
        {"id": "min_wage_no_raise", "title": "Does NOT WANT to raise the minimum wage"},
        {"id": "min_wage_unknown", "title": "Not enough information"},
    ],
    "gun_control": [
        {"id": "gun_more_restrictions", "title": "WANTS MORE gun restrictions"},
        {"id": "gun_fewer_restrictions", "title": "WANTS FEWER gun restrictions"},
        {"id": "gun_unknown", "title": "Not enough information"},
    ],
}

LABEL_INDEX_TO_ID: Dict[str, Dict[str, str]] = {
    "minimum_wage": {"1": "min_wage_raise", "2": "min_wage_no_raise", "3": "min_wage_unknown"},
    "gun_control": {"1": "gun_more_restrictions", "2": "gun_fewer_restrictions", "3": "gun_unknown"},
}

def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower().strip())

def _canon_vid(v: str) -> str:
    if not isinstance(v, str): return ""
    m = YTID_RE.search(v)
    return m.group(1) if m else v.strip()

def _is_nanlike(x: Any) -> bool:
    if x is None: return True
    s = str(x).strip().lower()
    return s in {"", "nan", "none", "null", "n/a"}

def _as_list_json(x: Any, default="[]") -> list:
    if isinstance(x, list): return x
    if isinstance(x, str):
        try:
            v = json.loads(x or default)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    # pyarrow List?
    try:
        import pyarrow as pa  # type: ignore
        if isinstance(x, pa.Array):
            return x.to_pylist()
    except Exception:
        pass
    return []


def _strip_session_video_id(vid: str) -> str:
    if not isinstance(vid, str):
        return ""
    vid = vid.strip()
    if not vid:
        return ""
    if len(vid) <= 11:
        return vid
    base = vid[:11]
    if YTID_RE.fullmatch(base):
        return base
    m = YTID_RE.search(vid)
    return m.group(1) if m else vid


def _resolve_capsule_data_root(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    if (path / "platform session data" / "sessions.json").exists():
        return path
    if (path / "data" / "platform session data" / "sessions.json").exists():
        return path / "data"
    return None


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning("Survey file missing: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str)
    except Exception as exc:
        log.error("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def _build_survey_index(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    if df.empty:
        return index
    cols = list(df.columns)
    if "urlid" not in cols:
        log.warning("Survey frame missing urlid column; columns=%s", cols)
        return index
    for _, row in df.iterrows():
        urlid = str(row.get("urlid") or "").strip()
        if not urlid:
            continue
        cleaned = {}
        for k, v in row.items():
            if pd.isna(v):
                cleaned[k] = None
            else:
                cleaned[k] = v
        index.setdefault(urlid, []).append(cleaned)
    return index


def _select_survey_row(rows: List[Dict[str, Any]], topic_id: str) -> Dict[str, Any]:
    if not rows:
        return {}
    topic_id = (topic_id or "").strip()
    if topic_id:
        for row in rows:
            r_topic = str(row.get("topic_id") or row.get("topicID") or "").strip()
            if r_topic and r_topic == topic_id:
                return row
    return rows[0]


def _load_video_metadata(base_dir: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    meta_dir = base_dir / "supplemental" / "metadata and ratings"
    if not meta_dir.exists():
        return meta
    for csv_path in meta_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        id_col = None
        for cand in ("originID", "originId", "video_id", "videoId", "id"):
            if cand in df.columns:
                id_col = cand
                break
        if not id_col:
            continue
        title_col = None
        for cand in ("title", "video_title", "name"):
            if cand in df.columns:
                title_col = cand
                break
        for _, row in df.iterrows():
            vid = row.get(id_col)
            if pd.isna(vid):
                continue
            vid_str = str(vid).strip()
            if not vid_str:
                continue
            base = _strip_session_video_id(vid_str)
            if title_col and not pd.isna(row.get(title_col)):
                title = str(row.get(title_col))
            else:
                title = ""
            meta.setdefault(base, title)
    return meta


def _build_codeocean_rows(data_root: Path) -> pd.DataFrame:
    sessions_path = data_root / "platform session data" / "sessions.json"
    with open(sessions_path, "r", encoding="utf-8") as fp:
        sessions = json.load(fp)

    capsule_root = data_root.parent
    survey_gun = _read_csv_if_exists(
        capsule_root / "intermediate data" / "gun control (issue 1)" / "guncontrol_qualtrics_w123_clean.csv"
    )
    survey_wage = _read_csv_if_exists(
        capsule_root / "intermediate data" / "minimum wage (issue 2)" / "qualtrics_w12_clean.csv"
    )

    surveys = {
        "gun_control": _build_survey_index(survey_gun),
        "minimum_wage": _build_survey_index(survey_wage),
    }

    video_meta = _load_video_metadata(data_root)

    rows: List[Dict[str, Any]] = []
    for sess in sessions:
        topic = sess.get("topicID")
        issue = TOPIC_TO_ISSUE.get(topic)
        if not issue:
            continue
        ratings = sess.get("ratingResults") or []
        if not ratings:
            continue
        urlid = str(sess.get("urlid") or "").strip()
        survey_row = _select_survey_row(surveys.get(issue, {}).get(urlid, []), topic or "")
        option_template = LABEL_OPTIONS[issue]
        index_map = LABEL_INDEX_TO_ID[issue]
        for rating in ratings:
            idx = str(rating.get("index") or "").strip()
            gold_id = index_map.get(idx)
            if not gold_id:
                continue
            vid = str(rating.get("vid") or "").strip()
            base_vid = _strip_session_video_id(vid)
            title = video_meta.get(base_vid) or video_meta.get(vid) or ""
            row: Dict[str, Any] = {
                "issue": issue,
                "urlid": urlid,
                "topic_id": topic,
                "current_video_id": base_vid or vid,
                "current_video_title": title or str(rating.get("copy") or ""),
                "slate_items_json": [dict(opt) for opt in option_template],
                "n_options": len(option_template),
                "next_video_id": gold_id,
                "rating_copy": rating.get("copy"),
                "rating_index": idx,
                "rating_video_id": base_vid or vid,
                "watched_detailed_json": [{"id": base_vid or vid, "title": title}],
                "watched_vids_json": [base_vid or vid],
            }
            if survey_row:
                for k, v in survey_row.items():
                    if k not in row and v is not None:
                        row[k] = v
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def _split_dataframe(df: pd.DataFrame, validation_ratio: float = 0.1) -> Dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    if not 0 < validation_ratio < 1:
        validation_ratio = 0.1
    indices = list(range(len(df)))
    random.Random(2024).shuffle(indices)
    val_size = max(1, int(len(indices) * validation_ratio)) if len(indices) > 1 else 0
    val_idx = set(indices[:val_size]) if val_size else set()
    splits = {
        "train": df.iloc[[i for i in indices if i not in val_idx]].reset_index(drop=True),
    }
    if val_idx:
        splits["validation"] = df.iloc[[i for i in indices if i in val_idx]].reset_index(drop=True)
    return splits

def _load_slate_items(ex: dict) -> List[dict]:
    arr = _as_list_json(ex.get("slate_items_json"))
    out: List[dict] = []
    for it in arr:
        if not isinstance(it, dict): continue
        t = (it.get("title") or "").strip()
        v = (it.get("id") or "").strip()
        if t or v: out.append({"title": t, "id": v})
    return out

def _secs(x: Any) -> str:
    try: return f"{int(round(float(x)))}s"
    except Exception: return "?"

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
    # party/ideo
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
    # education
    college = str(ex.get("college") or "").strip().lower()
    if college in {"true","1","yes","y"}: bits.append("college-educated")
    # youtube frequency
    fy = str(ex.get("freq_youtube") or "").strip()
    fmap = {"0":"rarely","1":"occasionally","2":"a few times a month","3":"weekly","4":"several times a week","5":"daily"}
    if fy in fmap:
        bits.append(f"watches YouTube {fmap[fy]}")
    s = ", ".join(b for b in bits if b)
    return s if s else "(no profile provided)"

def _build_user_prompt_from_columns(ex: dict, max_hist: int = 12) -> str:
    show_ids = os.getenv("GRAIL_SHOW_IDS", "0") == "1"
    lines: List[str] = []

    # PROFILE
    viewer = (ex.get("viewer_profile_sentence") or "").strip()
    if not viewer:
        viewer = _synthesize_viewer_sentence(ex)
    lines.append("PROFILE:")
    lines.append(viewer)

    # ATTRIBUTES (brief)
    def _clean(s: Any) -> str: return str(s).strip() if s is not None else ""
    def _truthy_str(s: Any) -> Optional[bool]:
        if s is None: return None
        v = str(s).strip().lower()
        if v in {"1","true","t","yes","y"}:  return True
        if v in {"0","false","f","no","n"}:  return False
        return None

    details: List[str] = []
    race = _clean(ex.get("race") or ex.get("ethnicity") or ex.get("q29"))
    if race and not _is_nanlike(race):
        details.append(f"race/ethnicity: {race}")

    gun_own = _truthy_str(ex.get("gun_own"))
    if gun_own is True:  details.append("owns a gun")
    elif gun_own is False: details.append("does not own a gun")

    fy = _clean(ex.get("freq_youtube"))
    fmap = {"0":"rarely","1":"occasionally","2":"a few times a month","3":"weekly","4":"several times a week","5":"daily"}
    if fy in fmap: details.append(f"YouTube frequency: {fmap[fy]}")

    fav = _clean(ex.get("q8") or ex.get("fav_channels"))
    if fav and not _is_nanlike(fav): details.append(f"favorite channels: {fav}")
    pop = _clean(ex.get("q78"))
    if pop and not _is_nanlike(pop): details.append(f"popular channels followed: {pop}")

    if details:
        lines.append("\nATTRIBUTES:")
        for d in details: lines.append(f"- {d}")

    # CURRENTLY WATCHING
    cvt  = (ex.get("current_video_title") or "").strip()
    cvid = (ex.get("current_video_id") or "").strip()
    if cvt or cvid:
        lines.append("\nCURRENTLY WATCHING:")
        if show_ids and cvid:
            lines.append(f"{cvt or '(untitled)'} — id: {cvid}")
        else:
            lines.append(f"{cvt or '(untitled)'}")

    # HISTORY (prior only, most recent first)
    det  = _as_list_json(ex.get("watched_detailed_json"))
    vids = _as_list_json(ex.get("watched_vids_json"))

    def _last_index(xs, val):
        if not isinstance(xs, list) or val is None: return None
        idx = None
        for i, v in enumerate(xs):
            if v == val: idx = i
        return idx

    cur_idx = None
    if cvid:
        cur_idx = _last_index(vids, cvid)
        if cur_idx is None and isinstance(det, list):
            for j in range(len(det) - 1, -1, -1):
                try:
                    if isinstance(det[j], dict) and (det[j].get("id") or "").strip() == cvid:
                        cur_idx = j; break
                except Exception:
                    pass
    if cur_idx is None and isinstance(vids, list) and vids:
        cur_idx = len(vids) - 1

    prior = []
    if isinstance(det, list) and cur_idx is not None and cur_idx > 0:
        prior = det[:cur_idx]

    if prior:
        lines.append("\nHISTORY (most recent first):")
        recent = list(reversed(prior))[:max_hist if max_hist and max_hist > 0 else len(prior)]
        for r in recent:
            name = (r.get("title") or (r.get("id") if show_ids else "") or "(untitled)").strip()
            ws = _secs(r.get("watch_seconds")); tl = _secs(r.get("total_length"))
            lines.append(f"- [{ws}/{tl}] {name}")

    # OPTIONS
    items = _load_slate_items(ex)
    lines.append("\nOPTIONS:")
    if items:
        for i, it in enumerate(items, 1):
            nm = (it.get("title") or (it.get("id") if show_ids else "") or "(untitled)").strip()
            lines.append(f"{i}. {nm}")
    else:
        lines.append("(no options provided)")

    return "\n".join(lines)

# ---------- “full history” & “prior slates” for discriminator ----------
def _render_full_history_lines_disc(ex: dict, include_current: bool = False) -> list[str]:
    tj = ex.get("trajectory_json")
    try:
        obj = json.loads(tj) if isinstance(tj, str) and tj.strip() else {}
    except Exception:
        obj = {}
    order = obj.get("order") if isinstance(obj, dict) else None
    if not isinstance(order, list): return []
    # sort by idx or end_ms
    def _key(r):
        try: return (0, int(r.get("idx")))
        except Exception:
            try: return (1, float(r.get("end_ms") or -1))
            except Exception: return (1, -1.0)
    seq = [r for r in order if isinstance(r, dict)]
    seq.sort(key=_key)

    cur_id = (ex.get("current_video_id") or "").strip()
    lines = []
    for r in seq:
        vid = (r.get("video_id") or r.get("id") or "")
        tit = (r.get("title") or r.get("video_title") or "")
        if not include_current and cur_id and vid == cur_id:
            break
        lines.append(f"- {tit or vid or '(untitled)'}")
    return lines

def _render_prior_slates(ex: dict) -> list[str]:
    tj = ex.get("trajectory_json")
    try:
        obj = json.loads(tj) if isinstance(tj, str) and tj.strip() else {}
    except Exception:
        obj = {}
    disp = obj.get("displayOrders") if isinstance(obj, dict) else None
    if not isinstance(disp, dict): return []
    out = []
    keys = sorted(
        [k for k in disp.keys() if re.match(r"^\s*(\d+)\s*[-_ ]*recs\s*$", str(k), re.I)],
        key=lambda k: int(re.search(r"(\d+)", str(k)).group(1))
    )
    for k in keys:
        val = disp.get(k) or []
        names = []
        if isinstance(val, list):
            for el in val:
                if isinstance(el, dict):
                    names.append(el.get("title") or el.get("id") or "(untitled)")
                else:
                    names.append(str(el))
        elif isinstance(val, dict):
            names = [str(x) for x in val.keys()]
        out.append(f"{k}: " + "; ".join(names[:10]))
    return out

def _build_state_disc_text(ex: dict) -> str:
    parts: List[str] = []
    now_line = (ex.get("current_video_title") or ex.get("current_video_id") or "(none)")
    parts += ["CURRENT:", now_line]
    hist_full = _render_full_history_lines_disc(ex, include_current=False)
    if hist_full:
        parts += ["", "HISTORY (full):", *hist_full]
    prior = _render_prior_slates(ex)
    if prior:
        parts += ["", "PRIOR_SLATES:", *prior]
    return "\n".join(parts)

# ---------- gold next id ----------
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
    cur = (ex.get("current_video_id") or "").strip()
    if sol_key and sol_key not in {"current_video_id", "current_id"}:
        v = ex.get(sol_key)
        if isinstance(v, str) and v.strip() and v.strip() != cur:
            return v.strip()
    for k in ("next_video_id", "clicked_id", "label", "answer"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip() and v.strip() != cur:
            return v.strip()
    return _derive_next_from_history(ex, cur)

def _gold_index_from_items(gold: str, items: List[dict]) -> int:
    gold = (gold or "").strip()
    if not gold or not items: return -1
    for i, it in enumerate(items, 1):
        if gold == (it.get("id") or ""): return i
    gc = _canon(gold)
    if gc:
        for i, it in enumerate(items, 1):
            if gc == _canon(it.get("title", "")): return i
    return -1

# ---------- row → clean example ----------
def _row_to_example(ex: dict, sys_prompt: Optional[str], sol_key: Optional[str], max_hist: int) -> Optional[dict]:
    items = _load_slate_items(ex)
    if not items: return None
    gold_id = _get_gold_next_id(ex, sol_key)
    gidx    = _gold_index_from_items(gold_id, items)
    if gidx < 1: return None

    user_msg = _build_user_prompt_from_columns(ex, max_hist=max_hist)
    sys_msg  = sys_prompt or (
        "You are choosing EXACTLY ONE item from a short slate for a specific viewer.\n"
        "Think briefly in <think>…</think>, then output ONLY the option NUMBER (1..N) inside <answer>…</answer>.\n"
        "Format (STRICT): <think>…</think><answer>3</answer>"
    )

    # enumerated slate text if not present
    slate_names = []
    for i, it in enumerate(items, 1):
        nm = (it.get("title") or it.get("id") or "(untitled)").strip()
        slate_names.append(f"{i}. {nm}")
    slate_text = "\n".join(slate_names)

    out = {
        "prompt": [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg},
        ],
        "answer": str(gidx),                 # GOLD index as string
        "gold_index": gidx,                  # int
        "gold_id": gold_id,
        "n_options": int(ex.get("n_options") or len(items) or 0),
        "viewer_profile": str(ex.get("viewer_profile_sentence") or _synthesize_viewer_sentence(ex)),
        "state_text": user_msg,              # LM sees this
        "state_disc_text": _build_state_disc_text(ex),  # disc sees richer info
        "slate_items": items,
        "slate_text":  str(ex.get("slate_text") or slate_text),
        # passthrough
        "watched_detailed_json": _as_list_json(ex.get("watched_detailed_json")),
        "watched_vids_json":     _as_list_json(ex.get("watched_vids_json")),
        "current_video_id":      str(ex.get("current_video_id") or ""),
        "current_video_title":   str(ex.get("current_video_title") or ""),
        "task": "GRAIL",
        "is_replay": False, "accuracy": 0.0, "mix_group_id": -1, "mix_copy_idx": -1,
    }
    for extra in ("issue", "rating_copy", "rating_index", "rating_video_id", "urlid", "topic_id"):
        if extra in ex:
            out[extra] = ex.get(extra)
    return out

# ---------- driver ----------
def _load_codeocean_dataset(dataset_name: str, validation_ratio: float = 0.1) -> DatasetDict:
    root = Path(dataset_name).expanduser()
    data_root = _resolve_capsule_data_root(root)
    if not data_root:
        raise ValueError(f"CodeOcean capsule data not found under {dataset_name}")
    log.info("Building dataset from CodeOcean capsule at %s", data_root)
    df = _build_codeocean_rows(data_root)
    if df.empty:
        raise ValueError("No usable rows found in CodeOcean sessions")
    split_frames = _split_dataframe(df, validation_ratio=validation_ratio)
    ds = {
        name: datasets.Dataset.from_pandas(frame, preserve_index=False)
        for name, frame in split_frames.items()
        if not frame.empty
    }
    log.info("CodeOcean rows: %s", {name: len(frame) for name, frame in split_frames.items()})
    return DatasetDict(ds)


def load_raw(dataset_name: str, validation_ratio: float = 0.1) -> DatasetDict:
    """Load from a HF hub id, a load_from_disk folder, or a single-split file."""
    if os.path.isdir(dataset_name):
        resolved = _resolve_capsule_data_root(Path(dataset_name))
        if resolved is not None:
            try:
                return _load_codeocean_dataset(str(resolved), validation_ratio=validation_ratio)
            except Exception as exc:
                log.error("Failed to build CodeOcean dataset: %s", exc)
                raise
        log.info("Loading dataset from disk: %s", dataset_name)
        ds = datasets.load_from_disk(dataset_name)
        if isinstance(ds, DatasetDict): return ds
        return DatasetDict({"train": ds})
    if os.path.isfile(dataset_name):
        ext = os.path.splitext(dataset_name)[1].lower()
        if ext in {".jsonl", ".json"}:
            ds = datasets.load_dataset("json", data_files=dataset_name)
        elif ext in {".csv", ".tsv"}:
            ds = datasets.load_dataset("csv", data_files=dataset_name, delimiter="," if ext==".csv" else "\t")
        else:
            raise ValueError(f"Unsupported file: {dataset_name}")
        return DatasetDict({k: v for k, v in ds.items()})
    # HF hub id
    log.info("Loading dataset from hub: %s", dataset_name)
    return datasets.load_dataset(dataset_name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", required=True, help="HF hub id, load_from_disk dir, or file")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split",  default="validation")
    ap.add_argument("--output-dir",  required=True)
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--max-history", type=int, default=int(os.environ.get("GRAIL_MAX_HISTORY", "12")))
    ap.add_argument("--validation-ratio", type=float, default=0.1, help="Validation share for CodeOcean data")
    ap.add_argument(
        "--issue-repo",
        action="append",
        default=[],
        help="Optional issue=repo mapping for pushing cleaned splits to the Hugging Face hub.",
    )
    ap.add_argument("--push-to-hub", action="store_true", help="Push cleaned datasets to the hub")
    ap.add_argument("--hub-token", default=None, help="Token for authenticated Hugging Face pushes")
    args = ap.parse_args()

    raw = load_raw(args.dataset_name, validation_ratio=args.validation_ratio)
    log.info("Splits available: %s", list(raw.keys()))

    sol_key = None  # can be wired to your config if you keep a named solution column

    # Filter to usable rows (slate present + gold-in-slate resolvable)
    def _ok(ex):
        items = _load_slate_items(ex)
        if not items: return False
        gold = _get_gold_next_id(ex, sol_key)
        if not gold: return False
        return _gold_index_from_items(gold, items) >= 1

    raw = raw.filter(_ok)
    log.info("Counts after filter: %s", {k: len(v) for k, v in raw.items()})

    # Map to cleaned schema
    mapped = raw.map(
        lambda ex: _row_to_example(ex, args.system_prompt, sol_key, max_hist=args.max_history),
        load_from_cache_file=False
    )

    # Drop any None rows (defensive)
    for split in list(mapped.keys()):
        mapped[split] = mapped[split].filter(lambda x: x is not None)

    # Keep only the columns trainer/rewards need (+ a few meta)
    keep_cols = {
        "prompt","answer","gold_index","gold_id","n_options",
        "viewer_profile","state_text","state_disc_text","slate_text","slate_items",
        "watched_detailed_json","watched_vids_json",
        "current_video_id","current_video_title",
        "task","is_replay","accuracy","mix_group_id","mix_copy_idx",
        "issue","rating_copy","rating_index","rating_video_id","urlid","topic_id"
    }
    for split in list(mapped.keys()):
        drop = [c for c in mapped[split].column_names if c not in keep_cols]
        if drop:
            mapped[split] = mapped[split].remove_columns(drop)

    # If only train exists, keep it; else preserve chosen splits
    desired = {}
    if args.train_split in mapped:
        desired["train"] = mapped[args.train_split]
    else:
        # pick any one split as train
        k0 = list(mapped.keys())[0]
        desired["train"] = mapped[k0]
    if args.test_split in mapped:
        desired["validation"] = mapped[args.test_split]

    final = DatasetDict(desired)
    os.makedirs(args.output_dir, exist_ok=True)
    log.info("Saving cleaned dataset to %s", args.output_dir)
    final.save_to_disk(args.output_dir)
    log.info("Done. Rows: %s", {k: len(v) for k, v in final.items()})

    # Optional per-issue exports / pushes
    issue_repo_map: Dict[str, str] = {}
    for spec in args.issue_repo:
        if "=" not in spec:
            raise ValueError(f"Invalid --issue-repo format: {spec!r}; expected issue=repo")
        issue, repo = spec.split("=", 1)
        issue_repo_map[issue.strip()] = repo.strip()

    if issue_repo_map or args.push_to_hub:
        has_issue = all("issue" in split.column_names for split in final.values())
        if not has_issue:
            log.warning("Issue-level exports requested, but 'issue' column missing in dataset")
        else:
            issues_in_data: set[str] = set(issue_repo_map.keys())
            for split in final.values():
                if "issue" in split.column_names:
                    issues_in_data.update(split.unique("issue"))
            for issue_name in sorted(issues_in_data):
                if not issue_name:
                    continue
                issue_ds = DatasetDict()
                for split_name, split_ds in final.items():
                    subset = split_ds.filter(lambda row, name=issue_name: row.get("issue") == name)
                    if len(subset):
                        issue_ds[split_name] = subset
                if not issue_ds:
                    log.warning("No rows for issue %s; skipping", issue_name)
                    continue
                issue_dir = os.path.join(args.output_dir, issue_name)
                os.makedirs(issue_dir, exist_ok=True)
                log.info("Saving issue '%s' dataset to %s", issue_name, issue_dir)
                issue_ds.save_to_disk(issue_dir)
                repo_id = issue_repo_map.get(issue_name)
                if args.push_to_hub and repo_id:
                    log.info("Pushing issue '%s' dataset to %s", issue_name, repo_id)
                    issue_ds.push_to_hub(repo_id, token=args.hub_token)

if __name__ == "__main__":
    main()
