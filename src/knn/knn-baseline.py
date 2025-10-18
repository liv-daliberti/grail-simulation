#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
knn_baseline.py
────────────────
Non-generative KNN baseline for next-video choice using the SAME gold-index
resolution and slate extraction logic as your GRPO/GPT-4o eval.

- Fits a per-slate-size KNN index on the TRAIN split (buckets: 1,2,3,4,5+).
- Saves/loads the index (npz files) for reproducible evaluation.
- Evaluates on EVAL split, predicting ONLY the numeric option index (1..N).
- Outputs per-example JSONL and metrics JSON mirroring your GPT-4o diagnostics.

Dependencies:
  pip install datasets numpy

Examples:
  # Train index (cap to 200k rows) and eval 200 examples
  python knn_baseline.py \
      --fit_index \
      --knn_k 25 --knn_metric cosine --knn_max_train 200000 \
      --eval_max 200 \
      --out_dir knn_eval \
      --overwrite

  # Load an existing index and eval full split
  python knn_baseline.py \
      --load_index knn_eval/index \
      --knn_k 25 --knn_metric l2 \
      --out_dir knn_eval_full \
      --overwrite
"""

from __future__ import annotations

import os, re, sys, json, time, argparse, logging, csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk

from prompt_builder import build_user_prompt, clean_text, synthesize_viewer_sentence

# ─────────────────────────── Data config (no YAML) ─────────────────────────────
DEFAULT_DATASET_SOURCE = "data/cleaned_grail"
TRAIN_SPLIT = "train"
EVAL_SPLIT = "validation"
PROMPT_COLUMN   = "state_text"                 # optional; becomes CONTEXT:
SOLUTION_COLUMN = "gold_id"                    # gold next-video id (matches GRPO prompt pipeline)

# ---------- title index (optional) ----------
# Add this near the top of the "Title index" section
DEFAULT_TITLE_DIRS = [
    "/n/fs/similarity/trees/data/results/capsule-5416997-data/recommendation trees/trees_gun",
    "/n/fs/similarity/trees/data/results/capsule-5416997-data/recommendation trees/trees_wage",
]

PROMPT_MAX_HISTORY = int(os.environ.get("KNN_PROMPT_MAX_HISTORY", os.environ.get("GRAIL_MAX_HISTORY", "12")))


def _load_dataset_source(source: str, cache_dir: str) -> DatasetDict:
    """Load a cleaned dataset from disk or from the Hub."""
    if os.path.isdir(source):
        return load_from_disk(source)
    ds = load_dataset(source, cache_dir=cache_dir)
    if isinstance(ds, DatasetDict):
        return ds
    raise ValueError(f"Dataset {source!r} did not return splits in a DatasetDict")


def _issues_in_dataset(ds: DatasetDict) -> List[str]:
    train_split = ds.get(TRAIN_SPLIT) or next(iter(ds.values()))
    if "issue" not in train_split.column_names:
        return ["all"]
    issues = sorted({str(x).strip() for x in train_split["issue"] if str(x).strip()})
    return issues or ["all"]


def _filter_dataset_for_issue(ds: DatasetDict, issue: str) -> DatasetDict:
    if issue == "all" or "issue" not in ds[TRAIN_SPLIT].column_names:
        return ds
    def _match_issue(row):
        value = row.get("issue")
        return str(value).strip() == issue

    filtered: Dict[str, Any] = {}
    for split_name, split_ds in ds.items():
        if "issue" not in split_ds.column_names:
            filtered[split_name] = split_ds
        else:
            filtered[split_name] = split_ds.filter(_match_issue)
    return DatasetDict(filtered)


def _parse_k_values(args) -> List[int]:
    values = {int(args.knn_k)} if getattr(args, "knn_k", None) is not None else set()
    sweep_raw = getattr(args, "knn_k_sweep", "")
    for token in sweep_raw.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError:
            continue
    k_vals = sorted(k for k in values if k > 0)
    if not k_vals:
        fallback = int(args.knn_k) if getattr(args, "knn_k", None) else 25
        k_vals = [fallback]
    return k_vals


def _select_best_k(k_values: List[int], accuracy_by_k: Dict[int, float]) -> int:
    if len(k_values) <= 2:
        return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))
    accuracies = [accuracy_by_k.get(k, 0.0) for k in k_values]
    slopes = []
    for i in range(1, len(k_values)):
        delta_acc = accuracies[i] - accuracies[i - 1]
        delta_k = k_values[i] - k_values[i - 1]
        slopes.append(delta_acc / delta_k if delta_k else 0.0)
    if not slopes:
        return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))
    first_slope = slopes[0]
    threshold = max(first_slope * 0.5, 0.001)
    for i, slope in enumerate(slopes[1:], start=1):
        if slope <= threshold:
            return k_values[i]
    return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))


def _resolve_reports_dir(out_dir: Path) -> Path:
    resolved = out_dir.resolve()
    parents = list(resolved.parents)
    if len(parents) >= 1 and parents[0].name == 'knn':
        resolved = parents[0]
        parents = list(resolved.parents)
    if len(parents) >= 1 and parents[0].name == 'models':
        root_dir = parents[0].parent
    elif len(parents) >= 2 and parents[1].name == 'models':
        root_dir = parents[1].parent
    else:
        root_dir = resolved.parent
    return root_dir / 'reports'


def _plot_elbow(k_values: List[int], accuracy_by_k: Dict[int, float], best_k: int, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("[KNN] Skipping elbow plot (%s)", exc)
        return

    plt.figure(figsize=(6, 4))
    ys = [accuracy_by_k.get(k, 0.0) for k in k_values]
    plt.plot(k_values, ys, marker='o', label='Accuracy')
    if best_k in accuracy_by_k:
        plt.scatter([best_k], [accuracy_by_k[best_k]], color='red', label=f'Best k={best_k}')
    plt.title('KNN accuracy vs k')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ───────────────────────────── Helpers / canon ─────────────────────────────────
ANS_TAG   = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
INDEX_ONLY= re.compile(r'^\s*(?:option\s*)?(\d+)\s*$', re.I)
YTID_RE   = re.compile(r'([A-Za-z0-9_-]{11})')
CANON_RE  = re.compile(r"[^a-z0-9]+")

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

# ───────────────────── Full profile extractors (EXACT-ish) ─────────────

def _truthy(x) -> bool:
    if x is None: return False
    if isinstance(x, (int, float)): return x != 0
    s = str(x).strip().lower()
    return s in {"1","true","t","yes","y","on"}

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


def _viewer_profile_sentence(ex: dict) -> str:
    sentence = clean_text(ex.get("viewer_profile_sentence"))
    if not sentence:
        sentence = clean_text(ex.get("viewer_profile"))
    if not sentence:
        try:
            sentence = synthesize_viewer_sentence(ex)
        except Exception:
            sentence = ""
    return sentence or ""


def _prompt_from_builder(ex: dict) -> str:
    try:
        return build_user_prompt(ex, max_hist=PROMPT_MAX_HISTORY)
    except Exception:
        return ""


def _humanize_profile(ex: dict) -> str:
    return _viewer_profile_sentence(ex)

# ───────────────────────────── Title index (optional) ──────────────────────────
def _split_env_list(s: str | None) -> list[str]:
    if not s: return []
    return [p for chunk in re.split(r'[:,\s]+', s) if (p:=chunk.strip())]

def _iter_csv_files_from_env() -> list[str]:
    files: list[str] = []
    # Env: explicit CSV files
    for p in _split_env_list(os.environ.get("GRAIL_TITLE_CSVS")):
        if os.path.isfile(p): files.append(p)

    # Env: directories (recursive)
    for d in _split_env_list(os.environ.get("GRAIL_TITLE_DIRS")):
        if os.path.isdir(d):
            for root, _, fnames in os.walk(d):
                for f in fnames:
                    if f.lower().endswith(".csv"):
                        files.append(os.path.join(root, f))

    # Env: glob(s)
    for pat in _split_env_list(os.environ.get("GRAIL_TITLE_GLOB")):
        try:
            from glob import glob
            for p in glob(pat): 
                if os.path.isfile(p):
                    files.append(p)
        except Exception:
            pass

    # Fallback to defaults if nothing from env
    if not files:
        for d in DEFAULT_TITLE_DIRS:
            if os.path.isdir(d):
                for root, _, fnames in os.walk(d):
                    for f in fnames:
                        if f.lower().endswith(".csv"):
                            files.append(os.path.join(root, f))

    # de-dup
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
    Falls back to trajectory_json['order'] exactly like GRPO, with the same title mapping.
    """
    items: List[Tuple[str,str]] = []

    # Prefer explicit slate_text
    st = ex.get("slate_text")
    if isinstance(st, str) and st.strip():
        for line in st.splitlines():
            s = line.strip()
            if not s or s == "-":
                continue
            # allow "- Foo" or "1. Foo" or "1) Foo"
            m = re.match(r"^\s*(?:-|\d+\s*[\.\)])\s*(.+)$", s)
            surface = m.group(1).strip() if m else s

            vid = _canon_vid(surface)
            if len(vid) == 11:
                # map 11-char ytid → human title if available
                title = _title_for(vid) or ""
                items.append((title, vid))
            else:
                # already a human-readable title
                items.append((surface, ""))

    # Fallback to trajectory_json['order']
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

    # de-dup preserve order
    seen, out = set(), []
    for t, vid in items:
        key = _canon(vid) or _canon(t)
        if key and key not in seen:
            seen.add(key); out.append((t, vid))
    return out

# ───────────────────── KNN featureization and model ─────────────────────

_IDEO_SCALE = {
    "extremely liberal": 1.0, "liberal": 0.85, "slightly liberal": 0.65,
    "moderate": 0.5, "slightly conservative": 0.35, "conservative": 0.2,
    "extremely conservative": 0.0
}
_PARTY_SCALE = {
    "democratic": 0.85, "democratic-leaning": 0.7,
    "independent": 0.5, "independent/other": 0.5,
    "republican-leaning": 0.3, "republican": 0.15
}
_RACES = ["White","Black","Asian","Hispanic/Latino","Native American","Pacific Islander","Middle Eastern","Multiracial","Other","Unspecified"]

def _income_to_float(s: Optional[str]) -> float:
    if not s: return 0.0
    nums = [int(x.replace(",","")) for x in re.findall(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*)", s)]
    if not nums: return 0.0
    val = (nums[0]+nums[1])/2.0 if len(nums)>=2 else float(nums[0])
    return max(0.0, min(val/200000.0, 1.0))

def _ideo_to_float(ex: dict) -> float:
    s = _format_ideology(ex.get("ideo"))
    if not s: return 0.5
    return _IDEO_SCALE.get(s.lower(), 0.5)

def _party_to_float(ex: dict) -> float:
    s = _extract_party(ex)
    if not s: return 0.5
    return _PARTY_SCALE.get(s.lower(), 0.5)

def _freq_to_float(ex: dict) -> float:
    v = str(ex.get("freq_youtube","")).strip()
    return (float(v)/5.0) if v.isdigit() else 0.0

def _onehot_race(ex: dict) -> List[float]:
    r = _extract_race(ex)
    vec = [0.0]*len(_RACES)
    if r:
        try:
            idx = _RACES.index(r)
            vec[idx] = 1.0
        except ValueError:
            vec[_RACES.index("Other")] = 1.0
    return vec

def _featurize(ex: dict) -> np.ndarray:
    age = ex.get("age")
    try: age = float(age)/100.0 if age is not None else 0.0
    except: age = 0.0
    female = 1.0 if _truthy(ex.get("female")) else 0.0
    male   = 1.0 if _truthy(ex.get("male"))   else 0.0
    income = _income_to_float(_extract_income(ex))
    ideo   = _ideo_to_float(ex)
    party  = _party_to_float(ex)
    freq   = _freq_to_float(ex)
    pos    = ex.get("video_index")
    try: pos = float(pos) if pos is not None else -1.0
    except: pos = -1.0
    pos_norm = max(0.0, min(pos/10.0, 1.0)) if pos >= 0 else 0.0
    race_oh = _onehot_race(ex)
    vec = [age, female, male, income, ideo, party, freq, pos_norm] + race_oh
    return np.asarray(vec, dtype=np.float32)

def _bucket_for_nopts(n: int) -> str:
    return "1" if n<=1 else ("2" if n==2 else ("3" if n==3 else ("4" if n==4 else "5+")))

def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0); sd[sd==0] = 1.0
    return (X - mu)/sd, mu, sd

def _standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu)/sd

def _cosine_dist(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    Ab = A @ b
    An = np.linalg.norm(A, axis=1) + 1e-8
    bn = np.linalg.norm(b) + 1e-8
    sim = Ab / (An*bn)
    return 1.0 - sim

def knn_predict(x: np.ndarray, nopts: int, knn_idx: Dict[str, Dict[str, Any]],
                k: int = 25, metric: str = "l2") -> Optional[int]:
    bucket = _bucket_for_nopts(nopts)
    pack = knn_idx.get(bucket)
    if not pack:
        return None
    X = pack["X"]; y = pack["y"]
    if X.shape[0] == 0:
        return None

    xz = _standardize_apply(x, pack["mu"], pack["sd"]).astype(np.float32)
    if metric == "cosine":
        d = _cosine_dist(X, xz)
    else:
        d = np.linalg.norm(X - xz, axis=1)

    k = min(k, X.shape[0])
    nn = np.argpartition(d, kth=k-1)[:k]
    w = 1.0 / (d[nn] + 1e-8)
    vote = defaultdict(float)
    for lab, wt in zip(y[nn], w):
        if 1 <= int(lab) <= nopts:
            vote[int(lab)] += float(wt)
    if not vote:
        counts = Counter([int(l) for l in y[nn] if 1 <= int(l) <= nopts])
        if not counts:
            return None
        return int(counts.most_common(1)[0][0])
    return max(vote.items(), key=lambda kv: kv[1])[0]

# ───────────────────────────── Index I/O ─────────────────────────────
def knn_predict_among_slate_multi(
    knn_index: dict,
    ex: dict,
    k_values: List[int],
    text_fields: list[str] | None = None,
    lowercase: bool = True,
) -> Dict[int, Optional[int]]:
    """
    Candidate-aware TF-IDF kNN over the *current slate* for multiple ``k`` values.
    Returns a mapping ``{k: predicted_option_index}`` (1-based). When no index is
    available the value is ``None``.
    """
    import numpy as np

    unique_k = sorted({int(k) for k in k_values if int(k) > 0})
    if not unique_k:
        return {}

    def _safe_str(x) -> str:
        try:
            s = "" if x is None else str(x)
        except Exception:
            s = ""
        s = s.strip()
        return s.lower() if lowercase else s

    slate_pairs = _extract_slate_items(ex)
    if not slate_pairs:
        return {k: 1 for k in unique_k}

    parts_base: list[str] = []
    prompt_text = _prompt_from_builder(ex)
    prompt_added = False
    if prompt_text:
        parts_base.append(_safe_str(prompt_text))
        prompt_added = True

    if (not prompt_added) and "_humanize_profile" in globals() and callable(globals()["_humanize_profile"]):
        try:
            vp = _humanize_profile(ex)
            if vp and vp.strip():
                parts_base.append(_safe_str(vp))
        except Exception:
            pass

    if (not prompt_added) and PROMPT_COLUMN in ex and ex[PROMPT_COLUMN] is not None:
        st = _safe_str(ex[PROMPT_COLUMN])
        if st:
            parts_base.append(st)

    for col in (text_fields or []):
        if col in ex and ex[col] is not None:
            val = _safe_str(ex[col])
            if val:
                parts_base.append(val)

    now = _extract_now_watching(ex)
    if now:
        now_title, now_id = now
        if now_title and now_title.strip():
            parts_base.append(_safe_str(now_title))
        if now_id and now_id.strip():
            parts_base.append(_safe_str(now_id))

    opt_surfaces = []
    for (t, vid) in slate_pairs:
        surf = (t if t and t.strip() and t != "(untitled)" else (_title_for(vid) or vid or ""))
        if surf and surf.strip():
            opt_surfaces.append(_safe_str(surf))
    if opt_surfaces:
        parts_base.append(" ".join(opt_surfaces))

    if not knn_index or ("vectorizer" not in knn_index):
        return {k: None for k in unique_k}

    vec = knn_index["vectorizer"]
    X = knn_index["X"]
    lab_id = knn_index["labels_id"]
    lab_title = knn_index["labels_title"]

    lab_id_canon = np.asarray([_canon_vid(x) for x in lab_id], dtype=object)
    lab_ti_canon = np.asarray([_canon(x or "") for x in lab_title], dtype=object)

    scores_by_k = {k: [] for k in unique_k}

    for (t, vid) in slate_pairs:
        surf = (t if t and t.strip() and t != "(untitled)" else (_title_for(vid) or vid or ""))
        parts = list(parts_base)
        if surf and surf.strip():
            parts.append(_safe_str(surf))

        q_text = "\n".join(p for p in parts if p)
        q = vec.transform([q_text])
        sims = (q @ X.T).toarray().ravel()

        vid_c = _canon_vid(vid or "")
        ti_c = _canon(t or "")
        mask = np.zeros_like(sims, dtype=bool)
        if vid_c:
            mask |= (lab_id_canon == vid_c)
        if ti_c:
            mask |= (lab_ti_canon == ti_c)

        if not mask.any():
            fallback = float(sims.max() * 0.01)
            for k in unique_k:
                scores_by_k[k].append(fallback)
            continue

        sims_m = sims[mask]
        if sims_m.size == 0:
            for k in unique_k:
                scores_by_k[k].append(0.0)
            continue

        sorted_sims = np.sort(sims_m)[::-1]
        cumsum = np.cumsum(sorted_sims)
        for k in unique_k:
            kk = int(min(max(1, k), sorted_sims.size))
            scores_by_k[k].append(float(cumsum[kk - 1]))

    predictions: Dict[int, Optional[int]] = {}
    for k, scores in scores_by_k.items():
        if not scores or not any(np.isfinite(scores)):
            predictions[k] = None
        else:
            predictions[k] = int(np.argmax(scores)) + 1
    return predictions

from scipy import sparse
import joblib

def save_tfidf_index(knn_idx: Dict[str, Any], out_dir: str) -> None:
    """
    Save TF-IDF index created by build_knn_index():
      - vectorizer (joblib)
      - X (CSR sparse, .npz)
      - labels_id, labels_title (.npy)
      - meta.json
    """
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    vec = knn_idx["vectorizer"]
    X   = knn_idx["X"]
    labels_id    = np.asarray(knn_idx["labels_id"], dtype=object)
    labels_title = np.asarray(knn_idx["labels_title"], dtype=object)

    joblib.dump(vec, d / "vectorizer.joblib")
    sparse.save_npz(d / "X.npz", X)
    np.save(d / "labels_id.npy", labels_id, allow_pickle=True)
    np.save(d / "labels_title.npy", labels_title, allow_pickle=True)

    meta = {"n_docs": int(X.shape[0]), "n_features": int(X.shape[1])}
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def load_tfidf_index(in_dir: str) -> Dict[str, Any]:
    """
    Load TF-IDF index saved by save_tfidf_index().
    """
    d = Path(in_dir)
    vec = joblib.load(d / "vectorizer.joblib")
    X   = sparse.load_npz(d / "X.npz")
    labels_id    = np.load(d / "labels_id.npy", allow_pickle=True).tolist()
    labels_title = np.load(d / "labels_title.npy", allow_pickle=True).tolist()
    return {"vectorizer": vec, "X": X, "labels_id": labels_id, "labels_title": labels_title}

# ───────────────────────────── Training KNN ─────────────────────────────
def build_knn_index(
    train_ds,
    max_train: int = 100_000,
    seed: int = 42,
    max_features: int | None = 200_000,
):
    """
    Create a TF-IDF kNN index from the TRAIN split.
    Returns: {"vectorizer", "X", "labels_id", "labels_title"}.
    """

    from sklearn.feature_extraction.text import TfidfVectorizer

    def _safe_str(x) -> str:
        try:
            s = "" if x is None else str(x)
        except Exception:
            s = ""
        s = s.strip()
        return s

    def _good(s: str) -> bool:
        if not s:
            return False
        sl = s.lower()
        return sl not in {"", "nan", "none", "(none)"}

    n = len(train_ds)
    if n == 0:
        raise RuntimeError("Train split is empty.")

    rng = np.random.default_rng(seed)
    if max_train and max_train > 0:
        take = min(max_train, n)
        order = rng.permutation(n)[:take].tolist()
    else:
        order = list(range(n))

    docs: list[str] = []
    labels_id: list[str] = []
    labels_title: list[str] = []

    # ------------- assemble documents -------------
    for i in order:
        ex = train_ds[int(i)]  # ensure plain int indexing

        parts: list[str] = []
        used_prompt = False

        prompt_text = _prompt_from_builder(ex)
        if _good(prompt_text):
            parts.append(prompt_text)
            used_prompt = True

        # humanized viewer profile (if available in this file)
        if not used_prompt:
            try:
                if "_humanize_profile" in globals() and callable(globals()["_humanize_profile"]):
                    vp = _humanize_profile(ex)
                    if _good(vp):
                        parts.append(vp)
            except Exception:
                pass

        # state/context text
        if (not used_prompt) and PROMPT_COLUMN in ex:
            st = _safe_str(ex[PROMPT_COLUMN])
            if _good(st):
                parts.append(st)

        # CURRENT item (title/id)
        try:
            now = _extract_now_watching(ex)
        except Exception:
            now = None
        if now:
            now_title, now_id = now
            if _good(now_title): parts.append(now_title)
            if _good(now_id):    parts.append(now_id)

        # SLATE items (titles or ids)
        try:
            slate_pairs = _extract_slate_items(ex)  # [(title, vid), ...]  ← now already title-resolved
        except Exception:
            slate_pairs = []
        if slate_pairs:
            opt_surfaces = []
            for t, vid in slate_pairs:
                # prefer human title; if blank, resolve from id; finally fall back to id
                surf = (t if _good(t) and t != "(untitled)"
                        else (_title_for(vid) or vid or ""))
                if _good(surf):
                    opt_surfaces.append(surf)
            if opt_surfaces:
                parts.append(" ".join(opt_surfaces))

        doc = " ".join(p for p in parts if _good(p)).strip()
        docs.append(doc)

        # label for neighbor → slate gating later
        vid = _safe_str(ex.get(SOLUTION_COLUMN, ""))
        labels_id.append(_canon_vid(vid))
        # map to title if possible (safe if title index is enabled)
        try:
            lab_title = _title_for(vid) or ""
        except Exception:
            lab_title = ""
        labels_title.append(lab_title)

    # ------------- filter empties -------------
    mask = [bool(d.strip()) for d in docs]
    kept = sum(mask)
    if kept == 0:
        # Helpful diagnostics: show which fields exist in train
        sample_cols = sorted(list(train_ds.features.keys()))
        raise RuntimeError(
            "All training documents are empty. Check columns on TRAIN split.\n"
            f"Seen columns: {sample_cols}\n"
            "Fixes: include slate items and CURRENTLY WATCHING in docs, avoid stop-word removal, "
            "or pass extra fields via --knn_text_fields."
        )

    if kept < len(docs):
        logging.warning("[KNN] Dropped %d empty docs out of %d.", len(docs) - kept, len(docs))
    docs         = [d  for d, m in zip(docs, mask) if m]
    labels_id    = [li for li, m in zip(labels_id, mask) if m]
    labels_title = [lt for lt, m in zip(labels_title, mask) if m]

    logging.info("[KNN] Assembled %d documents (kept %d non-empty).", len(order), kept)
    logging.info("[KNN] Example doc: %r", docs[0][:200])

    # ------------- vectorize -------------
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None,
        token_pattern=r"(?u)\b[\w\-]{2,}\b",  # keep hyphens
        max_features=max_features,
    )
    X = vectorizer.fit_transform(docs).astype(np.float32)

    logging.info("[KNN] TF-IDF: shape=%s, vocab=%d", X.shape, len(vectorizer.vocabulary_))

    return {
        "vectorizer": vectorizer,
        "X": X,  # CSR float32 [N,V]
        "labels_id": labels_id,
        "labels_title": labels_title,
    }

# ───────────────────────────── Eval loop ─────────────────────────────────

def _bucket_from_pos(pos_idx: int) -> str:
    if pos_idx < 0: return "unknown"
    if pos_idx == 0: return "1"
    if pos_idx == 1: return "2"
    if pos_idx == 2: return "3"
    if pos_idx == 3: return "4"
    return "5+"

def safe_div(n, d): return (n / d) if d else 0.0

def run_eval(args):
    """Evaluate the TF-IDF kNN baseline for each issue in the dataset."""
    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os.environ.setdefault("HF_HOME", args.cache_dir)

    dataset_source = args.dataset or DEFAULT_DATASET_SOURCE
    base_ds = _load_dataset_source(dataset_source, args.cache_dir)
    available_issues = _issues_in_dataset(base_ds)

    if args.issues:
        requested = [s.strip() for s in args.issues.split(",") if s.strip()]
        issues = requested if requested else available_issues
    else:
        issues = available_issues

    extra_fields = []
    if getattr(args, "knn_text_fields", ""):
        extra_fields = [s.strip() for s in args.knn_text_fields.split(",") if s.strip()]

    for issue in issues:
        ds_issue = _filter_dataset_for_issue(base_ds, issue)

        if TRAIN_SPLIT not in ds_issue:
            logging.warning("[KNN] train split missing for issue '%s'; skipping.", issue)
            continue
        train_ds = ds_issue[TRAIN_SPLIT]
        if len(train_ds) == 0:
            logging.warning("[KNN] train split empty for issue '%s'; skipping.", issue)
            continue

        if EVAL_SPLIT in ds_issue:
            eval_split_name = EVAL_SPLIT
        else:
            for alt in ("validation", "eval", "test"):
                if alt in ds_issue:
                    eval_split_name = alt
                    break
            else:
                logging.warning("[KNN] no eval split for issue '%s'; skipping.", issue)
                continue

        eval_ds = ds_issue[eval_split_name]
        if len(eval_ds) == 0:
            logging.warning("[KNN] evaluation split empty for issue '%s'; skipping.", issue)
            continue

        run_eval_for_issue(
            dataset_source=dataset_source,
            issue=issue,
            train_ds=train_ds,
            eval_ds=eval_ds,
            eval_split_name=eval_split_name,
            args=args,
            extra_fields=extra_fields,
        )


def run_eval_for_issue(
    *,
    dataset_source: str,
    issue: str,
    train_ds,
    eval_ds,
    eval_split_name: str,
    args,
    extra_fields: List[str],
) -> None:
    issue_slug = issue if issue != "all" else "all"
    logging.info(
        "[KNN] Evaluating issue=%s (train=%d, eval=%d)",
        issue_slug,
        len(train_ds),
        len(eval_ds),
    )

    knn_idx: Dict[str, Any] = {}
    if args.fit_index:
        knn_idx = build_knn_index(
            train_ds,
            max_train=args.knn_max_train,
            seed=args.knn_seed,
        )
        if args.save_index:
            save_dir = Path(args.save_index)
            if issue_slug:
                save_dir = save_dir / issue_slug
            save_tfidf_index(knn_idx, str(save_dir))

    if args.load_index:
        load_dir = Path(args.load_index)
        if issue_slug:
            load_dir = load_dir / issue_slug
        if load_dir.exists():
            knn_idx = load_tfidf_index(str(load_dir))
        else:
            logging.warning("[KNN] Index directory %s not found for issue '%s'.", load_dir, issue_slug)

    if not knn_idx:
        logging.warning("[KNN] No index available for issue '%s'; predictions will be None.", issue_slug)

    k_values = _parse_k_values(args)
    indices = list(range(len(eval_ds)))
    if args.eval_max and args.eval_max > 0:
        indices = indices[: min(args.eval_max, len(indices))]

    out_dir = Path(args.out_dir) / issue_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / f"knn_eval_{issue_slug}_{eval_split_name}.jsonl"
    metrics_json = out_dir / f"knn_eval_{issue_slug}_{eval_split_name}_metrics.json"
    if out_jsonl.exists() and not args.overwrite:
        print(f"[SKIP] Output exists (use --overwrite): {out_jsonl}")
        return

    buckets = ["1", "2", "3", "4", "5+", "unknown"]
    seen_b = {b: 0 for b in buckets}
    elig_b = {b: 0 for b in buckets}

    opts_buckets = ["1", "2", "3", "4", "5+"]
    seen_opts_b = {b: 0 for b in opts_buckets}
    elig_opts_b = {b: 0 for b in opts_buckets}

    seen_single = seen_multi = 0
    elig_single = elig_multi = 0

    gold_hist: Dict[int, int] = {}
    all_gold_indices: List[int] = []
    all_n_options: List[int] = []

    eligible_by_k = {k: 0 for k in k_values}
    correct_by_k = {k: 0 for k in k_values}

    rows: List[Dict[str, Any]] = []

    t0 = time.time()
    for idx in indices:
        ex = eval_ds[int(idx)]

        slate_pairs = _extract_slate_items(ex)
        nopts = len(slate_pairs)
        nbucket = _bin_nopts(nopts)
        pos_idx_raw = ex.get("video_index")
        try:
            pos_idx = int(pos_idx_raw) if pos_idx_raw is not None else -1
        except Exception:
            pos_idx = -1
        pbucket = _bucket_from_pos(pos_idx)
        seen_b[pbucket] += 1
        seen_opts_b[nbucket] += 1

        gold_idx = int(ex.get("gold_index") or -1)
        gold_raw = str(ex.get(SOLUTION_COLUMN, "")).strip()
        if gold_idx < 1 and slate_pairs:
            for i, (t, vid) in enumerate(slate_pairs, start=1):
                if gold_raw and (gold_raw == vid or _canon(gold_raw) == _canon(t)):
                    gold_idx = i
                    break

        predictions = knn_predict_among_slate_multi(
            knn_index=knn_idx,
            ex=ex,
            k_values=k_values,
            text_fields=extra_fields,
        )

        eligible = (gold_idx > 0 and nopts > 0)
        if eligible:
            elig_b[pbucket] += 1
            elig_opts_b[nbucket] += 1
            gold_hist[gold_idx] = gold_hist.get(gold_idx, 0) + 1
            all_gold_indices.append(gold_idx)
            all_n_options.append(nopts)
            if nopts == 1:
                elig_single += 1
            else:
                elig_multi += 1
        if nopts == 1:
            seen_single += 1
        elif nopts > 1:
            seen_multi += 1

        for k, pred in predictions.items():
            if eligible:
                eligible_by_k[k] += 1
                if pred is not None and int(pred) == int(gold_idx):
                    correct_by_k[k] += 1

        rows.append(
            {
                "predictions_by_k": predictions,
                "gold_index": int(gold_idx),
                "n_options": int(nopts),
                "n_options_bucket": nbucket,
                "eligible": bool(eligible),
                "position_index": int(pos_idx),
                "position_bucket": pbucket,
                "issue_value": ex.get("issue"),
            }
        )

        if len(rows) % 25 == 0:
            elapsed = time.time() - t0
            acc_progress = max(
                (safe_div(correct_by_k[k], eligible_by_k[k]) for k in k_values if eligible_by_k[k]),
                default=0.0,
            )
            print(f"[eval][{issue_slug}] {len(rows)}/{len(indices)}  interim-best-acc={acc_progress:.3f}  {elapsed:.1f}s")

    accuracy_by_k = {k: safe_div(correct_by_k[k], eligible_by_k[k]) for k in k_values}
    best_k = _select_best_k(k_values, accuracy_by_k)
    best_accuracy = accuracy_by_k.get(best_k, 0.0)
    eligible_overall = int(eligible_by_k.get(best_k, 0))
    correct_overall = int(correct_by_k.get(best_k, 0))

    corr_b = {b: 0 for b in buckets}
    corr_opts_b = {b: 0 for b in opts_buckets}
    corr_single = corr_multi = 0

    for row in rows:
        if not row["eligible"]:
            continue
        pred = row["predictions_by_k"].get(best_k)
        if pred is None:
            continue
        if int(pred) == row["gold_index"]:
            corr_b[row["position_bucket"]] += 1
            corr_opts_b[row["n_options_bucket"]] += 1
            if row["n_options"] == 1:
                corr_single += 1
            else:
                corr_multi += 1

    with open(out_jsonl, "w", encoding="utf-8") as w:
        for row in rows:
            preds_serializable = {str(k): (int(v) if v is not None else None) for k, v in row["predictions_by_k"].items()}
            best_pred = row["predictions_by_k"].get(best_k)
            out_row = {
                "knn_pred_index": int(best_pred) if best_pred is not None else None,
                "gold_index": row["gold_index"],
                "n_options": row["n_options"],
                "correct": bool(best_pred is not None and int(best_pred) == row["gold_index"]),
                "eligible": row["eligible"],
                "position_index": row["position_index"],
                "position_bucket": row["position_bucket"],
                "issue": issue_slug if issue_slug != "all" else row.get("issue_value"),
                "predictions_by_k": preds_serializable,
            }
            w.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    n_eval_final = len(rows)

    pos_stats = {
        b: {
            "n_seen": int(seen_b[b]),
            "n_eligible": int(elig_b[b]),
            "correct": int(corr_b[b]),
            "accuracy": safe_div(corr_b[b], elig_b[b]),
        }
        for b in buckets
    }
    by_n_options = {
        "hist_seen": {b: int(seen_opts_b[b]) for b in opts_buckets},
        "hist_eligible": {b: int(elig_opts_b[b]) for b in opts_buckets},
        "hist_correct": {b: int(corr_opts_b[b]) for b in opts_buckets},
        "accuracy": {b: safe_div(corr_opts_b[b], elig_opts_b[b]) for b in opts_buckets},
    }
    split_single_vs_multi = {
        "n_single": int(seen_single),
        "n_multi": int(seen_multi),
        "eligible_single": int(elig_single),
        "eligible_multi": int(elig_multi),
        "accuracy_single": safe_div(corr_single, elig_single),
        "accuracy_multi": safe_div(corr_multi, elig_multi),
    }

    gold_index_distribution = {str(k): int(v) for k, v in sorted(gold_hist.items())}
    if gold_hist:
        top_idx = max(gold_hist.items(), key=lambda kv: kv[1])[0]
        baseline_correct = sum(1 for gi in all_gold_indices if gi == top_idx)
        baseline_acc = safe_div(baseline_correct, eligible_overall)
    else:
        top_idx = None
        baseline_acc = 0.0

    random_baseline_expected_accuracy = float(np.mean([1.0 / n for n in all_n_options])) if all_n_options else 0.0

    accuracy_by_k_serializable = {str(k): float(accuracy_by_k[k]) for k in k_values}

    reports_dir = _resolve_reports_dir(Path(args.out_dir)) / "knn"
    reports_dir.mkdir(parents=True, exist_ok=True)
    elbow_path = reports_dir / f"elbow_{issue_slug}.png"
    _plot_elbow(k_values, accuracy_by_k, best_k, elbow_path)

    metrics = {
        "model": "knn",
        "dataset": dataset_source,
        "issue": issue_slug,
        "split": eval_split_name,
        "n_total": int(n_eval_final),
        "n_eligible": int(eligible_overall),
        "accuracy_overall": best_accuracy,
        "accuracy_by_k": accuracy_by_k_serializable,
        "best_k": int(best_k),
        "position_stats": pos_stats,
        "by_n_options": by_n_options,
        "split_single_vs_multi": split_single_vs_multi,
        "gold_index_distribution": gold_index_distribution,
        "baseline_most_frequent_gold_index": {
            "top_index": top_idx,
            "count": int(gold_hist.get(top_idx, 0) if top_idx is not None else 0),
            "accuracy": baseline_acc,
        },
        "random_baseline_expected_accuracy": random_baseline_expected_accuracy,
        "knn_hparams": {
            "k": int(args.knn_k),
            "k_sweep": [int(k) for k in k_values],
            "metric": args.knn_metric,
            "fit_index": bool(args.fit_index),
            "save_index": args.save_index or "",
            "load_index": args.load_index or "",
            "text_fields": extra_fields,
        },
        "elbow_plot": str(elbow_path),
        "notes": "Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.",
    }

    with open(metrics_json, "w", encoding="utf-8") as mj:
        json.dump(metrics, mj, ensure_ascii=False, indent=2)

    print(
        f"[DONE][{issue_slug}] split={eval_split_name}  n={n_eval_final}  eligible={eligible_overall}  "
        f"knn_acc={best_accuracy:.4f} (best_k={best_k})"
    )
    print(f"[WROTE] per-example: {out_jsonl}")
    print(f"[WROTE] metrics:     {metrics_json}")

# ─────────────────────────────────── CLI ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_index", action="store_true",
                    help="Build KNN index from train split before evaluation.")
    ap.add_argument("--save_index", default="", help="Directory to save KNN index (npz files).")
    ap.add_argument("--load_index", default="", help="Directory to load KNN index from (npz files).")
    ap.add_argument("--knn_k", type=int, default=25, help="K neighbors.")
    ap.add_argument("--knn_k_sweep", default="1,2,3,4,5,10,25,50",
                    help="Comma-separated list of alternative k values to evaluate in addition to --knn_k.")
    ap.add_argument("--knn_metric", default="l2", choices=["l2","cosine"], help="Distance metric.")
    ap.add_argument("--knn_max_train", type=int, default=200000, help="Cap training rows.")
    ap.add_argument("--knn_seed", type=int, default=42, help="Shuffle seed for train cap.")

    ap.add_argument("--eval_max", type=int, default=0, help="Limit eval examples (0=all).")
    ap.add_argument("--out_dir", default=str(Path("models") / "knn"), help="Output directory.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--cache_dir", default=os.path.join(os.getcwd(), "hf_cache"), help="HF datasets cache dir.")
    ap.add_argument("--knn_text_fields", default="", help="Comma-separated extra textual columns to include in TF-IDF query (e.g. 'extra1,extra2').")
    ap.add_argument("--dataset", default=DEFAULT_DATASET_SOURCE,
                    help="Cleaned dataset path (load_from_disk) or HF dataset id. Defaults to data/cleaned_grail.")
    ap.add_argument("--issues", default="",
                    help="Comma-separated issue names to evaluate (e.g. 'minimum_wage,gun_control'). Defaults to all issues found.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    run_eval(args)

if __name__ == "__main__":
    main()
