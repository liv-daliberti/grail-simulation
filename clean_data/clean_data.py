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
import os, sys, re, json, logging, argparse
from typing import Any, List, Optional, Tuple

import datasets
from datasets import DatasetDict

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
    return out

# ---------- driver ----------
def load_raw(dataset_name: str) -> DatasetDict:
    """Load from a HF hub id, a load_from_disk folder, or a single-split file."""
    if os.path.isdir(dataset_name):
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
    args = ap.parse_args()

    raw = load_raw(args.dataset_name)
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
        "task","is_replay","accuracy","mix_group_id","mix_copy_idx"
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

if __name__ == "__main__":
    main()
