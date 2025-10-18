#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean & prepare GRAIL rows for ``src/open_r1/grpo.py`` (plus optional GAIL).

Outputs (per row):
  - prompt:      chat-format messages ``[{role, content}, ...]``
  - answer:      ``str`` gold index (``1..N``)
  - gold_index:  ``int`` (``1..N``)
  - gold_id:     ``str`` (next chosen id)
  - n_options:   ``int``
  - viewer_profile: ``str`` one-liner
  - state_text:      ``str`` LM-facing tidy state shown in prompt
  - state_disc_text: ``str`` discriminator-facing state (full history + prior slates)
  - slate_items:     ``list[{"title","id"}]``
  - slate_text:      enumerated names (built if missing)
  - watched_detailed_json / watched_vids_json passthroughs
  - current_video_id, current_video_title
  - task / is_replay / accuracy / mix_group_id / mix_copy_idx defaults

Only rows with:
  - non-empty ``slate_items_json``
  - resolvable gold "next id" that exists in the slate
are kept. Both GRAIL issues (``gun_control`` and ``minimum_wage``) are preserved
when present in the source data.

Linux CLI quickstart::

    # 1) Create the Python environment
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    # 2) Fetch the CodeOcean capsule data (see README for full commands)
    git clone https://git.codeocean.com/capsule-5416997.git
    # ...download the two zip payloads, then unzip...

    # 3) Build the cleaned HF dataset for GRPO
    python clean_data/clean_data.py \
        --dataset-name capsule-5416997/data \
        --output-dir data/cleaned_grail

    # 4) (Optional) push per-issue subsets to the Hub
    python clean_data/clean_data.py \
        --dataset-name capsule-5416997/data \
        --output-dir data/cleaned_grail \
        --issue-repo gun_control=my-org/grail-gun --issue-repo minimum_wage=my-org/grail-wage \
        --push-to-hub --hub-token $HF_TOKEN

Env (optional):
  GRAIL_MAX_HISTORY (default 12)
  GRAIL_SHOW_IDS=0/1 (affects LM state rendering only)
"""

from __future__ import annotations
import os, sys, re, json, logging, argparse, random, csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

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

REQUIRED_FOR_GRPO = {
    "prompt",
    "answer",
    "gold_index",
    "gold_id",
    "n_options",
    "viewer_profile",
    "state_text",
    "slate_items",
    "slate_text",
    "watched_detailed_json",
    "watched_vids_json",
    "current_video_id",
    "current_video_title",
    "task",
    "is_replay",
    "accuracy",
    "mix_group_id",
    "mix_copy_idx",
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
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str)
    except Exception as exc:
        log.error("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def _read_survey_with_fallback(*candidates: Path) -> pd.DataFrame:
    """Try multiple survey CSV locations, skipping Git-LFS pointers that lack ``urlid``."""
    fallback_df: Optional[pd.DataFrame] = None
    for path in candidates:
        if not path.exists():
            continue
        df = _read_csv_if_exists(path)
        if df is None:
            continue
        columns = [str(c).strip().lower() for c in getattr(df, "columns", [])]
        if "urlid" in columns:
            return df
        if df.empty and fallback_df is None:
            fallback_df = df
        else:
            log.warning("Survey frame missing urlid column; columns=%s", list(df.columns))
    if fallback_df is not None:
        return fallback_df
    if candidates:
        log.warning("Survey file missing: %s", candidates[0])
    return pd.DataFrame()


def _normalize_urlid(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        num = float(text)
        if math.isfinite(num):
            if num.is_integer():
                return str(int(num))
            return text
    except Exception:
        pass
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _build_survey_index(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    if df.empty:
        return index
    cols = list(df.columns)
    if "urlid" not in cols:
        log.warning("Survey frame missing urlid column; columns=%s", cols)
        return index
    for _, row in df.iterrows():
        urlid = _normalize_urlid(row.get("urlid"))
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


TEXT_SUFFIXES = {
    "title": ["Title", "title", "Name"],
    "channel_title": ["Channel", "ChannelTitle", "channel"],
    "channel_id": ["ChannelID", "ChannelId", "channel_id"],
    "description": ["Description", "description"],
    "category": ["Cat", "Category", "category"],
    "caption": ["Caption", "caption"],
}

NUMERIC_SUFFIXES = {
    "view_count": ["ViewCount", "viewCount", "Views", "views"],
    "like_count": ["LikeCount", "likeCount", "Likes", "likes"],
    "dislike_count": ["DislikeCount", "dislikeCount", "Dislikes", "dislikes"],
    "favorite_count": ["FavoriteCount", "favoriteCount", "Favorites", "favorites"],
    "comment_count": ["CommentCount", "commentCount", "Comments", "comments"],
    "duration": ["Duration", "duration", "DurationSeconds", "duration_seconds"],
}

DEMOGRAPHIC_COLUMNS = [
    "age",
    "gender",
    "q26",
    "q29",
    "race",
    "ethnicity",
    "q31",
    "income",
    "household_income",
    "pid1",
    "ideo1",
    "freq_youtube",
    "college",
]


def _assign_if_missing(info: Dict[str, Any], key: str, value: Any) -> None:
    """Populate ``info[key]`` with ``value`` when the destination field is empty."""
    if value is None:
        return
    if isinstance(value, str):
        sval = value.strip()
        if not sval or _is_nanlike(sval):
            return
        value = sval
    if key in info:
        existing = info[key]
        if isinstance(existing, str):
            if existing.strip() and not existing.strip().startswith("(title missing"):
                return
        elif existing not in {None, "", 0}:
            return
    info[key] = value


def _find_prefixed_value(row: Dict[str, Any], prefix: str, suffix: str, allow_unprefixed: bool = False) -> Optional[Any]:
    """Return the first value in ``row`` that matches a ``prefix`` + ``suffix`` naming pattern."""
    candidates = []
    if prefix:
        candidates.extend(
            [
                f"{prefix}{suffix}",
                f"{prefix}_{suffix}",
                f"{prefix}{suffix.lower()}",
                f"{prefix}_{suffix.lower()}",
                f"{prefix}{suffix.upper()}",
                f"{prefix}_{suffix.upper()}",
            ]
        )
    if allow_unprefixed or not prefix:
        candidates.extend([suffix, suffix.lower(), suffix.upper()])
    for cand in candidates:
        if cand in row and row[cand] not in (None, ""):
            return row[cand]
    return None


def _apply_metadata_fields(info: Dict[str, Any], row: Dict[str, Any], prefix: str, allow_unprefixed: bool = False) -> None:
    """Copy standard video metadata fields from ``row`` into ``info`` using a column prefix."""
    for dest, suffixes in TEXT_SUFFIXES.items():
        for suffix in suffixes:
            val = _find_prefixed_value(row, prefix, suffix, allow_unprefixed)
            if val is not None:
                _assign_if_missing(info, dest, str(val))
                break
    for dest, suffixes in NUMERIC_SUFFIXES.items():
        for suffix in suffixes:
            val = _find_prefixed_value(row, prefix, suffix, allow_unprefixed)
            if val is not None:
                _assign_if_missing(info, dest, _coerce_session_value(val))
                break


def _augment_metadata_from_supplemental(meta: Dict[str, Dict[str, Any]], base_dir: Path) -> None:
    """Enrich the metadata index with additional CSVs under ``supplemental/metadata and ratings``."""
    sup_dir = base_dir / "supplemental" / "metadata and ratings"
    if not sup_dir.exists():
        return
    for csv_path in sup_dir.glob("*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8-sig") as fp:
                reader = csv.DictReader(fp)
                if not reader.fieldnames:
                    continue
                for row in reader:
                    for prefix, id_keys, allow_unprefixed in [
                        ("origin", ["originID", "originId", "origin_id"], True),
                        ("rec", ["recID", "recId", "rec_id"], False),
                        ("video", ["video_id", "videoId", "videoID"], True),
                        ("", ["id", "ID"], True),
                    ]:
                        vid_raw = None
                        for key in id_keys:
                            if key in row and row[key]:
                                vid_raw = row[key]
                                break
                        if not vid_raw:
                            continue
                        vid = _strip_session_video_id(str(vid_raw))
                        if not vid:
                            continue
                        info = meta.setdefault(vid, {"id": vid})
                        _apply_metadata_fields(info, row, prefix, allow_unprefixed)
        except Exception as exc:
            log.warning("Failed to read supplemental metadata %s: %s", csv_path, exc)


def _load_recommendation_tree_metadata(base_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Scan recommendation tree CSVs to build a metadata index keyed by video id."""
    tree_root = base_dir / "recommendation trees"
    if not tree_root.exists():
        return {}, {}

    meta: Dict[str, Dict[str, Any]] = {}
    issue_map: Dict[str, str] = {}

    issue_dirs = [
        ("gun_control", tree_root / "trees_gun"),
        ("minimum_wage", tree_root / "trees_wage"),
    ]

    for issue_name, folder in issue_dirs:
        if not folder.exists():
            continue
        for csv_path in folder.glob("*.csv"):
            try:
                with csv_path.open("r", encoding="utf-8-sig") as fp:
                    reader = csv.DictReader(fp)
                    for row in reader:
                        origin_raw = row.get("originId") or row.get("originID") or ""
                        base_vid = _strip_session_video_id(origin_raw)
                        if not base_vid:
                            continue
                        info = meta.setdefault(base_vid, {"id": base_vid})
                        issue_map.setdefault(base_vid, issue_name)
                        if origin_raw and origin_raw != base_vid:
                            raw_ids = info.setdefault("raw_ids", [])
                            if origin_raw not in raw_ids:
                                raw_ids.append(origin_raw)

                        def _fill(key: str, *cols: str) -> None:
                            if info.get(key):
                                return
                            for col in cols:
                                if not col:
                                    continue
                                val = row.get(col)
                                if val is None:
                                    continue
                                sval = str(val).strip()
                                if not sval or _is_nanlike(sval):
                                    continue
                                info[key] = sval
                                return

                        _fill("title", "originTitle", "title", "video_title")
                        _fill("channel_id", "originChannelId", "channel_id")
                        _fill("channel_title", "originChannel", "channelTitle")
                        _fill("description", "originDescription", "description")
                        _fill("duration", "originDuration", "duration")
                        _apply_metadata_fields(info, row, "origin", allow_unprefixed=True)
                        for thumb_key, col in [
                            ("thumbnail_default", "originThumbnailDefault"),
                            ("thumbnail_medium", "originThumbnailMedium"),
                            ("thumbnail_high", "originThumbnailHigh"),
                        ]:
                            if info.get(thumb_key):
                                continue
                            val = row.get(col)
                            if val is None:
                                continue
                            sval = str(val).strip()
                            if sval:
                                info[thumb_key] = sval
                        if "duration" in info:
                            dur_val = _coerce_session_value(info.get("duration"))
                            if dur_val is None:
                                info.pop("duration", None)
                            else:
                                info["duration"] = dur_val

                        if info.get("recs"):
                            continue  # keep only the first observed slate for this origin
                        rec_entries = []
                        rec_cols = [
                            (int(m.group(1)), col)
                            for col in row.keys()
                            for m in [re.match(r"rec(\d+)", str(col), re.I)]
                            if m is not None
                        ]
                        rec_cols.sort(key=lambda x: x[0])
                        if rec_cols:
                            for _, col in rec_cols:
                                raw_rec = row.get(col)
                                if raw_rec is None:
                                    continue
                                raw_str = str(raw_rec).strip()
                                if not raw_str or _is_nanlike(raw_str):
                                    continue
                                rec_base = _strip_session_video_id(raw_str)
                                if not rec_base:
                                    continue
                                already = {entry.get("id") for entry in rec_entries if isinstance(entry, dict)}
                                if rec_base in already:
                                    continue
                                rec_entry: Dict[str, Any] = {"id": rec_base}
                                if raw_str != rec_base:
                                    rec_entry["raw_id"] = raw_str
                                rec_entries.append(rec_entry)
                            if rec_entries:
                                info["recs"] = rec_entries
                        _apply_metadata_fields(info, row, "origin", allow_unprefixed=True)
            except Exception as exc:
                log.warning("Failed to read tree CSV %s: %s", csv_path, exc)
                continue

    _augment_metadata_from_supplemental(meta, base_dir)
    return meta, issue_map


def _infer_issue_from_topic(topic_id: str) -> Optional[str]:
    topic = (topic_id or "").strip().lower()
    if not topic:
        return None
    if "wage" in topic:
        return "minimum_wage"
    if "gun" in topic:
        return "gun_control"
    if "pro" in topic or "anti" in topic:
        # Heuristic: April 2022 topics correspond to gun control content
        if "april" in topic or "shoot" in topic:
            return "gun_control"
    return None


def _infer_participant_study(
    issue: str,
    survey_row: Optional[Dict[str, Any]],
    topic_id: str,
    session: Dict[str, Any],
) -> str:
    issue_norm = (issue or "").strip().lower()
    topic_norm = (topic_id or "").strip().lower()
    survey_keys = {str(k).strip().lower() for k in (survey_row or {}).keys()}

    if issue_norm == "gun_control":
        return "study1"

    if issue_norm == "minimum_wage":
        if survey_keys:
            if any(key in survey_keys for key in {"caseid", "sample", "weight"}):
                return "study3"
            if any(key in survey_keys for key in {"worker_id", "workerid", "assignment_id", "assignmentid", "hit_id", "hitid"}):
                return "study2"
        session_keys = {str(k).strip().lower() for k in session.keys()}
        if "shorts" in topic_norm or "2024" in topic_norm or "rabbit" in topic_norm:
            return "study4"
        if any(key.startswith("short") for key in session_keys):
            return "study4"

    return "unknown"


def _coerce_session_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            if "." in s:
                num = float(s)
                if num.is_integer():
                    return int(num)
                return num
            return int(s)
        except ValueError:
            return s
    return value


def _normalize_session_mapping(values: Any, raw_vids: List[str], base_vids: List[str]) -> Dict[str, Any]:
    """Convert per-video session arrays/dicts into a standard lookup dict."""
    mapping: Dict[str, Any] = {}
    if isinstance(values, dict):
        for key, val in values.items():
            if key is None:
                continue
            mapping[str(key)] = _coerce_session_value(val)
        return mapping
    if isinstance(values, list):
        for idx, val in enumerate(values):
            coerced = _coerce_session_value(val)
            if idx < len(raw_vids):
                mapping.setdefault(raw_vids[idx], coerced)
            if idx < len(base_vids):
                mapping.setdefault(base_vids[idx], coerced)
            mapping.setdefault(str(idx), coerced)
        return mapping
    return mapping


def _lookup_session_value(mapping: Dict[str, Any], raw_id: str, base_id: str) -> Any:
    if not mapping:
        return None
    for key in (raw_id, base_id, str(raw_id), str(base_id)):
        if key and key in mapping:
            return mapping[key]
    return None


def _build_codeocean_rows(data_root: Path) -> pd.DataFrame:
    sessions_path = data_root / "platform session data" / "sessions.json"
    with open(sessions_path, "r", encoding="utf-8") as fp:
        sessions = json.load(fp)

    capsule_root = data_root.parent
    survey_gun = _read_survey_with_fallback(
        capsule_root / "intermediate data" / "gun control (issue 1)" / "guncontrol_qualtrics_w123_clean.csv",
        capsule_root / "results" / "intermediate data" / "gun control (issue 1)" / "guncontrol_qualtrics_w123_clean.csv",
    )
    survey_wage_sources: List[pd.DataFrame] = []

    survey_wage_qualtrics = _read_survey_with_fallback(
        capsule_root / "intermediate data" / "minimum wage (issue 2)" / "qualtrics_w12_clean.csv",
        capsule_root / "results" / "intermediate data" / "minimum wage (issue 2)" / "qualtrics_w12_clean.csv",
    )
    if not survey_wage_qualtrics.empty:
        survey_wage_sources.append(survey_wage_qualtrics)

    yougov_dirs = [
        capsule_root / "intermediate data" / "minimum wage (issue 2)",
        capsule_root / "results" / "intermediate data" / "minimum wage (issue 2)",
    ]
    for folder in yougov_dirs:
        if not folder.exists():
            continue
        for csv_path in sorted(folder.glob("yg_*_clean.csv")):
            yougov_df = _read_csv_if_exists(csv_path)
            if yougov_df.empty:
                continue
            survey_wage_sources.append(yougov_df)

    if survey_wage_sources:
        survey_wage = pd.concat(survey_wage_sources, ignore_index=True, sort=False)
        if "urlid" in survey_wage.columns:
            dedupe_subset = ["urlid"]
            if "topic_id" in survey_wage.columns:
                dedupe_subset.append("topic_id")
            survey_wage = survey_wage.drop_duplicates(subset=dedupe_subset, keep="first").reset_index(drop=True)
    else:
        survey_wage = pd.DataFrame()

    surveys = {
        "gun_control": _build_survey_index(survey_gun),
        "minimum_wage": _build_survey_index(survey_wage),
    }

    tree_meta, tree_issue_map = _load_recommendation_tree_metadata(data_root)
    fallback_titles = _load_video_metadata(data_root)

    def _video_meta(base_id: str, raw_id: str = "") -> Dict[str, Any]:
        info = dict(tree_meta.get(base_id) or {})
        if raw_id and raw_id != base_id:
            raw_ids = list(info.get("raw_ids") or [])
            if raw_id not in raw_ids:
                raw_ids.append(raw_id)
            if raw_ids:
                info["raw_ids"] = raw_ids
        if not info.get("title"):
            title = fallback_titles.get(base_id) or fallback_titles.get(raw_id)
            if title:
                info["title"] = title
        if not info.get("issue"):
            issue_guess = tree_issue_map.get(base_id)
            if issue_guess:
                info["issue"] = issue_guess
        return info

    def _resolve_title(meta: Dict[str, Any], fallback_id: str) -> Tuple[str, bool]:
        """Return a display title and a missing flag for a video metadata record."""
        title = str(meta.get("title") or "").strip()
        if title:
            return title, False
        return f"(title missing for {fallback_id})", True


    def _resolve_channel(meta: Dict[str, Any]) -> Tuple[str, bool]:
        """Return a displayable channel title and a missing flag for a video metadata record."""
        channel = str(meta.get("channel_title") or "").strip()
        if channel:
            return channel, False
        return "(channel missing)", True

    def _attach_recommendation_stats(
        target: Dict[str, Any],
        meta: Optional[Dict[str, Any]],
        raw: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Populate slate item stats such as view/like counts and duration."""
        stat_fields = ("view_count", "like_count", "dislike_count", "favorite_count", "comment_count")
        for field in stat_fields:
            value = None
            if raw and raw.get(field) not in (None, ""):
                value = raw.get(field)
            elif meta and meta.get(field) not in (None, ""):
                value = meta.get(field)
            if value is None:
                continue
            coerced = _coerce_session_value(value)
            if coerced is not None:
                target[field] = coerced

        duration_value = None
        duration_candidates = ("duration", "duration_seconds", "length_seconds", "total_length")
        if raw:
            for cand in duration_candidates:
                raw_val = raw.get(cand)
                if raw_val not in (None, ""):
                    duration_value = raw_val
                    break
        if duration_value is None and meta and meta.get("duration") not in (None, ""):
            duration_value = meta.get("duration")
        if duration_value is not None:
            coerced_duration = _coerce_session_value(duration_value)
            if coerced_duration is not None:
                target["duration"] = coerced_duration
                target.setdefault("duration_seconds", coerced_duration)

    def _tree_slate_items(base_id: str) -> List[Dict[str, Any]]:
        """Return recommendation slate items stored alongside the tree metadata for ``base_id``."""
        info = tree_meta.get(base_id)
        if not isinstance(info, dict):
            return []
        recs = info.get("recs")
        if not isinstance(recs, list) or not recs:
            return []
        items: List[Dict[str, Any]] = []
        for rec in recs:
            if isinstance(rec, dict):
                rec_id = str(rec.get("id") or "").strip()
                rec_raw = str(rec.get("raw_id") or "").strip()
            else:
                rec_raw = str(rec).strip()
                rec_id = _strip_session_video_id(rec_raw)
            if not rec_id:
                continue
            if not rec_raw:
                rec_raw = rec_id
            rec_meta = _video_meta(rec_id, rec_raw)
            rec_title, rec_title_missing = _resolve_title(rec_meta, rec_id)
            rec_channel, rec_channel_missing = _resolve_channel(rec_meta)
            item = {
                "id": rec_id,
                "raw_id": rec_raw,
                "title": rec_title,
                "title_missing": rec_title_missing,
                "channel_title": rec_channel,
                "channel_missing": rec_channel_missing,
            }
            if rec_meta.get("channel_id"):
                item["channel_id"] = rec_meta["channel_id"]
            rec_info = meta.setdefault(rec_id, {"id": rec_id})
            if not rec_title_missing:
                _assign_if_missing(rec_info, "title", rec_title)
            elif "title" not in rec_info:
                rec_info["title"] = rec_title
            if rec_channel and not rec_channel_missing:
                _assign_if_missing(rec_info, "channel_title", rec_channel)
            elif "channel_title" not in rec_info:
                rec_info["channel_title"] = rec_channel
            if rec_meta.get("channel_id"):
                _assign_if_missing(rec_info, "channel_id", rec_meta.get("channel_id"))
            raw_dict = rec if isinstance(rec, dict) else None
            _attach_recommendation_stats(item, rec_meta, raw_dict)
            items.append(item)
        return items

    rows: List[Dict[str, Any]] = []
    interaction_stats: Counter[str] = Counter()

    for sess in sessions:
        interaction_stats["sessions_total"] += 1
        raw_vids = [str(v).strip() for v in (sess.get("vids") or []) if isinstance(v, str) and str(v).strip()]
        if len(raw_vids) < 2:
            interaction_stats["sessions_too_short"] += 1
            continue
        base_vids = [_strip_session_video_id(v) for v in raw_vids]
        if not all(base_vids):
            interaction_stats["sessions_invalid_ids"] += 1
            continue

        interaction_stats["pairs_total"] += max(0, len(base_vids) - 1)

        start_times = _normalize_session_mapping(sess.get("vidStartTimes"), raw_vids, base_vids)
        end_times = _normalize_session_mapping(sess.get("vidEndTimes"), raw_vids, base_vids)
        watch_times = _normalize_session_mapping(sess.get("vidWatchTimes"), raw_vids, base_vids)
        total_lengths = _normalize_session_mapping(sess.get("vidTotalLengths"), raw_vids, base_vids)
        delays = _normalize_session_mapping(sess.get("contentStartDelay"), raw_vids, base_vids)

        watched_details: List[Dict[str, Any]] = []
        for idx, raw_vid in enumerate(raw_vids):
            base = base_vids[idx]
            meta = _video_meta(base, raw_vid)
            title_val, title_missing = _resolve_title(meta, base)
            channel_val, channel_missing = _resolve_channel(meta)
            entry: Dict[str, Any] = {
                "id": base,
                "raw_id": raw_vid,
                "idx": idx,
                "title": title_val,
                "title_missing": title_missing,
                "channel_title": channel_val,
                "channel_missing": channel_missing,
            }
            if meta.get("channel_id"):
                entry["channel_id"] = meta["channel_id"]
            delay_val = _lookup_session_value(delays, raw_vid, base)
            if delay_val is not None:
                entry["start_delay_ms"] = delay_val

            start_val = _lookup_session_value(start_times, raw_vid, base)
            if start_val is not None:
                entry["start_ms"] = start_val

            end_val = _lookup_session_value(end_times, raw_vid, base)
            if end_val is not None:
                entry["end_ms"] = end_val

            watch_val = _lookup_session_value(watch_times, raw_vid, base)
            if watch_val is not None:
                entry["watch_seconds"] = watch_val

            length_val = _lookup_session_value(total_lengths, raw_vid, base)
            if length_val is not None:
                    entry["total_length"] = length_val
            elif meta.get("duration"):
                dur = _coerce_session_value(meta.get("duration"))
                if dur is not None:
                    entry.setdefault("total_length", dur)
            watched_details.append(entry)

        watched_vids_json = list(base_vids)

        topic = str(sess.get("topicID") or "")
        urlid = _normalize_urlid(sess.get("urlid"))
        session_id = str(sess.get("sessionID") or "").strip()

        issue = None
        issue_source = "unresolved"
        issue_detail = ""
        for base in base_vids:
            issue = tree_issue_map.get(base)
            if issue:
                issue_source = "tree_metadata"
                issue_detail = f"derived from video {base}"
                break
        if not issue:
            topic_issue = _infer_issue_from_topic(topic)
            if topic_issue:
                issue = topic_issue
                issue_source = "topic"
                issue_detail = f"matched topic '{topic}'"
        survey_rows: List[Dict[str, Any]] = []
        if issue and issue in surveys:
            survey_rows = surveys[issue].get(urlid, [])
        else:
            for issue_name, idx_map in surveys.items():
                if urlid and urlid in idx_map:
                    survey_rows = idx_map[urlid]
                    if not issue:
                        issue = issue_name
                        issue_source = "survey_url"
                        issue_detail = f"urlid match '{urlid}'"
                    break
        if not issue:
            issue = "unknown"
            issue_detail = f"no tree/topic/survey match (urlid={urlid or '(missing)'}, topic={topic or '(missing)'})"

        survey_row = _select_survey_row(survey_rows, topic or "")

        display_orders = sess.get("displayOrders") or {}
        display_orders_struct: Dict[str, List[Dict[str, Any]]] = {}
        original_slate_keys: set[str] = set()
        for key, values in display_orders.items():
            if not isinstance(values, list):
                continue
            items = []
            for raw in values:
                raw_str = str(raw).strip()
                if not raw_str:
                    continue
                base = _strip_session_video_id(raw_str)
                if not base:
                    continue
                meta = _video_meta(base, raw_str)
                title_val, title_missing = _resolve_title(meta, base)
                channel_val, channel_missing = _resolve_channel(meta)
                item = {
                    "id": base,
                    "raw_id": raw_str,
                    "title": title_val,
                    "title_missing": title_missing,
                    "channel_title": channel_val,
                    "channel_missing": channel_missing,
                }
                if meta.get("channel_id"):
                    item["channel_id"] = meta["channel_id"]
                _attach_recommendation_stats(item, meta)
                items.append(item)
            if items:
                key_str = str(key)
                display_orders_struct[key_str] = items
                original_slate_keys.add(key_str)

        for idx, current_base in enumerate(base_vids[:-1]):
            step_key = f"{idx + 2}-recs"
            existing = display_orders_struct.get(step_key)
            if existing:
                continue
            fallback_items = _tree_slate_items(current_base)
            if fallback_items:
                display_orders_struct[step_key] = fallback_items

        trajectory_payload = {
            "session_id": session_id,
            "urlid": urlid,
            "topic_id": topic,
            "order": [dict(item) for item in watched_details],
            "displayOrders": {k: [dict(it) for it in v] for k, v in display_orders_struct.items()},
        }
        trajectory_json = json.dumps(trajectory_payload)

        rec_slates_by_step: Dict[int, List[Dict[str, Any]]] = {}
        for key, items in display_orders_struct.items():
            if not re.search(r"rec", key, re.I):
                continue
            m = re.search(r"(\d+)", key)
            if not m:
                continue
            step_num = int(m.group(1))
            rec_slates_by_step[step_num] = items

        for idx in range(len(base_vids) - 1):
            current_base = base_vids[idx]
            next_base = base_vids[idx + 1]
            step_num = idx + 2  # display order keys start at 2
            step_key = f"{step_num}-recs"
            rec_items_raw = rec_slates_by_step.get(step_num, [])
            if not rec_items_raw:
                rec_items_raw = display_orders_struct.get(step_key, [])

            normalized_items: List[Dict[str, Any]] = []
            seen_ids: set[str] = set()
            for it in rec_items_raw:
                if not isinstance(it, dict):
                    continue
                vid = str(it.get("id") or "").strip()
                if not vid:
                    continue
                if vid in seen_ids:
                    continue
                seen_ids.add(vid)
                raw_id = str(it.get("raw_id") or vid)
                meta_for_norm = _video_meta(vid, raw_id)
                norm = {
                    "id": vid,
                    "raw_id": raw_id,
                }
                title_val = str(it.get("title") or "").strip()
                title_missing = bool(it.get("title_missing"))
                channel_val = str(it.get("channel_title") or "").strip()
                channel_missing = bool(it.get("channel_missing"))
                if not title_val:
                    title_val, title_missing = _resolve_title(meta_for_norm, vid)
                if not channel_val:
                    channel_val, channel_missing = _resolve_channel(meta_for_norm)
                norm["title"] = title_val
                norm["title_missing"] = title_missing
                norm["channel_title"] = channel_val if channel_val else "(channel missing)"
                norm["channel_missing"] = channel_missing
                if it.get("channel_id"):
                    norm["channel_id"] = it["channel_id"]
                elif meta_for_norm and meta_for_norm.get("channel_id"):
                    norm["channel_id"] = meta_for_norm["channel_id"]
                _attach_recommendation_stats(norm, meta_for_norm, it)
                normalized_items.append(norm)

            if next_base and next_base not in seen_ids:
                next_meta = _video_meta(next_base, raw_vids[idx + 1])
                next_title, next_title_missing = _resolve_title(next_meta, next_base)
                next_channel, next_channel_missing = _resolve_channel(next_meta)
                norm = {
                    "id": next_base,
                    "raw_id": raw_vids[idx + 1],
                    "title": next_title,
                    "title_missing": next_title_missing,
                    "channel_title": next_channel,
                    "channel_missing": next_channel_missing,
                }
                if next_meta.get("channel_id"):
                    norm["channel_id"] = next_meta["channel_id"]
                _attach_recommendation_stats(norm, next_meta)
                normalized_items.append(norm)
                seen_ids.add(next_base)

            if not normalized_items:
                interaction_stats["pairs_no_options"] += 1
                continue

            current_meta = _video_meta(current_base, raw_vids[idx])
            next_meta = _video_meta(next_base, raw_vids[idx + 1])
            current_title, current_title_missing = _resolve_title(current_meta, current_base)
            current_channel, current_channel_missing = _resolve_channel(current_meta)
            next_title, next_title_missing = _resolve_title(next_meta, next_base)
            next_channel, next_channel_missing = _resolve_channel(next_meta)

            issue_value = str(issue or "").strip() or "unknown"
            interaction_stats[f"pairs_issue_{issue_value}"] += 1

            if step_key in original_slate_keys:
                slate_source = "session_log"
            elif rec_slates_by_step.get(step_num):
                slate_source = "tree_fallback"
            else:
                slate_source = "next_video_only"

            row: Dict[str, Any] = {
                "issue": issue_value,
                "issue_source": issue_source,
                "issue_detail": issue_detail,
                "urlid": urlid,
                "topic_id": topic,
                "session_id": session_id,
                "step_index": idx + 1,
                "display_step": step_num,
                "display_order_key": step_key,
                "slate_source": slate_source,
                "current_video_id": current_base,
                "current_video_raw_id": raw_vids[idx],
                "current_video_title": current_title,
                "current_video_title_missing": current_title_missing,
                "current_video_channel": current_channel,
                "current_video_channel_missing": current_channel_missing,
                "current_video_channel_id": current_meta.get("channel_id") or "",
                "slate_items_json": normalized_items,
                "n_options": len(normalized_items),
                "next_video_id": next_base,
                "next_video_raw_id": raw_vids[idx + 1],
                "next_video_title": next_title,
                "next_video_title_missing": next_title_missing,
                "next_video_channel": next_channel,
                "next_video_channel_missing": next_channel_missing,
                "next_video_channel_id": next_meta.get("channel_id") or "",
                "watched_detailed_json": [dict(item) for item in watched_details],
                "watched_vids_json": list(watched_vids_json),
                "trajectory_json": trajectory_json,
                "percent_visible": sess.get("percentVisible"),
                "session_finished": bool(sess.get("sessionFinished")),
                "start_time_ms": sess.get("startTime"),
                "end_time_ms": sess.get("endTime"),
            }

            row["participant_study"] = _infer_participant_study(
                issue_value,
                survey_row or {},
                topic,
                sess,
            )

            if survey_row:
                for k, v in survey_row.items():
                    if k not in row and v is not None:
                        row[k] = v

            rows.append(row)
            interaction_stats["pairs_emitted"] += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        sample_rows = df.head(2).to_dict(orient="records")
        log.info("Sample cleaned rows (first 2): %s", json.dumps(sample_rows, ensure_ascii=False))
        demographic_cols = [c for c in DEMOGRAPHIC_COLUMNS if c in df.columns]
        if demographic_cols:
            has_demo_mask = df[demographic_cols].apply(
                lambda row: any(not _is_nanlike(row.get(col)) for col in demographic_cols),
                axis=1,
            )
            dropped = int((~has_demo_mask).sum())
            if dropped:
                log.info(
                    "Dropping %d rows missing demographic fields (checked columns=%s)",
                    dropped,
                    demographic_cols,
                )
            df = df.loc[has_demo_mask].reset_index(drop=True)
    log.info("Interaction statistics: %s", dict(interaction_stats))
    return df


def _split_dataframe(df: pd.DataFrame, validation_ratio: float = 0.1) -> Dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    if not 0 < validation_ratio < 1:
        validation_ratio = 0.1

    def _pick_group(row_idx: int) -> str:
        urlid = str(df.iloc[row_idx].get("urlid") or "").strip()
        session = str(df.iloc[row_idx].get("session_id") or "").strip()
        if urlid and urlid.lower() != "nan":
            return f"urlid::{urlid}"
        if session and session.lower() != "nan":
            return f"session::{session}"
        return f"row::{row_idx}"

    group_keys = [_pick_group(i) for i in range(len(df))]
    unique_groups = list(dict.fromkeys(group_keys))
    rng = random.Random(2024)
    rng.shuffle(unique_groups)

    val_group_count = max(1, int(len(unique_groups) * validation_ratio)) if len(unique_groups) > 1 else 0
    val_groups = set(unique_groups[:val_group_count]) if val_group_count else set()
    is_val = pd.Series(group_keys).isin(val_groups)

    splits: Dict[str, pd.DataFrame] = {
        "train": df.loc[~is_val].reset_index(drop=True),
    }
    if val_groups:
        splits["validation"] = df.loc[is_val].reset_index(drop=True)
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
    out["slate_items_with_meta"] = _as_list_json(ex.get("slate_items_json"))

    passthrough = {
        "issue", "rating_copy", "rating_index", "rating_video_id", "urlid", "topic_id",
        "session_id", "step_index", "display_step", "display_order_key",
        "current_video_raw_id", "current_video_channel", "current_video_channel_id",
        "next_video_id", "next_video_raw_id", "next_video_title", "next_video_channel", "next_video_channel_id",
        "percent_visible", "session_finished", "start_time_ms", "end_time_ms", "trajectory_json",
        "issue_source", "issue_detail", "slate_source",
    }
    for extra in passthrough:
        if extra in ex:
            out[extra] = ex.get(extra)

    for key, val in ex.items():
        if key in out:
            continue
        cleaned = val
        try:
            if pd.isna(cleaned):  # type: ignore[arg-type]
                cleaned = None
        except Exception:
            pass
        out[key] = cleaned
    return out

# ---------- driver ----------
def _load_codeocean_dataset(dataset_name: str, validation_ratio: float = 0.1) -> DatasetDict:
    root = Path(dataset_name).expanduser()
    data_root = _resolve_capsule_data_root(root)
    if not data_root:
        raise ValueError(f"CodeOcean capsule data not found under {dataset_name}")
    log.info("Building dataset from CodeOcean capsule at %s", data_root)
    df = _build_codeocean_rows(data_root)
    if df is None or df.empty:
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
    ap.add_argument(
        "--prompt-stats-dir",
        default=None,
        help="If set, generate prompt feature histograms and summary statistics into this directory.",
    )
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

    # Domain coverage diagnostics for GRPO
    issue_counts: Dict[str, Dict[str, int]] = {}
    for split_name, split_ds in raw.items():
        if "issue" not in split_ds.column_names:
            continue
        counter = Counter(str(x or "").strip() for x in split_ds["issue"])  # type: ignore[index]
        issue_counts[split_name] = {k or "(missing)": v for k, v in counter.items()}
    if issue_counts:
        log.info("Issue distribution per split: %s", issue_counts)

    # Map to cleaned schema
    mapped = raw.map(
        lambda ex: _row_to_example(ex, args.system_prompt, sol_key, max_hist=args.max_history),
        load_from_cache_file=False
    )

    # Drop any None rows (defensive)
    for split in list(mapped.keys()):
        mapped[split] = mapped[split].filter(lambda x: x is not None)

    # Keep only the columns trainer/rewards need (+ a few meta)
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

    for split_name, split_ds in final.items():
        missing = sorted(REQUIRED_FOR_GRPO - set(split_ds.column_names))
        if missing:
            raise ValueError(f"Split '{split_name}' is missing required columns for GRPO: {missing}")

    final.save_to_disk(args.output_dir)
    log.info("Done. Rows: %s", {k: len(v) for k, v in final.items()})

    if args.prompt_stats_dir:
        if {"train", "validation"}.issubset(final.keys()):
            try:
                from clean_data.prompt_stats import generate_prompt_feature_report  # type: ignore
            except ModuleNotFoundError:
                from prompt_stats import generate_prompt_feature_report  # type: ignore

            log.info("Generating prompt feature report under %s", args.prompt_stats_dir)
            generate_prompt_feature_report(
                final,
                output_dir=Path(args.prompt_stats_dir),
                train_split="train",
                validation_split="validation",
            )
        else:
            log.warning(
                "Prompt stats requested (dir=%s) but dataset lacks both 'train' and 'validation' splits; skipping.",
                args.prompt_stats_dir,
            )

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
