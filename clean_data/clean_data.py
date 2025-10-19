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

import argparse
import ast
import csv
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import datasets
from datasets import DatasetDict, Features, Sequence, Value
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
    "minimum_wage": {
        "1": "min_wage_raise",
        "2": "min_wage_no_raise",
        "3": "min_wage_unknown",
    },
    "gun_control": {
        "1": "gun_more_restrictions",
        "2": "gun_fewer_restrictions",
        "3": "gun_unknown",
    },
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

def _canon(text: str) -> str:
    """Normalize a label by lowercasing and removing punctuation.

    :param text: Input string that may contain mixed case or punctuation.
    :return: Canonical lower-case alphanumeric string used for comparisons.
    """

    return re.sub(r"[^a-z0-9]+", "", (text or "").lower().strip())


def _canon_vid(value: str) -> str:
    """Extract the canonical 11-character YouTube id from a raw identifier.

    :param value: Raw video identifier or URL fragment emitted by the platform logs.
    :return: Canonical YouTube id, or an empty string when not parseable.
    """

    if not isinstance(value, str):
        return ""
    match = YTID_RE.search(value)
    return match.group(1) if match else value.strip()


def _is_nanlike(value: Any) -> bool:
    """Determine whether a value represents a missing token.

    :param value: Arbitrary scalar loaded from CSV/JSON sources.
    :return: ``True`` if the value should be treated as missing, ``False`` otherwise.
    """

    if value is None:
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "n/a"}


def _as_list_json(value: Any, default: str = "[]") -> list:
    """Convert serialized list-like values into Python lists.

    :param value: Value that may already be a list, JSON string, or Arrow array.
    :param default: JSON literal used when ``value`` is empty.
    :return: Python list representation (empty when parsing fails).
    """

    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            v = json.loads(value or default)
            return v if isinstance(v, list) else []
        except (TypeError, json.JSONDecodeError):
            return []
    # pyarrow List?
    try:
        import pyarrow as pa  # type: ignore
    except ImportError:
        return []

    if isinstance(value, pa.Array):
        return value.to_pylist()
    return []


def _strip_session_video_id(vid: str) -> str:
    """Reduce a raw session video identifier to the canonical YouTube id.

    :param vid: Raw identifier stored in the session logs.
    :return: Canonical 11-character video id, or the original string when parsing fails.
    """

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
    """Locate the CodeOcean capsule root or ``data`` directory from a user path.

    :param path: User-supplied directory that may point at the capsule root or ``data`` subdir.
    :return: Normalized path containing ``platform session data/sessions.json`` or ``None``.
    """

    if not path.exists():
        return None
    if (path / "platform session data" / "sessions.json").exists():
        return path
    if (path / "data" / "platform session data" / "sessions.json").exists():
        return path / "data"
    return None


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read a CSV file into a dataframe, returning empty on failure.

    :param path: Filesystem path to the CSV file.
    :return: Dataframe of the file contents or an empty frame when missing/invalid.
    """

    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        log.error("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def _read_survey_with_fallback(*candidates: Path) -> pd.DataFrame:
    """Return the first survey export that includes a ``urlid`` column.

    :param candidates: Ordered paths to possible survey CSV files.
    :return: Dataframe for the first candidate containing ``urlid``; falls back to the
        first empty frame when none qualify.
    """
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
    """Standardize URL identifiers for dictionary lookups.

    :param value: Raw identifier that may be numeric or a string with trailing decimals.
    :return: Cleaned string identifier suitable for use as a key.
    """

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
    except ValueError:
        pass
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _build_survey_index(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Create a mapping from ``urlid`` to associated survey rows.

    :param df: Survey dataframe that contains participant metadata.
    :return: Dictionary keyed by ``urlid`` with lists of row dictionaries.
    """

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
    """Choose the most appropriate survey row for a session.

    :param rows: All survey rows sharing the same ``urlid``.
    :param topic_id: Topic identifier from the interaction log.
    :return: Matching survey row when a topic-specific entry exists, otherwise the first row.
    """

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
    """Load supplemental video titles from the metadata bundle.

    :param base_dir: Capsule directory that contains ``supplemental/metadata and ratings``.
    :return: Mapping of canonical video ids to titles.
    """

    meta: Dict[str, str] = {}
    meta_dir = base_dir / "supplemental" / "metadata and ratings"
    if not meta_dir.exists():
        return meta
    for csv_path in meta_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            log.warning("Failed to read metadata %s: %s", csv_path, exc)
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

YOUTUBE_FREQ_MAP = {
    "0": "rarely",
    "1": "occasionally",
    "2": "a few times a month",
    "3": "weekly",
    "4": "several times a week",
    "5": "daily",
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


def _clean_str(value: Any) -> str:
    """Return a trimmed string, converting ``None`` to an empty string."""

    return str(value).strip() if value is not None else ""


def _truthy_str_flag(value: Any) -> Optional[bool]:
    """Interpret common string boolean markers."""

    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _last_index(values: Any, target: Any) -> Optional[int]:
    """Return the last index of ``target`` inside ``values`` when ``values`` is a list."""

    if not isinstance(values, list) or target is None:
        return None
    last: Optional[int] = None
    for index, candidate in enumerate(values):
        if candidate == target:
            last = index
    return last


def _find_prefixed_value(
    row: Dict[str, Any],
    prefix: str,
    suffix: str,
    allow_unprefixed: bool = False,
) -> Optional[Any]:
    """Locate the first value that matches a ``prefix`` + ``suffix`` naming pattern.

    :param row: Metadata row dictionary.
    :param prefix: Column prefix (for example ``origin`` or ``target``).
    :param suffix: Field suffix such as ``Title`` or ``ChannelId``.
    :param allow_unprefixed: Whether to search unprefixed variants when prefixed columns are missing.
    :return: Matching value from ``row`` or ``None`` when not present.
    """
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


def _apply_metadata_fields(
    info: Dict[str, Any],
    row: Dict[str, Any],
    prefix: str,
    allow_unprefixed: bool = False,
) -> None:
    """Copy common video metadata fields from ``row`` into ``info`` using a column prefix.

    :param info: Destination metadata dictionary.
    :param row: Source row produced by tree/supplemental CSVs.
    :param prefix: Column prefix group (e.g., ``origin``).
    :param allow_unprefixed: Whether to consider suffix-only columns when prefixed ones are absent.
    """
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
    """Enrich the metadata index with supplemental CSV content.

    :param meta: Mapping of video ids to metadata dictionaries.
    :param base_dir: Capsule root that holds the supplemental metadata folder.
    """
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
        except (OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
            log.warning("Failed to read supplemental metadata %s: %s", csv_path, exc)


def _fill_metadata_field(
    info: Dict[str, Any],
    row: Dict[str, Any],
    key: str,
    *candidate_columns: str,
) -> None:
    """Populate ``info[key]`` from the first valid entry in ``candidate_columns``.

    :param info: Destination metadata dictionary being populated.
    :param row: Source row containing potential values.
    :param key: Target key in ``info``.
    :param candidate_columns: Ordered column names to probe inside ``row``.
    """
    if info.get(key):
        return
    for col in candidate_columns:
        if not col:
            continue
        value = row.get(col)
        if value is None:
            continue
        sval = str(value).strip()
        if not sval or _is_nanlike(sval):
            continue
        info[key] = sval
        return


def _escape_r_string(path: Path) -> str:
    """Escape a filesystem path so it can be embedded in inline R code.

    :param path: Filesystem path to escape.
    :return: Path with backslashes normalized and single quotes escaped.
    """

    text = str(path)
    text = text.replace("\\", "/")
    return text.replace("'", "\\'")


def _read_rds_dataframe(path: Path) -> pd.DataFrame:
    """Convert an RDS file into a dataframe via ``Rscript``.

    :param path: Filesystem location of the RDS file.
    :return: Pandas dataframe containing the R object or an empty frame if conversion fails.
    """
    if not path.exists():
        return pd.DataFrame()
    tmp_file: Optional[Path] = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp_file = Path(tmp.name)
        tmp.close()
        path_str = _escape_r_string(path)
        tmp_str = _escape_r_string(tmp_file)
        r_code = (
            "options(warn=2);"
            f"d <- readRDS('{path_str}');"
            "if (!is.data.frame(d)) d <- as.data.frame(d);"
            f"write.csv(d, file='{tmp_str}', row.names=FALSE, na='')"
        )
        subprocess.run(
            ["Rscript", "-e", r_code],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        df = pd.read_csv(tmp_file, dtype=str)
        return df
    except subprocess.CalledProcessError as exc:
        log.warning("Rscript failed while loading %s: %s", path, exc)
        return pd.DataFrame()
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        log.warning("Failed to parse RDS %s: %s", path, exc)
        return pd.DataFrame()
    finally:
        if tmp_file and tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass


def _maybe_literal_eval(value: Any) -> Any:
    """Convert stringified lists/dicts/bools from R exports into native objects.

    :param value: Scalar value from the RDS export.
    :return: Parsed Python object when possible, otherwise the original value.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"na", "nan", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "inf":
        return float("inf")
    if lowered == "-inf":
        return float("-inf")
    if text.startswith("{") or text.startswith("["):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
    return text


def _load_shorts_sessions(data_root: Path) -> List[Dict[str, Any]]:
    """Load Shorts (Study 4) interaction logs from the YTRecs RDS export.

    :param data_root: Capsule ``data`` directory containing the ``shorts`` subfolder.
    :return: List of session dictionaries normalized to match the JSON schema.
    """
    rds_path = data_root / "shorts" / "ytrecs_sessions_may2024.rds"
    df = _read_rds_dataframe(rds_path)
    if df.empty:
        return []

    df = df.where(pd.notna(df), None)
    sessions: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        session: Dict[str, Any] = {}
        for col in df.columns:
            value = row[col]
            if value is None:
                session[col] = None
                continue
            if isinstance(value, str):
                parsed = _maybe_literal_eval(value)
                session[col] = parsed
            else:
                session[col] = value

        list_like_keys = [
            "vids",
            "ignoredVideos",
            "optionalInstructions",
            "pauseInteractions",
            "playInteractions",
            "ratingEvents",
            "ratingResults",
            "replayInteractions",
            "saveList",
            "seekBackwardInteractions",
            "seekForwardInteractions",
            "skipInteractions",
            "thumbInteractions",
        ]
        for key in list_like_keys:
            value = session.get(key)
            if isinstance(value, str):
                parsed = _maybe_literal_eval(value)
                session[key] = parsed if isinstance(parsed, list) else []
            elif value is None:
                session[key] = []
            elif not isinstance(value, list):
                session[key] = list(value) if value is not None else []

        dict_like_keys = [
            "displayOrders",
            "vidEndTimes",
            "vidStartTimes",
            "vidTotalLengths",
            "vidWatchTimes",
            "thumbStates",
        ]
        for key in dict_like_keys:
            value = session.get(key)
            if isinstance(value, str):
                parsed = _maybe_literal_eval(value)
                session[key] = parsed if isinstance(parsed, dict) else {}
            elif value is None:
                session[key] = {}

        vids_value = session.get("vids")
        if isinstance(vids_value, (list, tuple)):
            session["vids"] = [
                str(v).strip() for v in vids_value if isinstance(v, str) and v.strip()
            ]
        else:
            session["vids"] = []

        for key in ("endTime", "startTime"):
            value = session.get(key)
            if isinstance(value, str) and value.strip().isdigit():
                session[key] = int(value.strip())

        percent_visible = session.get("percentVisible")
        if isinstance(percent_visible, str):
            try:
                session["percentVisible"] = float(percent_visible)
            except ValueError:
                session["percentVisible"] = None

        session_finished = session.get("sessionFinished")
        if isinstance(session_finished, str):
            session["sessionFinished"] = session_finished.strip().lower() == "true"

        any_watched = session.get("anyVideoWatched")
        if isinstance(any_watched, str):
            session["anyVideoWatched"] = any_watched.strip().lower() == "true"

        sessions.append(session)

    return sessions


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

                        _fill_metadata_field(info, row, "title", "originTitle", "title", "video_title")
                        _fill_metadata_field(info, row, "channel_id", "originChannelId", "channel_id")
                        _fill_metadata_field(info, row, "channel_title", "originChannel", "channelTitle")
                        _fill_metadata_field(info, row, "description", "originDescription", "description")
                        _fill_metadata_field(info, row, "duration", "originDuration", "duration")
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
            except (OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
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


def _normalize_identifier(value: Any) -> str:
    """Normalize worker/case identifiers by trimming whitespace and dropping null tokens.

    :param value: Raw identifier from survey rows.
    :return: Cleaned identifier string or ``""`` when unset.
    """

    text = str(value or "").strip()
    if text and text.lower() not in {"nan", "none", "null"}:
        return text
    return ""


_MISSING_STRINGS = {"", "na", "nan", "none", "null", "n/a"}


def _is_missing_value(value: Any) -> bool:
    """Return ``True`` when ``value`` should be treated as a missing entry.

    :param value: Scalar extracted from survey or session data.
    :return: ``True`` when the value represents a missing token, ``False`` otherwise.
    """

    if value is None:
        return True
    if isinstance(value, float):
        if math.isnan(value):
            return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in _MISSING_STRINGS


def _parse_timestamp_ns(value: Any) -> Optional[int]:
    """Parse mixed-format timestamps into UTC nanoseconds.

    :param value: Timestamp stored as float, integer, or string.
    :return: Nanosecond-resolution epoch timestamp or ``None`` when parsing fails.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        ts_from_num = pd.to_datetime(value, unit="ms", errors="coerce", utc=True)
        if pd.notna(ts_from_num):
            return int(ts_from_num.value)
    text = str(value).strip()
    if not text or text.lower() in _MISSING_STRINGS:
        return None
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.notna(parsed):
        return int(parsed.value)
    try:
        num = float(text)
    except ValueError:
        return None
    ts_from_num = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
    if pd.notna(ts_from_num):
        return int(ts_from_num.value)
    return None


def _load_participant_allowlists(capsule_root: Path) -> Dict[str, Dict[str, Set[str]]]:
    """Reconstruct participant allow-lists used by the original R preprocessing.

    The filtering logic mirrors the CodeOcean scripts for each study and captures the
    identifiers (worker IDs or case IDs) that survived the attention checks and analysis
    filters.

    :param capsule_root: Path to the cloned CodeOcean capsule.
    :return: Nested dictionary containing per-study allow-lists (worker/case ids and urlids).
    """
    allowlists: Dict[str, Dict[str, Set[str]]] = {
        "gun_control": {"worker_ids": set(), "urlids": set()},
        "minimum_wage": {
            "study2_worker_ids": set(),
            "study2_urlids": set(),
            "study3_caseids": set(),
            "study4_worker_ids": set(),
            "study4_urlids": set(),
        },
    }

    def _normalize_series(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip()

    def _nonempty_mask(series: pd.Series) -> pd.Series:
        normalized = _normalize_series(series)
        lower = normalized.str.lower()
        return ~(normalized.eq("") | lower.isin(_MISSING_STRINGS))

    def _dedupe_earliest(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
        if df.empty or id_column not in df.columns:
            return df
        working = df.copy()
        sort_columns: List[str] = []
        if "start_time2" in working.columns:
            working["_sort_start_time2"] = pd.to_datetime(working["start_time2"], errors="coerce", utc=True)
            sort_columns.append("_sort_start_time2")
        if "start_time" in working.columns:
            numeric = pd.to_numeric(working["start_time"], errors="coerce")
            working["_sort_start_time"] = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
            sort_columns.append("_sort_start_time")
        if not sort_columns:
            sort_columns = [id_column]
        else:
            sort_columns.append(id_column)
        working = working.sort_values(by=sort_columns, kind="mergesort")
        deduped = working.drop_duplicates(subset=[id_column], keep="first")
        deduped = deduped.drop(columns=["_sort_start_time2", "_sort_start_time"], errors="ignore")
        return deduped

    # Gun control (Study 1)
    gun_dir = capsule_root / "results" / "intermediate data" / "gun control (issue 1)"
    gun_wave1 = _read_csv_if_exists(gun_dir / "guncontrol_qualtrics_w1_clean.csv")
    gun_w123 = _read_csv_if_exists(gun_dir / "guncontrol_qualtrics_w123_clean.csv")
    required_wave1_cols = {"worker_id", "q87", "q89", "survey_time", "gun_index"}
    required_followup_cols = {"worker_id", "treatment_arm", "pro", "anti"}
    if not gun_wave1.empty and not gun_w123.empty:
        if required_wave1_cols.issubset(gun_wave1.columns) and required_followup_cols.issubset(gun_w123.columns):
            wave1 = gun_wave1.copy()
            wave1["_worker_id"] = _normalize_series(wave1["worker_id"])
            mask = _nonempty_mask(wave1["_worker_id"])
            mask &= wave1["q87"].fillna("").astype(str).str.strip().eq("Quick and easy")
            mask &= wave1["q89"].fillna("").astype(str).str.strip().eq("wikiHow")
            times = pd.to_numeric(wave1["survey_time"], errors="coerce")
            mask &= times >= 120
            gun_index = pd.to_numeric(wave1["gun_index"], errors="coerce")
            mask &= gun_index.between(0.05, 0.95, inclusive="both")
            valid_wave1_workers = set(wave1.loc[mask, "_worker_id"])
            valid_wave1_workers.discard("")

            merged = gun_w123.copy()
            merged["_worker_id"] = _normalize_series(merged["worker_id"])
            if valid_wave1_workers:
                merged = merged[merged["_worker_id"].isin(valid_wave1_workers)]
            treatment_series = merged["treatment_arm"].fillna("").astype(str).str.strip().str.lower()
            merged = merged[
                (treatment_series != "control")
                & _nonempty_mask(merged["treatment_arm"])
            ]
            merged = merged[_nonempty_mask(merged["pro"])]
            merged = merged[_nonempty_mask(merged["anti"])]
            merged = _dedupe_earliest(merged, "_worker_id")
            gun_workers = {w for w in merged["_worker_id"] if w}
            allowlists["gun_control"]["worker_ids"] = gun_workers
            if "urlid" in merged.columns:
                gun_urlids = {_normalize_identifier(v) for v in merged["urlid"].tolist() if not _is_missing_value(v)}
            else:
                gun_urlids = set()
            allowlists["gun_control"]["urlids"] = gun_urlids
            log.info(
                "Allow-list (gun control): %d worker_ids (urlids=%d)",
                len(gun_workers),
                len(gun_urlids),
            )
        else:
            missing = required_wave1_cols.difference(gun_wave1.columns) | required_followup_cols.difference(gun_w123.columns)
            if missing:
                log.warning("Gun control allow-list skipped missing columns: %s", ", ".join(sorted(missing)))
    else:
        log.warning("Gun control allow-list: missing wave1 or merged dataset; skipping strict filters")

    wage_dir = capsule_root / "results" / "intermediate data" / "minimum wage (issue 2)"

    # Minimum wage Study 2 (MTurk)
    wage_mt = _read_csv_if_exists(wage_dir / "qualtrics_w12_clean.csv")
    required_mt_cols = {"worker_id", "q87", "q89", "survey_time", "mw_index_w1", "treatment_arm", "pro", "anti"}
    if not wage_mt.empty:
        if required_mt_cols.issubset(wage_mt.columns):
            mt_df = wage_mt.copy()
            mt_df["_worker_id"] = _normalize_series(mt_df["worker_id"])
            mask = _nonempty_mask(mt_df["_worker_id"])
            mask &= mt_df["q87"].fillna("").astype(str).str.strip().eq("Quick and easy")
            mask &= mt_df["q89"].fillna("").astype(str).str.strip().eq("wikiHow")
            mask &= pd.to_numeric(mt_df["survey_time"], errors="coerce") >= 120
            mw_index = pd.to_numeric(mt_df["mw_index_w1"], errors="coerce")
            mask &= mw_index.between(0.025, 0.975, inclusive="both")
            mt_df = mt_df.loc[mask]
            treatment_series = mt_df["treatment_arm"].fillna("").astype(str).str.strip().str.lower()
            mt_df = mt_df[
                (treatment_series != "control")
                & _nonempty_mask(mt_df["treatment_arm"])
            ]
            mt_df = mt_df[_nonempty_mask(mt_df["pro"])]
            mt_df = mt_df[_nonempty_mask(mt_df["anti"])]
            mt_df = _dedupe_earliest(mt_df, "_worker_id")
            study2_workers = {w for w in mt_df["_worker_id"] if w}
            allowlists["minimum_wage"]["study2_worker_ids"] = study2_workers
            study2_urlids = {
                _normalize_urlid(val)
                for val in mt_df.get("urlid", [])
                if isinstance(val, str) and _normalize_urlid(val)
            }
            allowlists["minimum_wage"]["study2_urlids"] = study2_urlids
            log.info("Allow-list (minimum wage Study 2): %d worker_ids", len(study2_workers))
        else:
            missing = required_mt_cols.difference(wage_mt.columns)
            log.warning("Minimum wage Study 2 allow-list skipped missing columns: %s", ", ".join(sorted(missing)))
    else:
        log.warning("Minimum wage Study 2 allow-list: dataset missing")

    # Minimum wage Study 3 (YouGov)
    wage_yg = _read_csv_if_exists(wage_dir / "yg_w12_clean.csv")
    caseid_col = "caseid" if "caseid" in wage_yg.columns else ("CaseID" if "CaseID" in wage_yg.columns else None)
    required_yg_cols = {"treatment_arm", "pro", "anti"}
    if not wage_yg.empty and caseid_col:
        if required_yg_cols.issubset(wage_yg.columns):
            yg_df = wage_yg.copy()
            yg_df["_caseid"] = _normalize_series(yg_df[caseid_col])
            yg_df = yg_df[_nonempty_mask(yg_df["_caseid"])]
            treatment_series = yg_df["treatment_arm"].fillna("").astype(str).str.strip().str.lower()
            yg_df = yg_df[
                (treatment_series != "control")
                & _nonempty_mask(yg_df["treatment_arm"])
            ]
            yg_df = yg_df[_nonempty_mask(yg_df["pro"])]
            yg_df = yg_df[_nonempty_mask(yg_df["anti"])]
            yg_df = _dedupe_earliest(yg_df, "_caseid")
            study3_caseids = {cid for cid in yg_df["_caseid"] if cid}
            allowlists["minimum_wage"]["study3_caseids"] = study3_caseids
            log.info("Allow-list (minimum wage Study 3): %d caseids", len(study3_caseids))
        else:
            missing = required_yg_cols.difference(wage_yg.columns)
            log.warning("Minimum wage Study 3 allow-list skipped missing columns: %s", ", ".join(sorted(missing)))
    else:
        if wage_yg.empty:
            log.warning("Minimum wage Study 3 allow-list: dataset missing")
        else:
            log.warning("Minimum wage Study 3 allow-list: missing caseid column")

    # Minimum wage Study 4 (Shorts)
    shorts_path = capsule_root / "results" / "intermediate data" / "shorts" / "qualtrics_w12_clean_ytrecs_may2024.csv"
    wage_shorts = _read_csv_if_exists(shorts_path)
    if not wage_shorts.empty:
        if "worker_id" in wage_shorts.columns:
            shorts_df = wage_shorts.copy()
            shorts_df["_worker_id"] = _normalize_series(shorts_df["worker_id"])
            mask = _nonempty_mask(shorts_df["_worker_id"])
            if "q81" in shorts_df.columns:
                mask &= shorts_df["q81"].fillna("").astype(str).str.strip().eq("Quick and easy")
            if "q82" in shorts_df.columns:
                mask &= shorts_df["q82"].fillna("").astype(str).str.strip().eq("wikiHow")
            if "video_link" in shorts_df.columns:
                mask &= _nonempty_mask(shorts_df["video_link"])
            shorts_df = shorts_df.loc[mask]
            shorts_df = _dedupe_earliest(shorts_df, "_worker_id")
            study4_workers = {w for w in shorts_df["_worker_id"] if w}
            allowlists["minimum_wage"]["study4_worker_ids"] = study4_workers
            study4_urlids = {
                _normalize_urlid(val)
                for val in shorts_df.get("urlid", [])
                if isinstance(val, str) and _normalize_urlid(val)
            }
            allowlists["minimum_wage"]["study4_urlids"] = study4_urlids
            log.info("Allow-list (minimum wage Study 4): %d worker_ids", len(study4_workers))
        else:
            log.warning("Minimum wage Study 4 allow-list: missing worker_id column")
    else:
        log.warning("Minimum wage Study 4 allow-list: dataset missing")

    return allowlists


def _participant_key(
    worker_id: str,
    case_id: str,
    anon_id: str,
    urlid: str,
    session_id: str,
    fallback_counter: int,
) -> Tuple[str, int]:
    """Choose the canonical participant identifier for deduplication.

    Preference order:
        1. MTurk/Shorts worker ids
        2. YouGov case ids
        3. Firebase anonymous ids
        4. URLIDs
        5. Session ids

    :param worker_id: Worker id from the survey export.
    :param case_id: YouGov case id (Study 3).
    :param anon_id: Firebase anonymous id present in session logs.
    :param urlid: Survey url identifier.
    :param session_id: Platform session id.
    :param fallback_counter: Counter used when minting synthetic identifiers.
    :return: Tuple of the chosen participant id and the updated fallback counter.
    """
    for candidate in (worker_id, case_id, anon_id, urlid, session_id):
        val = _normalize_identifier(candidate)
        if val:
            return val, fallback_counter
    return f"anon::{fallback_counter}", fallback_counter + 1


def _coerce_session_value(value: Any) -> Any:
    """Convert session log values to numeric scalars when possible.

    :param value: Raw value from session arrays or dictionaries.
    :return: Integer/float when coercion succeeds, otherwise the original value.
    """

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
    """Convert per-video session arrays/dicts into a standard lookup dict.

    :param values: Value taken from the session log (list or dict).
    :param raw_vids: Raw video identifiers as stored in the log.
    :param base_vids: Canonical 11-character video identifiers.
    :return: Mapping from video identifiers to coerced values.
    """
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
    """Retrieve a session metric for either the raw or canonical video id.

    :param mapping: Dictionary returned by :func:`_normalize_session_mapping`.
    :param raw_id: Raw video identifier from the log.
    :param base_id: Canonical 11-character video id.
    :return: Matching value or ``None`` if unavailable.
    """

    if not mapping:
        return None
    for key in (raw_id, base_id, str(raw_id), str(base_id)):
        if key and key in mapping:
            return mapping[key]
    return None


def _build_codeocean_rows(data_root: Path) -> pd.DataFrame:
    """Construct the full interaction dataframe from raw CodeOcean assets.

    :param data_root: Path pointing at the capsule ``data`` directory.
    :return: Pandas dataframe containing one row per usable recommendation decision.
    """

    sessions_path = data_root / "platform session data" / "sessions.json"
    with open(sessions_path, "r", encoding="utf-8") as fp:
        sessions = json.load(fp)

    shorts_sessions = _load_shorts_sessions(data_root)
    if shorts_sessions:
        sessions.extend(shorts_sessions)
        log.info("Loaded %d Shorts sessions from %s", len(shorts_sessions), data_root / "shorts" / "ytrecs_sessions_may2024.rds")

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

    shorts_survey = _read_survey_with_fallback(
        capsule_root / "intermediate data" / "shorts" / "qualtrics_w12_clean_ytrecs_may2024.csv",
        capsule_root / "results" / "intermediate data" / "shorts" / "qualtrics_w12_clean_ytrecs_may2024.csv",
    )
    if not shorts_survey.empty:
        survey_wage_sources.append(shorts_survey)

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
    allowlists = _load_participant_allowlists(capsule_root)
    gun_valid_workers: Set[str] = allowlists.get("gun_control", {}).get("worker_ids", set())
    wage_study2_workers: Set[str] = allowlists.get("minimum_wage", {}).get("study2_worker_ids", set())
    wage_study3_caseids: Set[str] = allowlists.get("minimum_wage", {}).get("study3_caseids", set())
    wage_study4_workers: Set[str] = allowlists.get("minimum_wage", {}).get("study4_worker_ids", set())
    wage_study2_urlids: Set[str] = allowlists.get("minimum_wage", {}).get("study2_urlids", set())
    wage_study4_urlids: Set[str] = allowlists.get("minimum_wage", {}).get("study4_urlids", set())
    enforce_gun_allowlist = bool(gun_valid_workers)
    enforce_wage_allowlist = bool(wage_study2_workers or wage_study3_caseids or wage_study4_workers)

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
    seen_participant_issue: set[Tuple[str, str]] = set()
    fallback_participant_counter = 0

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

        issue_value = str(issue or "").strip() or "unknown"

        anon_id = str(sess.get("anonymousFirebaseAuthUID") or "").strip()
        selected_survey_row: Dict[str, Any] = {}
        worker_id_value = ""
        case_id_value = ""
        participant_study_label = "unknown"
        candidate_entries: List[Tuple[int, str, str, str, str, Dict[str, Any]]] = []

        if survey_rows:
            for candidate_row in survey_rows:
                worker_candidate = _normalize_identifier(
                    candidate_row.get("worker_id")
                    or candidate_row.get("workerid")
                    or candidate_row.get("WorkerID")
                )
                case_candidate = _normalize_identifier(
                    candidate_row.get("caseid")
                    or candidate_row.get("CaseID")
                )
                start_ns = _parse_timestamp_ns(candidate_row.get("start_time2"))
                if start_ns is None:
                    start_ns = _parse_timestamp_ns(candidate_row.get("start_time"))
                if start_ns is None:
                    start_ns = _parse_timestamp_ns(candidate_row.get("start_time_w2"))
                if start_ns is None:
                    start_ns = int(1e20)

                study_label = "unknown"
                participant_token = ""
                valid = False

                if issue_value == "gun_control" and gun_valid_workers:
                    if worker_candidate and worker_candidate in gun_valid_workers:
                        study_label = "study1"
                        participant_token = worker_candidate
                        valid = True
                elif issue_value == "minimum_wage":
                    urlid_norm = urlid
                    if (
                        wage_study4_urlids
                        and urlid_norm
                        and urlid_norm in wage_study4_urlids
                        and worker_candidate
                        and worker_candidate in wage_study4_workers
                    ):
                        study_label = "study4"
                        participant_token = worker_candidate
                        valid = True
                    elif wage_study3_caseids and case_candidate and case_candidate in wage_study3_caseids:
                        study_label = "study3"
                        participant_token = case_candidate
                        valid = True
                    elif (
                        wage_study2_urlids
                        and urlid_norm
                        and urlid_norm in wage_study2_urlids
                        and worker_candidate
                        and worker_candidate in wage_study2_workers
                    ):
                        study_label = "study2"
                        participant_token = worker_candidate
                        valid = True
                    elif wage_study2_workers and worker_candidate and worker_candidate in wage_study2_workers:
                        study_label = "study2"
                        participant_token = worker_candidate
                        valid = True
                    elif wage_study4_workers and worker_candidate and worker_candidate in wage_study4_workers:
                        study_label = "study4"
                        participant_token = worker_candidate
                        valid = True

                if valid and study_label in {"study1", "study2", "study3"}:
                    treat_val = candidate_row.get("treatment_arm")
                    if _is_missing_value(treat_val) or str(treat_val).strip().lower() == "control":
                        valid = False
                    if _is_missing_value(candidate_row.get("pro")) or _is_missing_value(candidate_row.get("anti")):
                        valid = False

                if valid:
                    candidate_entries.append((start_ns, participant_token, worker_candidate, case_candidate, study_label, candidate_row))

        enforce_allowlist = False
        if issue_value == "gun_control" and enforce_gun_allowlist:
            enforce_allowlist = True
        elif issue_value == "minimum_wage" and enforce_wage_allowlist:
            enforce_allowlist = True

        if candidate_entries:
            candidate_entries.sort(key=lambda item: (item[0], item[1]))
            _, _, worker_id_value, case_id_value, participant_study_label, selected_survey_row = candidate_entries[0]
        elif survey_rows:
            selected_survey_row = _select_survey_row(survey_rows, topic or "")
            worker_id_value = _normalize_identifier(
                selected_survey_row.get("worker_id")
                or selected_survey_row.get("workerid")
                or selected_survey_row.get("WorkerID")
            )
            case_id_value = _normalize_identifier(
                selected_survey_row.get("caseid")
                or selected_survey_row.get("CaseID")
            )
            participant_study_label = _infer_participant_study(
                issue_value,
                selected_survey_row or {},
                topic,
                sess,
            )
        else:
            selected_survey_row = {}

        if enforce_allowlist and not candidate_entries:
            interaction_stats["sessions_filtered_allowlist"] += 1
            continue

        survey_row = selected_survey_row
        worker_id = worker_id_value
        case_id = case_id_value

        if participant_study_label == "study4":
            interaction_stats["sessions_skipped_study4"] += 1
            continue

        participant_identifier, fallback_participant_counter = _participant_key(
            worker_id,
            case_id,
            anon_id,
            urlid,
            session_id,
            fallback_participant_counter,
        )
        participant_issue_key = (participant_identifier, issue_value)
        if participant_issue_key in seen_participant_issue:
            interaction_stats["sessions_duplicate_participant_issue"] += 1
            continue
        seen_participant_issue.add(participant_issue_key)

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

            resolved_study = participant_study_label or "unknown"
            if not resolved_study or resolved_study == "unknown":
                resolved_study = _infer_participant_study(
                    issue_value,
                    survey_row or {},
                    topic,
                    sess,
                )
            row["participant_study"] = resolved_study
            row["participant_id"] = participant_identifier

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
        if not isinstance(it, dict):
            continue
        t = (it.get("title") or "").strip()
        v = (it.get("id") or "").strip()
        if t or v:
            out.append({"title": t, "id": v})
    return out

def _secs(value: Any) -> str:
    """Render a human-readable duration in seconds.

    :param value: Raw duration (string or numeric).
    :return: Duration formatted as ``"<n>s"`` or ``"?"`` when parsing fails.
    """

    try:
        return f"{int(round(float(value)))}s"
    except (TypeError, ValueError):
        return "?"

def _synthesize_viewer_sentence(ex: dict) -> str:
    bits: List[str] = []
    # age
    age = ex.get("age")
    try:
        age_i = int(age) if age not in (None, "", "nan") else None
    except (TypeError, ValueError):
        age_i = None
    if isinstance(age_i, int) and age_i > 0:
        bits.append(f"{age_i}-year-old")
    # gender (q26)
    gender = str(ex.get("q26") or "").strip().lower()
    if gender in {"man", "male"}:
        bits.append("man")
    elif gender in {"woman", "female"}:
        bits.append("woman")
    elif gender:
        bits.append(gender.title())
    # race (q29)
    race = str(ex.get("q29") or "").strip()
    if race and race.lower() != "nan":
        bits.append(race)
    # party/ideo
    pid1 = str(ex.get("pid1") or "").strip()
    ideo1 = str(ex.get("ideo1") or "").strip()
    if pid1 and pid1.lower() != "nan":
        if ideo1 and ideo1.lower() != "nan":
            bits.append(f"{pid1} {ideo1}".lower())
        else:
            bits.append(pid1)
    elif ideo1 and ideo1.lower() != "nan":
        bits.append(ideo1.lower())
    # income (q31)
    inc = str(ex.get("q31") or "").strip()
    if inc and inc.lower() != "nan":
        bits.append(inc)
    # education
    college = str(ex.get("college") or "").strip().lower()
    if college in {"true", "1", "yes", "y"}:
        bits.append("college-educated")
    # youtube frequency
    fy = str(ex.get("freq_youtube") or "").strip()
    fmap = {
        "0": "rarely",
        "1": "occasionally",
        "2": "a few times a month",
        "3": "weekly",
        "4": "several times a week",
        "5": "daily",
    }
    if fy in fmap:
        bits.append(f"watches YouTube {fmap[fy]}")
    s = ", ".join(b for b in bits if b)
    return s if s else "(no profile provided)"

def _viewer_attribute_lines(ex: dict) -> List[str]:
    """Return per-viewer attribute strings for the prompt."""

    details: List[str] = []
    race = _clean_str(ex.get("race") or ex.get("ethnicity") or ex.get("q29"))
    if race and not _is_nanlike(race):
        details.append(f"race/ethnicity: {race}")

    gun_own = _truthy_str_flag(ex.get("gun_own"))
    if gun_own is True:
        details.append("owns a gun")
    elif gun_own is False:
        details.append("does not own a gun")

    freq = _clean_str(ex.get("freq_youtube"))
    if freq in YOUTUBE_FREQ_MAP:
        details.append(f"YouTube frequency: {YOUTUBE_FREQ_MAP[freq]}")

    fav = _clean_str(ex.get("q8") or ex.get("fav_channels"))
    if fav and not _is_nanlike(fav):
        details.append(f"favorite channels: {fav}")

    pop = _clean_str(ex.get("q78"))
    if pop and not _is_nanlike(pop):
        details.append(f"popular channels followed: {pop}")

    return details


def _current_watch_lines(ex: dict, show_ids: bool) -> List[str]:
    """Return lines describing the currently watched video."""

    title = (ex.get("current_video_title") or "").strip()
    vid = (ex.get("current_video_id") or "").strip()
    if not (title or vid):
        return []
    heading = ["\nCURRENTLY WATCHING:"]
    if show_ids and vid:
        heading.append(f"{title or '(untitled)'} — id: {vid}")
    else:
        heading.append(f"{title or '(untitled)'}")
    return heading


def _history_lines(ex: dict, show_ids: bool, max_hist: int) -> List[str]:
    """Generate history lines (most recent first) for the prompt."""

    detailed = _as_list_json(ex.get("watched_detailed_json"))
    vids = _as_list_json(ex.get("watched_vids_json"))
    current_id = (ex.get("current_video_id") or "").strip()

    cur_idx: Optional[int] = None
    if current_id:
        cur_idx = _last_index(vids, current_id)
        if cur_idx is None and isinstance(detailed, list):
            for j in range(len(detailed) - 1, -1, -1):
                entry = detailed[j]
                if isinstance(entry, dict) and (entry.get("id") or "").strip() == current_id:
                    cur_idx = j
                    break
    if cur_idx is None and isinstance(vids, list) and vids:
        cur_idx = len(vids) - 1

    prior_entries: List[Dict[str, Any]] = []
    if isinstance(detailed, list) and cur_idx is not None and cur_idx > 0:
        prior_entries = detailed[:cur_idx]

    if not prior_entries:
        return []

    heading = ["\nHISTORY (most recent first):"]
    limit = max_hist if max_hist and max_hist > 0 else len(prior_entries)
    for entry in reversed(prior_entries[-limit:]):
        name = (
            entry.get("title")
            or (entry.get("id") if show_ids else "")
            or "(untitled)"
        ).strip()
        watch_time = _secs(entry.get("watch_seconds"))
        total_length = _secs(entry.get("total_length"))
        heading.append(f"- [{watch_time}/{total_length}] {name}")
    return heading


def _build_user_prompt_from_columns(ex: dict, max_hist: int = 12) -> str:
    show_ids = os.getenv("GRAIL_SHOW_IDS", "0") == "1"
    lines: List[str] = ["PROFILE:"]

    viewer = (ex.get("viewer_profile_sentence") or "").strip()
    if not viewer:
        viewer = _synthesize_viewer_sentence(ex)
    lines.append(viewer)

    details = _viewer_attribute_lines(ex)
    if details:
        lines.append("\nATTRIBUTES:")
        lines.extend(f"- {detail}" for detail in details)

    lines.extend(_current_watch_lines(ex, show_ids))
    lines.extend(_history_lines(ex, show_ids, max_hist))

    items = _load_slate_items(ex)
    lines.append("\nOPTIONS:")
    if items:
        for idx, item in enumerate(items, 1):
            name = (item.get("title") or (item.get("id") if show_ids else "") or "(untitled)").strip()
            lines.append(f"{idx}. {name}")
    else:
        lines.append("(no options provided)")

    return "\n".join(lines)

# ---------- “full history” & “prior slates” for discriminator ----------
def _render_full_history_lines_disc(ex: dict, include_current: bool = False) -> list[str]:
    """Render full viewing history lines for the discriminator state.

    :param ex: Row dictionary representing the current interaction.
    :param include_current: Whether to include the current video in the history output.
    :return: List of history lines formatted for the discriminator prompt.
    """

    tj = ex.get("trajectory_json")
    try:
        obj = json.loads(tj) if isinstance(tj, str) and tj.strip() else {}
    except (TypeError, json.JSONDecodeError):
        obj = {}
    order = obj.get("order") if isinstance(obj, dict) else None
    if not isinstance(order, list):
        return []

    def _key(row: dict) -> tuple[int, float]:
        try:
            return (0, int(row.get("idx")))
        except (TypeError, ValueError):
            try:
                return (1, float(row.get("end_ms") or -1))
            except (TypeError, ValueError):
                return (1, -1.0)

    seq = [row for row in order if isinstance(row, dict)]
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
    """Summarize prior recommendation slates from the trajectory payload.

    :param ex: Row dictionary with ``trajectory_json`` attached.
    :return: List of strings describing earlier slates.
    """

    tj = ex.get("trajectory_json")
    try:
        obj = json.loads(tj) if isinstance(tj, str) and tj.strip() else {}
    except (TypeError, json.JSONDecodeError):
        obj = {}
    disp = obj.get("displayOrders") if isinstance(obj, dict) else None
    if not isinstance(disp, dict):
        return []
    out = []
    matching_keys = [
        key
        for key in disp.keys()
        if re.match(r"^\s*(\d+)\s*[-_ ]*recs\s*$", str(key), re.I)
    ]
    keys = sorted(
        matching_keys,
        key=lambda key: int(re.search(r"(\d+)", str(key)).group(1)),
    )
    for key in keys:
        val = disp.get(key) or []
        names = []
        if isinstance(val, list):
            for el in val:
                if isinstance(el, dict):
                    names.append(el.get("title") or el.get("id") or "(untitled)")
                else:
                    names.append(str(el))
        elif isinstance(val, dict):
            names = [str(x) for x in val.keys()]
        out.append(f"{key}: " + "; ".join(names[:10]))
    return out

def _build_state_disc_text(ex: dict) -> str:
    """Build the discriminator state text from a cleaned example row.

    :param ex: Example dictionary containing metadata and trajectory info.
    :return: Multiline string with current video, history, and prior slates.
    """

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
    """Infer the next video id from the watch history when explicit labels are missing.

    :param ex: Session row dictionary.
    :param current_id: Canonical id of the current video.
    :return: Next video id or an empty string when cannot be determined.
    """

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
    """Resolve the gold next-video id for a session step.

    :param ex: Session row being transformed.
    :param sol_key: Optional alternate column name containing the gold id.
    :return: Canonical next-video id or an empty string when unavailable.
    """
    cur = (ex.get("current_video_id") or "").strip()
    if sol_key and sol_key not in {"current_video_id", "current_id"}:
        v = ex.get(sol_key)
        if isinstance(v, str) and v.strip() and v.strip() != cur:
            return v.strip()
    candidate_fields = ("next_video_id", "clicked_id", "label", "answer")
    for field in candidate_fields:
        value = ex.get(field)
        if isinstance(value, str) and value.strip() and value.strip() != cur:
            return value.strip()
    return _derive_next_from_history(ex, cur)

def _gold_index_from_items(gold: str, items: List[dict]) -> int:
    """Locate the 1-based index of ``gold`` inside the slate items list.

    :param gold: Gold video id (canonical string).
    :param items: Slate items pulled from the session log.
    :return: Index in ``items`` or ``-1`` when the id cannot be matched.
    """

    gold = (gold or "").strip()
    if not gold or not items:
        return -1
    for i, it in enumerate(items, 1):
        if gold == (it.get("id") or ""):
            return i
    gc = _canon(gold)
    if gc:
        for i, it in enumerate(items, 1):
            if gc == _canon(it.get("title", "")):
                return i
    return -1


PASSTHROUGH_COLUMNS: Set[str] = {
    "issue",
    "rating_copy",
    "rating_index",
    "rating_video_id",
    "urlid",
    "topic_id",
    "session_id",
    "step_index",
    "display_step",
    "display_order_key",
    "current_video_raw_id",
    "current_video_channel",
    "current_video_channel_id",
    "next_video_id",
    "next_video_raw_id",
    "next_video_title",
    "next_video_channel",
    "next_video_channel_id",
    "percent_visible",
    "session_finished",
    "start_time_ms",
    "end_time_ms",
    "trajectory_json",
    "issue_source",
    "issue_detail",
    "slate_source",
}

# ---------- row → clean example ----------
def _row_to_example(ex: dict, sys_prompt: Optional[str], sol_key: Optional[str], max_hist: int) -> Optional[dict]:
    """Transform a raw session pair into the cleaned GRPO example structure.

    :param ex: Interaction row produced during session processing.
    :param sys_prompt: Optional system prompt override.
    :param sol_key: Alternate column to treat as the gold next-video id.
    :param max_hist: Maximum number of prior history entries to render.
    :return: Cleaned example dictionary or ``None`` when the row is unusable.
    """

    items = _load_slate_items(ex)
    if not items:
        return None
    gold_id = _get_gold_next_id(ex, sol_key)
    gidx = _gold_index_from_items(gold_id, items)
    if gidx < 1:
        return None

    user_msg = _build_user_prompt_from_columns(ex, max_hist=max_hist)
    sys_msg = sys_prompt or (
        "You are choosing EXACTLY ONE item from a short slate for a specific viewer.\n"
        "Think briefly in <think>…</think>, then output ONLY the option NUMBER (1..N) inside <answer>…</answer>.\n"
        "Format (STRICT): <think>…</think><answer>3</answer>"
    )

    slate_text = ex.get("slate_text")
    if not slate_text:
        slate_text = "\n".join(
            f"{idx}. {(item.get('title') or item.get('id') or '(untitled)').strip()}"
            for idx, item in enumerate(items, 1)
        )

    out = {
        "prompt": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        "answer": str(gidx),  # GOLD index as string
        "gold_index": gidx,  # int
        "gold_id": gold_id,
        "n_options": int(ex.get("n_options") or len(items) or 0),
        "viewer_profile": str(
            ex.get("viewer_profile_sentence") or _synthesize_viewer_sentence(ex)
        ),
        "state_text": user_msg,  # LM sees this
        "state_disc_text": _build_state_disc_text(ex),  # disc sees richer info
        "slate_items": items,
        "slate_text": str(slate_text),
        # passthrough
        "watched_detailed_json": _as_list_json(ex.get("watched_detailed_json")),
        "watched_vids_json": _as_list_json(ex.get("watched_vids_json")),
        "current_video_id": str(ex.get("current_video_id") or ""),
        "current_video_title": str(ex.get("current_video_title") or ""),
        "task": "GRAIL",
        "is_replay": False,
        "accuracy": 0.0,
        "mix_group_id": -1,
        "mix_copy_idx": -1,
    }
    out["slate_items_with_meta"] = _as_list_json(ex.get("slate_items_json"))

    for extra in PASSTHROUGH_COLUMNS:
        if extra in ex:
            out[extra] = ex.get(extra)

    for key, val in ex.items():
        if key in out:
            continue
        cleaned = val
        try:
            if pd.isna(cleaned):  # type: ignore[arg-type]
                cleaned = None
        except (TypeError, ValueError):
            pass
        out[key] = cleaned
    return out


def _ensure_shared_schema(datasets_map: Dict[str, datasets.Dataset]) -> Dict[str, datasets.Dataset]:
    """Ensure every dataset split exposes the same columns and dtypes.

    :param datasets_map: Mapping of split name to ``datasets.Dataset``.
    :return: New mapping where each split has identical columns/features.
    """
    if not datasets_map:
        return datasets_map
    all_columns: Set[str] = set()
    feature_template: Dict[str, Any] = {}
    for split_ds in datasets_map.values():
        features = split_ds.features
        for name, feature in features.items():
            all_columns.add(name)
            feature_template.setdefault(name, feature)
    # default filler per feature type
    def _default_for_feature(feature: Any, length: int) -> List[Any]:
        if isinstance(feature, Sequence):
            inner = feature.feature if hasattr(feature, "feature") else None
            if inner and isinstance(inner, Value) and inner.dtype == "string":
                return [[] for _ in range(length)]
            return [[] for _ in range(length)]
        return [None] * length

    merged_features = Features(feature_template)
    aligned: Dict[str, datasets.Dataset] = {}
    for split_name, split_ds in datasets_map.items():
        missing = [col for col in all_columns if col not in split_ds.column_names]
        if missing:
            for col in missing:
                feature = feature_template.get(col)
                filler = _default_for_feature(feature, len(split_ds))
                split_ds = split_ds.add_column(col, filler)
        split_ds = split_ds.cast(merged_features)
        aligned[split_name] = split_ds
    return aligned

# ---------- driver ----------
def _load_codeocean_dataset(dataset_name: str, validation_ratio: float = 0.1) -> DatasetDict:
    """Load the raw CodeOcean capsule and convert it into a dataset dictionary.

    :param dataset_name: Path to the capsule ``data`` directory.
    :param validation_ratio: Fraction of participants to allocate to the validation split.
    :return: ``DatasetDict`` with ``train``/``validation`` splits populated.
    """

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
    """Load the raw dataset from disk or Hugging Face Hub.

    :param dataset_name: HF repo id, ``load_from_disk`` folder, file path, or capsule directory.
    :param validation_ratio: Fraction used when splitting CodeOcean data.
    :return: ``DatasetDict`` containing the available splits.
    """
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
        if isinstance(ds, DatasetDict):
            return ds
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
    """Command-line entry point for building the cleaned dataset and reports."""

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
    def _ok(ex: dict) -> bool:
        items = _load_slate_items(ex)
        if not items:
            return False
        gold = _get_gold_next_id(ex, sol_key)
        if not gold:
            return False
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

    desired = _ensure_shared_schema(desired)
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
