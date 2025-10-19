"""Helper utilities shared across the clean_data package."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional

import pandas as pd

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
        import pyarrow as pa  # type: ignore  # pylint: disable=import-outside-toplevel
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


def _normalize_urlid(value: Any) -> str:
    """Standardize URL identifiers for dictionary lookups."""
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


def _normalize_identifier(value: Any) -> str:
    """Normalize worker/case identifiers by trimming whitespace and dropping null tokens."""
    text = str(value or "").strip()
    if text and text.lower() not in {"nan", "none", "null"}:
        return text
    return ""


def _coerce_session_value(value: Any) -> Any:
    """Convert session log values to numeric scalars when possible."""
    # pylint: disable=too-many-return-statements
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


_MISSING_STRINGS = {"", "na", "nan", "none", "null", "n/a"}


def _is_missing_value(value: Any) -> bool:
    """Return ``True`` when ``value`` should be treated as a missing entry."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in _MISSING_STRINGS


def _parse_timestamp_ns(value: Any) -> Optional[int]:
    """Parse mixed-format timestamps into UTC nanoseconds."""
    # pylint: disable=too-many-return-statements
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
