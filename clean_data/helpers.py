#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common low-level helpers reused across the cleaning pipeline.

This module holds normalization routines, regex utilities, and shared
constants that power session parsing, prompt construction, and survey
processing. Anything that manipulates identifiers, serialised JSON, or
core feature definitions typically lives here to avoid circular imports
between the higher-level modules. All routines are made available under the
repository's Apache 2.0 license; review the LICENSE file for full terms.
"""

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
            parsed_value = json.loads(value or default)
            return parsed_value if isinstance(parsed_value, list) else []
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
    match = YTID_RE.search(vid)
    return match.group(1) if match else vid


def _normalize_urlid(value: Any) -> str:
    """Standardize URL identifiers for dictionary lookups.

    :param value: Incoming identifier that may be numeric, string-like, or ``None``.
    :returns: Normalized identifier string suitable for dict keys.
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


def _normalize_identifier(value: Any) -> str:
    """Normalize worker/case identifiers by trimming whitespace and dropping null tokens.

    :param value: Identifier value from the raw dataset.
    :returns: Sanitized identifier string or ``""`` when the value is missing.
    """
    text = str(value or "").strip()
    if text and text.lower() not in {"nan", "none", "null"}:
        return text
    return ""


def _coerce_session_value(value: Any) -> Any:
    """Convert session log values to numeric scalars when possible.

    :param value: Raw value from the session logs.
    :returns: An ``int``/``float`` when conversion succeeds, the stripped string, or
        the original value for unhandled types.
    """
    if isinstance(value, (int, float)):
        return value

    result: Any = value
    if isinstance(value, str):
        value_str = value.strip()
        if not value_str:
            result = None
        else:
            try:
                if "." in value_str:
                    num = float(value_str)
                    result = int(num) if num.is_integer() else num
                else:
                    result = int(value_str)
            except ValueError:
                result = value_str
    return result


_MISSING_STRINGS = {"", "na", "nan", "none", "null", "n/a"}


def _is_missing_value(value: Any) -> bool:
    """Return ``True`` when ``value`` should be treated as a missing entry.

    :param value: Value pulled from the dataset.
    :returns: ``True`` when the value is blank, null-like, or ``NaN``.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in _MISSING_STRINGS


def _parse_timestamp_ns(value: Any) -> Optional[int]:
    """Parse mixed-format timestamps into UTC nanoseconds.

    :param value: Timestamp encoded as number, string, or ``None``.
    :returns: Integer nanoseconds since epoch or ``None`` when parsing fails.
    """
    if value is None:
        return None

    result: Optional[int] = None
    is_nan_float = isinstance(value, float) and math.isnan(value)
    if isinstance(value, (int, float)) and not is_nan_float:
        ts_from_num = pd.to_datetime(value, unit="ms", errors="coerce", utc=True)
        if pd.notna(ts_from_num):
            result = int(ts_from_num.value)
    text = str(value).strip()
    if result is None and text and text.lower() not in _MISSING_STRINGS:
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
        if pd.notna(parsed):
            result = int(parsed.value)
        else:
            try:
                num = float(text)
            except ValueError:
                num = None
            if num is not None:
                ts_from_num = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
                if pd.notna(ts_from_num):
                    result = int(ts_from_num.value)
    return result
