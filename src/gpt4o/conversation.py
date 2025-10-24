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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

# pylint: disable=too-many-return-statements,too-many-branches,too-many-locals
# pylint: disable=too-many-statements,too-many-nested-blocks,broad-exception-caught

from __future__ import annotations

import json
import logging
import os
import re
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

from common.prompt_docs import (
    PromptDocumentBuilder,
    default_title_resolver,
    load_trajectory_entries,
)
from common.prompt_fields import (
    NOW_PLAYING_ID_KEYS,
    NOW_PLAYING_TITLE_KEYS,
    NOW_PLAYING_TITLE_KEYS_WITH_META,
)
from .config import PROMPT_COLUMN, SOLUTION_COLUMN, SYSTEM_PROMPT
from .utils import canon_text, canon_video_id, is_nan_like, truthy

_PROMPT_CONSTANTS = import_module("prompt_builder.constants")
YT_FREQ_MAP = _PROMPT_CONSTANTS.YT_FREQ_MAP

_PROMPT_DOC_BUILDER = PromptDocumentBuilder(
    prompt_column=PROMPT_COLUMN,
    solution_column=SOLUTION_COLUMN,
    max_history=int(os.environ.get("GRAIL_MAX_HISTORY", "8")),
    title_lookup=default_title_resolver(),
    log_prefix="[GPT4O]",
    logger=logging.getLogger("gpt4o.prompts"),
)

_INCOME_HINT_KEYS = ["income", "income_bracket", "q30", "q31", "q32", "q34"]
_INCOME_PATTERN = re.compile(r"\$\s?\d{1,3}(?:,\d{3})?(?:\s*-\s*\$\s?\d{1,3}(?:,\d{3})?)?")

_MARITAL_KEYS = ["marital", "marital_status", "q18", "q56", "q77"]
_RACE_TEXT_FIELDS = ["q26", "q27", "q28", "q26_3_text", "q33_14_text", "race", "ethnicity"]
_RACE_MAP = {
    "white": "White",
    "caucasian": "White",
    "caucasian/white": "White",
    "black": "Black",
    "africanamerican": "Black",
    "african american": "Black",
    "asian": "Asian",
    "hispanic": "Hispanic/Latino",
    "latino": "Hispanic/Latino",
    "latina": "Hispanic/Latino",
    "latinx": "Hispanic/Latino",
    "nativeamerican": "Native American",
    "americanindian": "Native American",
    "pacificislander": "Pacific Islander",
    "middleeastern": "Middle Eastern",
    "other": "Other",
    "mixed": "Multiracial",
    "twoormore": "Multiracial",
    "multiracial": "Multiracial",
    "prefernottoanswer": "Unspecified",
    "unknown": "Unspecified",
}


def pick_case_insensitive(record: dict, *candidates: str) -> Optional[str]:
    """Return the first present key (case-insensitive) value from the record."""

    if not isinstance(record, dict):
        return None
    lower_keys = {key.lower(): key for key in record.keys()}
    for candidate in candidates:
        if candidate.lower() in lower_keys:
            value = record.get(lower_keys[candidate.lower()])
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def extract_income(example: dict) -> Optional[str]:
    """Mirror the income extraction used in the GRPO baseline."""

    for key in _INCOME_HINT_KEYS:
        value = example.get(key)
        if isinstance(value, str) and not is_nan_like(value):
            stripped = value.strip()
            if _INCOME_PATTERN.search(stripped) or "income" in stripped.lower():
                return stripped
    try:
        for key, value in example.items():
            if isinstance(value, str) and not is_nan_like(value) and _INCOME_PATTERN.search(value):
                return value.strip()
    except Exception:
        pass
    if "income_gt50k" in example:
        high_income = bool(example.get("income_gt50k"))
        return ">$50k household income" if high_income else "≤$50k household income"
    return None


def extract_party(example: dict) -> Optional[str]:
    """Return a party label derived from standard survey signals."""

    pid_text = example.get("pid")
    if isinstance(pid_text, str) and not is_nan_like(pid_text):
        lowered = pid_text.strip().lower()
        if "dem" in lowered:
            return "Democratic"
        if "rep" in lowered or "gop" in lowered:
            return "Republican"
        if "independent" in lowered:
            return "Independent"
        if "libertarian" in lowered:
            return "Libertarian"
        if "green" in lowered:
            return "Green"
        if "closer to the democratic" in lowered:
            return "Democratic-leaning"
        if "closer to the republican" in lowered:
            return "Republican-leaning"
        return pid_text.strip()

    values: list[float] = []
    for key in ("pid1", "pid2", "pid3", "pid4"):
        try:
            value = example.get(key)
            if value is not None and str(value).strip() != "":
                values.append(float(value))
        except Exception:
            continue
    if values:
        mean_value = sum(values) / len(values)
        if mean_value <= 3.0:
            return "Democratic-leaning"
        if mean_value >= 5.0:
            return "Republican-leaning"
        return "Independent/Other"
    return None


def format_ideology(raw: Any) -> Optional[str]:
    """Convert ideology encodings to descriptive text."""

    if raw is None:
        return None
    value = str(raw).strip()
    if is_nan_like(value):
        return None
    try:
        numeric = float(value)
        if numeric <= 1.5:
            return "Extremely liberal"
        if numeric <= 2.5:
            return "Liberal"
        if numeric <= 3.5:
            return "Slightly liberal"
        if numeric <= 4.5:
            return "Moderate"
        if numeric <= 5.5:
            return "Slightly conservative"
        if numeric <= 6.5:
            return "Conservative"
        return "Extremely conservative"
    except Exception:
        pass
    lowered = value.lower()
    if "extreme" in lowered and "lib" in lowered:
        return "Extremely liberal"
    if "lib" in lowered:
        return "Liberal"
    if "moderate" in lowered or "centrist" in lowered:
        return "Moderate"
    if "conservative" in lowered and "extreme" in lowered:
        return "Extremely conservative"
    if "conservative" in lowered:
        return "Conservative"
    return value


def extract_marital_status(example: dict) -> Optional[str]:
    """Extract a brief marital descriptor."""

    for key in _MARITAL_KEYS:
        value = example.get(key)
        if isinstance(value, str) and not is_nan_like(value):
            lowered = value.strip().lower()
            if "married" in lowered:
                return "Married"
            if "partner" in lowered or "cohabit" in lowered:
                return "Living with partner"
            if "single" in lowered:
                return "Single"
            if "divorced" in lowered:
                return "Divorced"
            if "widow" in lowered:
                return "Widowed"
            if "separated" in lowered:
                return "Separated"
            return value
    return None


def _normalise_race_token(raw: str) -> Optional[str]:
    """Normalise a free-form race token into a canonical label.

    :param raw: Raw race description extracted from the example.
    :returns: Canonical race label, the stripped token, or ``None`` when empty.
    """
    if not raw:
        return None
    key = canon_text(raw)
    if key in _RACE_MAP:
        return _RACE_MAP[key]
    for candidate, target in _RACE_MAP.items():
        if candidate in key:
            return target
    return raw.strip()


def extract_race(example: dict) -> Optional[str]:
    """Extract race information following the GRPO baseline rules."""

    try:
        if truthy(example.get("white")) and truthy(example.get("black")):
            return "Multiracial"
        if truthy(example.get("white")):
            return "White"
        if truthy(example.get("black")):
            return "Black"
    except Exception:
        pass

    for field in _RACE_TEXT_FIELDS:
        value = example.get(field)
        if isinstance(value, str) and not is_nan_like(value):
            normalised = _normalise_race_token(value)
            if normalised and not is_nan_like(normalised):
                return normalised
    return None


def humanise_profile(example: dict) -> str:
    """Build the human-readable profile sentence."""

    fragments: list[str] = []
    age = example.get("age")
    if isinstance(age, (int, float)) and age > 0:
        fragments.append(f"{int(age)}-year-old")

    race = extract_race(example)
    female = truthy(example.get("female"))
    male = truthy(example.get("male"))
    gender = "woman" if (female and not male) else ("man" if (male and not female) else None)

    if race and gender:
        fragments.append(f"{race} {gender}")
    else:
        if gender:
            fragments.append(gender)
        if race and not gender:
            fragments.append(race)

    marital = extract_marital_status(example)
    if marital and not is_nan_like(marital):
        if marital in {"Married", "Single", "Divorced", "Widowed", "Separated"}:
            fragments.append(marital.lower())
        else:
            fragments.append(marital)

    party = extract_party(example)
    ideology = format_ideology(example.get("ideo"))
    if party and ideology:
        fragments.append(f"{party.lower()} {ideology.lower()}")
    elif party:
        fragments.append(party)
    elif ideology:
        fragments.append(ideology.lower())

    income = extract_income(example)
    if income and not is_nan_like(income):
        fragments.append(income)

    if truthy(example.get("college")) and not any(
        "degree" in (income or "").lower() for _ in [0]
    ):
        fragments.append("college-educated")

    youtube_freq = str(example.get("freq_youtube", "")).strip()
    if youtube_freq in YT_FREQ_MAP:
        viewing_phrase = YT_FREQ_MAP.get(youtube_freq, "regularly")
        fragments.append(f"watches YouTube {viewing_phrase}")

    sentence_parts = [
        fragment for fragment in fragments if fragment and not is_nan_like(fragment)
    ]
    sentence = ", ".join(sentence_parts)
    return sentence if sentence else "(no profile provided)"


def build_profile_block(example: dict) -> str:
    """Structured key/value profile block (parity with GRPO)."""

    lines: list[str] = []
    race = extract_race(example)
    marital = extract_marital_status(example)
    party = extract_party(example)
    ideology = format_ideology(example.get("ideo"))
    income = extract_income(example)
    if race:
        lines.append(f"race: {race}")
    if marital:
        lines.append(f"marital: {marital}")
    if party:
        lines.append(f"party: {party}")
    if ideology:
        lines.append(f"ideology: {ideology}")
    if income:
        lines.append(f"income: {income}")

    default_cols = [
        "age",
        "gender",
        "female",
        "male",
        "college",
        "income_gt50k",
        "pid",
        "ideo",
        "pol_interest",
        "freq_youtube",
        "fav_channels",
        "popular_channels",
        "gun_index",
        "gun_index_2",
        "minwage_text_r_w1",
    ]
    columns = os.environ.get("GRAIL_PROFILE_COLS", ",".join(default_cols))
    desired = [col.strip() for col in columns.split(",") if col.strip()]
    for column in desired:
        if column in example and example[column] is not None:
            value = str(example[column]).strip()
            if not is_nan_like(value):
                lines.append(f"{column}: {value}")
    if len(lines) > 24:
        lines = lines[:24] + ["..."]
    return "\n".join(lines) if lines else "(none)"


def _extract_now_watching(example: dict) -> Tuple[str, str] | None:
    """Derive the currently watched title/id pair from the example payload.

    :param example: Dataset row containing now-watching metadata fields.
    :returns: ``(title, video_id)`` tuple or ``None`` when unavailable.
    """
    video_id = pick_case_insensitive(example, "video_id", "videoId")
    if video_id and not is_nan_like(video_id):
        title = (
            pick_case_insensitive(example, *NOW_PLAYING_TITLE_KEYS_WITH_META)
            or _PROMPT_DOC_BUILDER.title_for(video_id)
            or ""
        )
        return (title or "(untitled)"), video_id
    title = pick_case_insensitive(example, *NOW_PLAYING_TITLE_KEYS)
    video_id = pick_case_insensitive(example, *NOW_PLAYING_ID_KEYS)
    if (title and not is_nan_like(title)) or (video_id and not is_nan_like(video_id)):
        if is_nan_like(title) and video_id:
            title = _PROMPT_DOC_BUILDER.title_for(video_id) or ""
        return (title or "(untitled)"), (video_id or "")

    trajectory_json = example.get("trajectory_json")
    if isinstance(trajectory_json, str) and trajectory_json.strip():
        try:
            data = json.loads(trajectory_json)
            for key in (
                "current",
                "now",
                "active",
                "playing",
                "nowPlaying",
                "currentVideo",
                "watching",
            ):
                current = data.get(key)
                if isinstance(current, dict):
                    title = pick_case_insensitive(
                        current,
                        "title",
                        "video_title",
                        "name",
                        "videoTitle",
                    )
                    video_id = pick_case_insensitive(
                        current,
                        "video_id",
                        "vid",
                        "id",
                        "videoId",
                        "originId",
                        "origin_id",
                    )
                    if video_id and not title:
                        title = _PROMPT_DOC_BUILDER.title_for(video_id) or ""
                    if title or video_id:
                        return (title or "(untitled)"), (video_id or "")
        except Exception:
            pass
    return None


def _get_history_pointer(example: dict) -> Tuple[int | None, float | None]:
    """Extract the trajectory pointer describing the current history position.

    :param example: Dataset row containing the ``trajectory_json`` blob.
    :returns: Tuple of ``(current_index, current_end_ms)`` values.
    """
    trajectory_json = example.get("trajectory_json")
    if not isinstance(trajectory_json, str) or not trajectory_json.strip():
        return None, None
    try:
        data = json.loads(trajectory_json)
    except Exception:
        return None, None
    pointer = data.get("current_index")
    if pointer is None:
        pointer = data.get("currentIdx")
    end_ms = data.get("current_end_ms")
    if end_ms is None:
        end_ms = data.get("currentEndMs")
    try:
        pointer_int = int(pointer) if pointer is not None else None
    except Exception:
        pointer_int = None
    try:
        end_float = float(end_ms) if end_ms is not None else None
    except Exception:
        end_float = None
    return pointer_int, end_float


def _extract_history(
    example: dict,
    *,
    up_to_idx: int | None = None,
    up_to_end_ms: float | None = None,
    include_current: bool = False,
) -> list[dict]:
    """Parse interaction history entries from the trajectory payload.

    :param example: Dataset row containing the ``trajectory_json`` field.
    :param up_to_idx: Optional index cap for the returned rows.
    :param up_to_end_ms: Optional end-time cap for the returned rows.
    :param include_current: Whether to retain the current item at the caps.
    :returns: List of canonicalised history dictionaries.
    """
    trajectory_json = example.get("trajectory_json")
    if not isinstance(trajectory_json, str) or not trajectory_json.strip():
        return []
    try:
        data = json.loads(trajectory_json)
    except Exception:
        return []
    rows = data.get("order") or data.get("videos") or data.get("history")
    if not isinstance(rows, list):
        return []
    extracted: list[dict] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        row_copy = dict(row)
        row_copy.setdefault("idx", row_copy.get("index", row_copy.get("idx", idx)))
        row_copy.setdefault(
            "title",
            pick_case_insensitive(
                row_copy,
                "title",
                "video_title",
                "name",
                "videoTitle",
            ),
        )
        row_copy.setdefault(
            "video_id",
            pick_case_insensitive(
                row_copy,
                "video_id",
                "id",
                "videoId",
                "originId",
                "origin_id",
            ),
        )
        row_copy.setdefault(
            "watch_seconds",
            row_copy.get("watch_seconds")
            or row_copy.get("watch_time")
            or row_copy.get("secondsWatched"),
        )
        row_copy.setdefault(
            "total_length",
            row_copy.get("total_length")
            or row_copy.get("total")
            or row_copy.get("videoLength"),
        )
        extracted.append(row_copy)

    def _hist_key(entry: dict) -> tuple[int, float]:
        """Provide a stable sort key for partially-specified history rows.

        :param entry: Raw history row to normalise for sorting.
        :returns: Tuple distinguishing indexed items from timing fallbacks.
        """
        idx_val = entry.get("idx")
        if isinstance(idx_val, int):
            return (0, idx_val)
        try:
            end = float(entry.get("end_ms") or -1)
        except Exception:
            end = -1.0
        return (1, end)

    extracted.sort(key=_hist_key)
    if up_to_idx is not None:
        extracted = [
            row
            for row in extracted
            if (row.get("idx") is None)
            or (row["idx"] < up_to_idx or (include_current and row["idx"] == up_to_idx))
        ]
    elif up_to_end_ms is not None:
        try:
            threshold = float(up_to_end_ms)
            extracted = [
                row
                for row in extracted
                if (row.get("end_ms") is None)
                or (
                    float(row["end_ms"]) < threshold
                    or (include_current and float(row["end_ms"]) <= threshold)
                )
            ]
        except Exception:
            pass
    return extracted


def _extract_slate_items(example: dict) -> List[Tuple[str, str]]:
    """Collect slate titles/ids from text or trajectory metadata.

    :param example: Dataset row containing slate information.
    :returns: Sequence of unique ``(title, video_id)`` pairs.
    """
    items: list[Tuple[str, str]] = []
    slate_text = example.get("slate_text")
    if isinstance(slate_text, str) and slate_text.strip():
        for line in slate_text.splitlines():
            entry = line.strip()
            if not entry or entry == "-":
                continue
            match = re.match(r"^\s*(?:-|\d+\s*[\.\)])\s*(.+)$", entry)
            surface = match.group(1).strip() if match else entry
            video_id = canon_video_id(surface)
            if len(video_id) == 11:
                title = _PROMPT_DOC_BUILDER.title_for(video_id) or ""
                items.append((title, video_id))
            else:
                items.append((surface, ""))

    if not items:
        try:
            for element in load_trajectory_entries(example.get("trajectory_json")):
                raw_id = str(
                    pick_case_insensitive(
                        element,
                        "video_id",
                        "id",
                        "videoId",
                    )
                    or ""
                ).strip()
                title = str(
                    pick_case_insensitive(
                        element,
                        "title",
                        "video_title",
                        "name",
                        "videoTitle",
                    )
                    or ""
                ).strip()
                if is_nan_like(title) and raw_id:
                    title = _PROMPT_DOC_BUILDER.title_for(raw_id) or ""
                if raw_id or title:
                    items.append((title or "(untitled)", raw_id))
        except Exception:
            pass

    seen: set[str] = set()
    deduped: list[Tuple[str, str]] = []
    for title, video_id in items:
        key = canon_text(video_id) or canon_text(title)
        if key and key not in seen:
            seen.add(key)
            deduped.append((title, video_id))
    return deduped


def _format_history_lines(sequence: List[dict]) -> List[str]:
    """Render history dictionaries into user-readable bullet points.

    :param sequence: Canonicalised history rows.
    :returns: List of formatted history strings.
    """
    def _fmt_seconds(value: Any) -> str:
        """Format a raw seconds value into a friendly suffix representation.

        :param value: Raw seconds or timing value to convert.
        :returns: Human-readable suffix string (e.g. ``10s`` or ``?``).
        """
        try:
            return f"{int(round(float(value)))}s"
        except Exception:
            return "?"

    lines: list[str] = []
    for row in sequence:
        idx = row.get("idx")
        watched = row.get("watch_seconds")
        total = row.get("total_length")
        title = (row.get("title") or "").strip() or "(untitled)"
        left: list[str] = []
        if idx is not None:
            left.append(str(idx))
        if watched is not None or total is not None:
            left.append(f"{_fmt_seconds(watched)}/{_fmt_seconds(total)}")
        prefix = f"[{' • '.join(left)}] " if left else ""
        lines.append(f"- {prefix}{title}")
    return lines


def make_conversation_record(example: dict) -> Dict[str, Any]:
    """Transform a dataset row into the prompt payload consumed by GPT-4o."""

    max_history = int(os.environ.get("GRAIL_MAX_HISTORY", "8"))
    history_full_env = str(os.environ.get("GRAIL_HISTORY_FULL", "0")).lower()
    history_full_mode = str(os.environ.get("GRAIL_HISTORY_MODE_FULL", "0")).lower()
    show_full = (
        history_full_env in {"1", "true", "t", "yes", "y"}
        or history_full_mode in {"1", "true", "t", "yes", "y"}
        or max_history <= 0
    )

    builder_history = 0 if show_full else max_history
    _PROMPT_DOC_BUILDER.max_history = builder_history

    raw_context = str(example.get(PROMPT_COLUMN, "") or "").strip()
    if is_nan_like(raw_context):
        raw_context = ""

    base_prompt = _PROMPT_DOC_BUILDER.prompt_from_builder(example)
    if not base_prompt and raw_context:
        base_prompt = raw_context

    instruction_tail = (
        "\n\nAfter thinking in <think>, choose exactly one candidate from OPTIONS and"
        " return ONLY its NUMBER in <answer>."
    )
    user_message = (
        f"{base_prompt}{instruction_tail}" if base_prompt else instruction_tail.lstrip()
    )

    slate_pairs = _PROMPT_DOC_BUILDER.extract_slate_items(example)
    gold_raw = str(example.get(SOLUTION_COLUMN, "")).strip()
    gold_index = -1
    if gold_raw and slate_pairs:
        gold_id_canon = canon_video_id(gold_raw)
        gold_title_canon = canon_text(gold_raw)
        for idx, (title, video_id) in enumerate(slate_pairs, start=1):
            vid_canon = canon_video_id(video_id)
            title_canon = canon_text(title)
            if (
                gold_raw == video_id
                or (gold_id_canon and vid_canon == gold_id_canon)
                or (gold_title_canon and title_canon == gold_title_canon)
            ):
                gold_index = idx
                break

    now_watching = _PROMPT_DOC_BUILDER.extract_now_watching(example)
    if now_watching:
        now_title, now_id = now_watching
        now_line_state = f"{now_title or '(untitled)'}{(' — id: ' + now_id) if now_id else ''}"
    else:
        now_line_state = "(none)"

    profile_block = build_profile_block(example)

    position_index = example.get("video_index")
    try:
        position_index = int(position_index) if position_index is not None else -1
    except Exception:
        position_index = -1

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "gold_index": gold_index,
        "n_options": len(slate_pairs),
        "position_index": position_index,
        "metadata": {
            "now_playing": now_line_state,
            "profile_block": profile_block,
        },
    }
