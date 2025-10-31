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

"""Prompt construction utilities for GPT-4o slate evaluations."""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import suppress
from importlib import import_module
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import PROMPT_COLUMN, SOLUTION_COLUMN, SYSTEM_PROMPT
from .utils import canon_text, canon_video_id, is_nan_like, truthy
_docs = import_module("common.prompts.docs")
PromptDocumentBuilder = _docs.PromptDocumentBuilder
default_title_resolver = _docs.default_title_resolver
load_trajectory_entries = _docs.load_trajectory_entries
_fields = import_module("common.prompts.fields")
NOW_PLAYING_ID_KEYS = _fields.NOW_PLAYING_ID_KEYS
NOW_PLAYING_TITLE_KEYS = _fields.NOW_PLAYING_TITLE_KEYS
NOW_PLAYING_TITLE_KEYS_WITH_META = _fields.NOW_PLAYING_TITLE_KEYS_WITH_META

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


def _gender_label(example: dict) -> Optional[str]:
    """Return a gender label inferred from binary survey signals.

    :param example: Dataset row containing boolean gender indicators.
    :returns: Canonical gender string or ``None`` when unspecified.
    """

    female = truthy(example.get("female"))
    male = truthy(example.get("male"))
    if female and not male:
        return "woman"
    if male and not female:
        return "man"
    return None


def _party_from_text(lowered: str, original: str) -> Optional[str]:
    """Return a normalised party label based on ``lowered`` text.

    :param lowered: Lower-cased party descriptor.
    :param original: Original string prior to normalisation.
    :returns: Canonical party description when recognised.
    """

    mapping = {
        "democrat": "Democratic",
        "dem": "Democratic",
        "republican": "Republican",
        "rep": "Republican",
        "gop": "Republican",
        "independent": "Independent",
        "libertarian": "Libertarian",
        "green": "Green",
    }
    for token, label in mapping.items():
        if token in lowered:
            return label
    if "closer to the democratic" in lowered:
        return "Democratic-leaning"
    if "closer to the republican" in lowered:
        return "Republican-leaning"
    return original.strip() if original.strip() else None


def _numeric_ideology_label(value: str) -> Optional[str]:
    """Return a canonical ideology label derived from numeric encodings.

    :param value: Raw string value representing the ideology indicator.
    :returns: Canonical ideology label or ``None`` when parsing fails.
    """

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    thresholds = (
        (1.5, "Extremely liberal"),
        (2.5, "Liberal"),
        (3.5, "Slightly liberal"),
        (4.5, "Moderate"),
        (5.5, "Slightly conservative"),
        (6.5, "Conservative"),
    )
    selected = "Extremely conservative"
    for limit, label in thresholds:
        if numeric <= limit:
            selected = label
            break
    return selected


def _textual_ideology_label(value: str) -> Optional[str]:
    """Return a canonical ideology label derived from textual descriptors.

    :param value: Raw string description of ideology.
    :returns: Canonical ideology label or ``None`` when unrecognised.
    """

    lowered = value.lower()
    mapping = (
        ("extreme", "lib", "Extremely liberal"),
        ("lib", None, "Liberal"),
        ("moderate", None, "Moderate"),
        ("centrist", None, "Moderate"),
        ("conservative", "extreme", "Extremely conservative"),
        ("conservative", None, "Conservative"),
    )
    for token, qualifier, label in mapping:
        if token in lowered and (qualifier is None or qualifier in lowered):
            return label
    return value.strip() or None


def pick_case_insensitive(record: dict, *candidates: str) -> Optional[str]:
    """Return the first matching candidate value from ``record``.

    :param record: Mapping inspected for possible keys.
    :param candidates: Ordered list of candidate keys to search (case-insensitive).
    :returns: Stripped value associated with the first matching key, or ``None``.
    """

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
    """Mirror the income extraction logic used in the GRPO baseline.

    :param example: Dataset row containing demographic metadata.
    :returns: Income phrase or normalised descriptor, else ``None``.
    """

    for key in _INCOME_HINT_KEYS:
        value = example.get(key)
        if isinstance(value, str) and not is_nan_like(value):
            stripped = value.strip()
            if _INCOME_PATTERN.search(stripped) or "income" in stripped.lower():
                return stripped
    if isinstance(example, dict):
        for key, value in example.items():
            if isinstance(value, str) and not is_nan_like(value) and _INCOME_PATTERN.search(value):
                return value.strip()
    if "income_gt50k" in example:
        high_income = bool(example.get("income_gt50k"))
        return ">$50k household income" if high_income else "≤$50k household income"
    return None


def extract_party(example: dict) -> Optional[str]:
    """Return a party label derived from standard survey signals.

    :param example: Dataset row containing partisan survey responses.
    :returns: Normalised party label or ``None`` when absent.
    """

    pid_text = example.get("pid")
    if isinstance(pid_text, str) and not is_nan_like(pid_text):
        lowered = pid_text.strip().lower()
        candidate = _party_from_text(lowered, pid_text)
        if candidate:
            return candidate

    values: list[float] = []
    for key in ("pid1", "pid2", "pid3", "pid4"):
        value = example.get(key)
        if value is not None and str(value).strip() != "":
            with suppress(TypeError, ValueError):
                values.append(float(value))
    if values:
        mean_value = sum(values) / len(values)
        if mean_value <= 3.0:
            return "Democratic-leaning"
        if mean_value >= 5.0:
            return "Republican-leaning"
        return "Independent/Other"
    return None


def format_ideology(raw: Any) -> Optional[str]:
    """Convert ideology encodings to descriptive text.

    :param raw: Raw ideology indicator (numeric or textual) from the dataset.
    :returns: Canonical ideology phrase or ``None`` when unavailable.
    """

    if raw is None:
        return None
    value = str(raw).strip()
    if is_nan_like(value):
        return None

    numeric_label = _numeric_ideology_label(value)
    if numeric_label is not None:
        return numeric_label

    textual_label = _textual_ideology_label(value)
    return textual_label


def extract_marital_status(example: dict) -> Optional[str]:
    """Extract a brief marital descriptor.

    :param example: Dataset row containing marital-status related fields.
    :returns: Normalised marital descriptor or ``None`` if unknown.
    """

    synonyms = (
        (("married",), "Married"),
        (("partner", "cohabit"), "Living with partner"),
        (("single",), "Single"),
        (("divorced",), "Divorced"),
        (("widow",), "Widowed"),
        (("separated",), "Separated"),
    )
    result: Optional[str] = None
    for key in _MARITAL_KEYS:
        value = example.get(key)
        if isinstance(value, str) and not is_nan_like(value):
            lowered = value.strip().lower()
            for tokens, label in synonyms:
                if any(token in lowered for token in tokens):
                    result = label
                    break
            if result is None and value.strip():
                result = value.strip()
            break
    return result


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


def _demographic_fragments(example: dict) -> list[str]:
    """Return demographic fragments used in the profile sentence.

    :param example: Dataset row containing demographic indicators
        (e.g. ``female``, ``male``, race fields, marital status).
    :returns: List of short phrases that describe the viewer's demographics.
    :rtype: list[str]
    """

    fragments: list[str] = []
    race = extract_race(example)
    gender = _gender_label(example)

    if race and gender:
        fragments.append(f"{race} {gender}")
    else:
        fragments.extend([frag for frag in (gender, race) if frag])

    marital = extract_marital_status(example)
    if marital and not is_nan_like(marital):
        common = {"Married", "Single", "Divorced", "Widowed", "Separated"}
        fragments.append(marital.lower() if marital in common else marital)

    return fragments


def _political_fragments(example: dict) -> list[str]:
    """Return political fragments for the viewer profile.

    :param example: Dataset row with partisan/ideology and income signals.
    :returns: List of phrases capturing party/ideology and income when present.
    :rtype: list[str]
    """

    fragments: list[str] = []
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
    return fragments


def _education_fragments(example: dict) -> list[str]:
    """Return education-related fragments for the profile.

    :param example: Dataset row with a ``college`` boolean flag.
    :returns: List containing a single education phrase when applicable.
    :rtype: list[str]
    """

    if truthy(example.get("college")):
        return ["college-educated"]
    return []


def _viewing_fragments(example: dict) -> list[str]:
    """Return viewing-habit fragments for the profile.

    :param example: Dataset row with ``freq_youtube`` frequency values.
    :returns: List containing a single viewing-habit phrase when recognised.
    :rtype: list[str]
    """

    youtube_freq = str(example.get("freq_youtube", "")).strip()
    if youtube_freq in YT_FREQ_MAP:
        viewing_phrase = YT_FREQ_MAP.get(youtube_freq, "regularly")
        return [f"watches YouTube {viewing_phrase}"]
    return []


def _age_fragments(example: dict) -> list[str]:
    """Return age-based fragments for the viewer profile.

    :param example: Dataset row containing age metadata.
    :returns: List containing a single age descriptor when available.
    """

    age = example.get("age")
    if isinstance(age, (int, float)) and age > 0:
        return [f"{int(age)}-year-old"]
    return []


def extract_race(example: dict) -> Optional[str]:
    """Extract race information following the GRPO baseline rules.

    :param example: Dataset row containing race attributes or textual fields.
    :returns: Canonical race label or ``None`` when no signal is found.
    """

    if truthy(example.get("white")) and truthy(example.get("black")):
        return "Multiracial"
    if truthy(example.get("white")):
        return "White"
    if truthy(example.get("black")):
        return "Black"

    for field in _RACE_TEXT_FIELDS:
        value = example.get(field)
        if isinstance(value, str) and not is_nan_like(value):
            normalised = _normalise_race_token(value)
            if normalised and not is_nan_like(normalised):
                return normalised
    return None


def humanise_profile(example: dict) -> str:
    """Build a human-readable viewer profile sentence.

    :param example: Dataset row containing demographic and behavioural signals.
    :returns: Comma-delimited sentence summarising the viewer.
    """

    fragments: list[str] = []
    fragments.extend(_age_fragments(example))
    fragments.extend(_demographic_fragments(example))
    fragments.extend(_political_fragments(example))
    fragments.extend(_education_fragments(example))
    fragments.extend(_viewing_fragments(example))

    sentence_parts = [
        fragment for fragment in fragments if fragment and not is_nan_like(fragment)
    ]
    sentence = ", ".join(sentence_parts)
    return sentence if sentence else "(no profile provided)"


def build_profile_block(example: dict) -> str:
    """Return the structured key/value profile block.

    :param example: Dataset row containing viewer metadata.
    :returns: Multi-line string enumerating profile facts.
    """

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
        except (json.JSONDecodeError, TypeError, AttributeError):
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
    except (json.JSONDecodeError, TypeError):
        return None, None
    pointer = data.get("current_index")
    if pointer is None:
        pointer = data.get("currentIdx")
    end_ms = data.get("current_end_ms")
    if end_ms is None:
        end_ms = data.get("currentEndMs")
    pointer_int = None
    with suppress(TypeError, ValueError):
        pointer_int = int(pointer) if pointer is not None else None
    end_float = None
    with suppress(TypeError, ValueError):
        end_float = float(end_ms) if end_ms is not None else None
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
    except (json.JSONDecodeError, TypeError):
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
        with suppress(TypeError, ValueError):
            return (1, float(entry.get("end_ms") or -1))
        return (1, -1.0)

    extracted.sort(key=_hist_key)
    if up_to_idx is not None:
        extracted = [
            row
            for row in extracted
            if (row.get("idx") is None)
            or (row["idx"] < up_to_idx or (include_current and row["idx"] == up_to_idx))
        ]
    elif up_to_end_ms is not None:
        with suppress(TypeError, ValueError):
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
        with suppress(json.JSONDecodeError, TypeError, AttributeError):
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
        with suppress(TypeError, ValueError):
            return f"{int(round(float(value)))}s"
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


def _history_builder_size() -> int:
    """Return the max history size configured for prompt construction.

    :returns: Integer describing the history depth for the prompt builder.
    """

    max_history = int(os.environ.get("GRAIL_MAX_HISTORY", "8"))
    history_full_env = str(os.environ.get("GRAIL_HISTORY_FULL", "0")).lower()
    history_full_mode = str(os.environ.get("GRAIL_HISTORY_MODE_FULL", "0")).lower()
    show_full = (
        history_full_env in {"1", "true", "t", "yes", "y"}
        or history_full_mode in {"1", "true", "t", "yes", "y"}
        or max_history <= 0
    )
    return 0 if show_full else max_history


def _sanitise_context(example: dict) -> str:
    """Return the raw prompt context stripped of NaN-like placeholders.

    :param example: Dataset row containing prompt metadata.
    :returns: Stripped context string or an empty fallback.
    """

    raw_context = str(example.get(PROMPT_COLUMN, "") or "").strip()
    return "" if is_nan_like(raw_context) else raw_context


def _compose_user_message(base_prompt: str) -> str:
    """Compose the user message including the instruction tail.

    :param base_prompt: Viewer context built by the prompt builder.
    :returns: Full user message dispatched to GPT-4o.
    """

    instruction_tail = (
        "\n\nAfter thinking in <think>, choose exactly one candidate from OPTIONS and"
        " return ONLY its NUMBER in <answer>."
    )
    return f"{base_prompt}{instruction_tail}" if base_prompt else instruction_tail.lstrip()


def _resolve_gold_index(example: dict, slate_pairs: Sequence[Tuple[str, str]]) -> int:
    """Determine the gold option index associated with ``example``.

    :param example: Dataset row containing answer metadata.
    :param slate_pairs: Ordered slate entries presented to the model.
    :returns: 1-based gold index or ``-1`` when unavailable.
    """

    gold_raw = _preferred_answer(example)
    matched = _match_gold_to_slate(gold_raw, slate_pairs)
    if matched > 0:
        return matched

    limit = len(slate_pairs) if slate_pairs else None
    dataset_index = _bounded_index(example.get("gold_index"), limit)
    if dataset_index is not None:
        return dataset_index
    answer_index = _bounded_index(example.get("answer"), limit)
    return answer_index if answer_index is not None else -1


def _now_playing_line(example: dict) -> str:
    """Return the formatted now-playing metadata line.

    :param example: Dataset row containing now-playing metadata.
    :returns: Human-readable description of the current video.
    """

    now_watching = _PROMPT_DOC_BUILDER.extract_now_watching(example)
    if not now_watching:
        return "(none)"
    title, video_id = now_watching
    return f"{title or '(untitled)'}{(' — id: ' + video_id) if video_id else ''}"


def _position_index(example: dict) -> int:
    """Return the integer position index for ``example``.

    :param example: Dataset row containing ``video_index`` metadata.
    :returns: Zero-based position index or ``-1`` by default.
    """

    with suppress(TypeError, ValueError):
        raw = example.get("video_index")
        if raw is not None:
            return int(raw)
    return -1


def _preferred_answer(example: dict) -> str:
    """Return the canonical answer string extracted from the dataset row.

    :param example: Dataset row containing potential answer keys.
    :returns: Preferred answer string or an empty string when absent.
    """

    candidate_keys = (
        SOLUTION_COLUMN if isinstance(SOLUTION_COLUMN, str) else "",
        "gold_id",
        "next_video_id",
    )
    for key in candidate_keys:
        if not key:
            continue
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _match_gold_to_slate(gold_raw: str, slate_pairs: Sequence[Tuple[str, str]]) -> int:
    """Return the 1-based slate index matching ``gold_raw`` if available.

    :param gold_raw: Answer string extracted from the dataset.
    :param slate_pairs: Ordered slate entries presented to the model.
    :returns: 1-based index when a match is found; ``-1`` otherwise.
    """

    if not gold_raw or not slate_pairs:
        return -1
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
            return idx
    return -1


def _bounded_index(value: object, limit: Optional[int]) -> Optional[int]:
    """Return a positive index when within ``limit``; otherwise ``None``.

    :param value: Raw candidate index to coerce.
    :param limit: Optional inclusive upper bound for valid indices.
    :returns: Positive integer index or ``None`` when out of bounds.
    """

    with suppress(TypeError, ValueError):
        candidate = int(value)
        if candidate > 0 and (limit is None or candidate <= limit):
            return candidate
    return None


def make_conversation_record(example: dict) -> Dict[str, Any]:
    """Transform a dataset row into the prompt payload consumed by GPT-4o.

    :param example: Dataset row containing slate options and viewer metadata.
    :returns: Dictionary with prompt messages and evaluation metadata.
    """
    _PROMPT_DOC_BUILDER.max_history = _history_builder_size()

    base_prompt = _PROMPT_DOC_BUILDER.prompt_from_builder(example)
    if not base_prompt:
        base_prompt = _sanitise_context(example)
    user_message = _compose_user_message(base_prompt)

    slate_pairs = _extract_slate_items(example)
    gold_index = _resolve_gold_index(example, slate_pairs)
    now_line_state = _now_playing_line(example)
    profile_block = build_profile_block(example)

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "gold_index": gold_index,
        "n_options": len(slate_pairs),
        "position_index": _position_index(example),
        "metadata": {
            "now_playing": now_line_state,
            "profile_block": profile_block,
        },
    }
