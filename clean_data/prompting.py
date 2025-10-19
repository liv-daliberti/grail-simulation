"""Prompt rendering and row transformation utilities for cleaned GRAIL data."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from clean_data.helpers import _as_list_json, _canon, _is_nanlike
from clean_data.prompt.constants import (
    DEFAULT_SYSTEM_PROMPT,
    PASSTHROUGH_COLUMNS,
    YOUTUBE_FREQ_MAP,
)


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


def load_slate_items(ex: dict) -> List[dict]:
    """Parse ``slate_items_json`` into a list of dictionaries with ``title`` and ``id`` keys."""
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


def _format_age(age: Any) -> Optional[str]:
    """Return a formatted age fragment or ``None``."""
    try:
        age_i = int(age) if age not in (None, "", "nan") else None
    except (TypeError, ValueError):
        age_i = None
    if isinstance(age_i, int) and age_i > 0:
        return f"{age_i}-year-old"
    return None


def _format_gender(gender: Any) -> Optional[str]:
    """Return a formatted gender fragment or ``None``."""
    normalized = str(gender or "").strip().lower()
    if normalized in {"man", "male"}:
        return "man"
    if normalized in {"woman", "female"}:
        return "woman"
    if normalized:
        return normalized.title()
    return None


def _format_party_affiliation(pid1: Any, ideo1: Any) -> Optional[str]:
    """Return a formatted party/ideology fragment or ``None``."""
    party = str(pid1 or "").strip()
    ideo = str(ideo1 or "").strip()
    if party and party.lower() != "nan":
        if ideo and ideo.lower() != "nan":
            return f"{party} {ideo}".lower()
        return party
    if ideo and ideo.lower() != "nan":
        return ideo.lower()
    return None


def _format_income(income: Any) -> Optional[str]:
    """Return a formatted income fragment or ``None``."""
    inc = str(income or "").strip()
    if inc and inc.lower() != "nan":
        return inc
    return None


def _format_college(college: Any) -> Optional[str]:
    """Return a formatted education fragment or ``None``."""
    if _truthy_str_flag(college):
        return "college-educated"
    return None


def _format_race(race: Any) -> Optional[str]:
    """Return a formatted race/ethnicity fragment or ``None``."""
    text = _clean_str(race)
    if text and not _is_nanlike(text):
        return text
    return None


def _format_youtube_freq(freq: Any) -> Optional[str]:
    """Return a formatted YouTube frequency fragment or ``None``."""
    freq_key = str(freq or "").strip()
    if freq_key in YOUTUBE_FREQ_MAP:
        return f"watches YouTube {YOUTUBE_FREQ_MAP[freq_key]}"
    return None


def _synthesize_viewer_sentence(ex: dict) -> str:
    bits: List[str] = []
    formatters = (
        lambda: _format_age(ex.get("age")),
        lambda: _format_gender(ex.get("q26")),
        lambda: _format_race(ex.get("q29")),
        lambda: _format_party_affiliation(ex.get("pid1"), ex.get("ideo1")),
        lambda: _format_income(ex.get("q31")),
        lambda: _format_college(ex.get("college")),
        lambda: _format_youtube_freq(ex.get("freq_youtube")),
    )
    for formatter in formatters:
        value = formatter()
        if value:
            bits.append(value)
    s = ", ".join(bits)
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

    items = load_slate_items(ex)
    lines.append("\nOPTIONS:")
    if items:
        for idx, item in enumerate(items, 1):
            raw_name = (
                item.get("title")
                or (item.get("id") if show_ids else "")
                or "(untitled)"
            )
            lines.append(f"{idx}. {raw_name.strip()}")
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

def get_gold_next_id(ex: dict, sol_key: Optional[str]) -> str:
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

def gold_index_from_items(gold: str, items: List[dict]) -> int:
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


@dataclass(frozen=True)
class ExampleComponents:
    """Intermediate bundle for prompt example creation."""

    items: List[dict]
    gold_id: str
    gold_index: int
    user_message: str
    system_message: str
    slate_text: str


def _resolve_system_prompt(system_prompt: Optional[str]) -> str:
    """Return the caller-provided system prompt or the default template."""

    return system_prompt or DEFAULT_SYSTEM_PROMPT


def _resolve_slate_text(ex: dict, items: List[dict]) -> str:
    """Return a non-empty slate text representation."""

    existing = ex.get("slate_text")
    if existing:
        return str(existing)
    return "\n".join(
        f"{idx}. {(item.get('title') or item.get('id') or '(untitled)').strip()}"
        for idx, item in enumerate(items, 1)
    )


def _collect_example_components(
    ex: dict,
    system_prompt: Optional[str],
    sol_key: Optional[str],
    max_hist: int,
) -> Optional[ExampleComponents]:
    """Gather the prerequisite values required to build a cleaned example."""

    items = load_slate_items(ex)
    if not items:
        return None
    gold_id = get_gold_next_id(ex, sol_key)
    gidx = gold_index_from_items(gold_id, items)
    if gidx < 1:
        return None
    user_msg = _build_user_prompt_from_columns(ex, max_hist=max_hist)
    system_msg = _resolve_system_prompt(system_prompt)
    slate_text = _resolve_slate_text(ex, items)
    return ExampleComponents(
        items=items,
        gold_id=gold_id,
        gold_index=gidx,
        user_message=user_msg,
        system_message=system_msg,
        slate_text=slate_text,
    )


# ---------- row → clean example ----------
def row_to_example(
    ex: dict,
    system_prompt: Optional[str],
    sol_key: Optional[str],
    max_hist: int,
) -> Optional[dict]:
    """Transform a raw session pair into the cleaned GRPO example structure.

    :param ex: Interaction row produced during session processing.
    :param system_prompt: Optional system prompt override.
    :param sol_key: Alternate column to treat as the gold next-video id.
    :param max_hist: Maximum number of prior history entries to render.
    :returns: Cleaned example dictionary or ``None`` when the row is unusable.
    """

    components = _collect_example_components(
        ex,
        system_prompt,
        sol_key,
        max_hist,
    )
    if components is None:
        return None

    out = {
        "prompt": [
            {"role": "system", "content": components.system_message},
            {"role": "user", "content": components.user_message},
        ],
        "answer": str(components.gold_index),  # GOLD index as string
        "gold_index": components.gold_index,  # int
        "gold_id": components.gold_id,
        "n_options": int(ex.get("n_options") or len(components.items) or 0),
        "viewer_profile": str(
            ex.get("viewer_profile_sentence") or _synthesize_viewer_sentence(ex)
        ),
        "state_text": components.user_message,  # LM sees this
        "state_disc_text": _build_state_disc_text(ex),  # disc sees richer info
        "slate_items": components.items,
        "slate_text": components.slate_text,
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


__all__ = [
    "YOUTUBE_FREQ_MAP",
    "PASSTHROUGH_COLUMNS",
    "load_slate_items",
    "get_gold_next_id",
    "gold_index_from_items",
    "row_to_example",
]
