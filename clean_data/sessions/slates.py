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

"""Utilities for deriving recommendation slates from session metadata."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from clean_data.helpers import (
    _as_list_json,
    _canon,
    _strip_session_video_id,
)


def _normalize_display_step_index(key: Any) -> Optional[int]:
    """Return the zero-based step index encoded by the display-order key."""

    if isinstance(key, int):
        return key if key >= 0 else None
    if not isinstance(key, str):
        return None
    if key.isdigit():
        return int(key)
    if key.startswith("step"):
        suffix = key[4:].strip("_:")
        return int(suffix) if suffix.isdigit() else None
    return None


def normalize_display_orders(display_orders: Any) -> Dict[int, List[str]]:
    """Return normalized mapping from step index to canonical video ids."""

    normalized: Dict[int, List[str]] = {}
    if not isinstance(display_orders, dict):
        return normalized
    for key, value in display_orders.items():
        if not isinstance(value, (list, tuple)):
            continue
        step_idx = _normalize_display_step_index(key)
        if step_idx is None:
            continue

        vids = [
            stripped
            for item in value
            if isinstance(item, str)
            for stripped in [_strip_session_video_id(item)]
            if stripped
        ]
        if vids:
            normalized[step_idx] = vids
    return normalized


def _gather_slate_candidates(
    step_index: int,
    display_orders: Dict[int, List[str]],
    recommendations: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Tuple[str, Optional[str]]], str]:
    """Return ordered ``(base_id, raw_id)`` candidates and their source."""

    order_ids = display_orders.get(step_index, [])
    if order_ids:
        pairs: List[Tuple[str, Optional[str]]] = []
        for vid in order_ids:
            base_vid = _strip_session_video_id(vid)
            if not base_vid:
                continue
            raw_id = vid if vid != base_vid else None
            pairs.append((base_vid, raw_id))
        return pairs, "display_orders"

    candidates: List[Tuple[str, Optional[str]]] = []
    if not isinstance(recommendations, list):
        recommendations = []
    for rec in recommendations:
        if not isinstance(rec, dict):
            continue
        rec_id = rec.get("id") or rec.get("video_id") or rec.get("raw_id")
        rec_id = _strip_session_video_id(str(rec_id or ""))
        if not rec_id:
            continue
        raw_id = rec.get("raw_id")
        if raw_id:
            raw_id = _strip_session_video_id(str(raw_id))
        candidates.append((rec_id, raw_id if raw_id != rec_id else None))
    return candidates, "tree_metadata"


def _make_slate_item(
    base_id: str,
    raw_id: Optional[str],
    tree_meta: Dict[str, Dict[str, Any]],
    fallback_titles: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct a single slate entry with metadata fallbacks."""

    meta = tree_meta.get(base_id, {})
    title_candidates = (
        meta.get("title"),
        fallback_titles.get(base_id),
        fallback_titles.get(raw_id) if raw_id else None,
    )
    title = next(
        (
            str(candidate).strip()
            for candidate in title_candidates
            if isinstance(candidate, str) and str(candidate).strip()
        ),
        "",
    )
    if not title and raw_id:
        title = raw_id
    if not title:
        title = base_id
    item: Dict[str, Any] = {"id": base_id, "title": title}
    if raw_id:
        item["raw_id"] = raw_id
    if meta.get("channel_title"):
        item["channel_title"] = meta["channel_title"]
    if meta.get("channel_id"):
        item["channel_id"] = meta["channel_id"]
    for stat_key in (
        "view_count",
        "like_count",
        "dislike_count",
        "favorite_count",
        "comment_count",
    ):
        stat_value = meta.get(stat_key)
        if stat_value not in (None, ""):
            item[stat_key] = stat_value
    duration_value = meta.get("duration")
    if duration_value not in (None, ""):
        item["duration"] = duration_value
    return item


def build_slate_items(
    step_index: int,
    display_orders: Dict[int, List[str]],
    recommendations: Optional[List[Dict[str, Any]]],
    tree_meta: Dict[str, Dict[str, Any]],
    fallback_titles: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    """Derive the slate items for the given interaction step."""

    candidates, source = _gather_slate_candidates(
        step_index,
        display_orders,
        recommendations,
    )

    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for base_id, raw_id in candidates:
        if base_id in seen:
            continue
        seen.add(base_id)
        items.append(_make_slate_item(base_id, raw_id, tree_meta, fallback_titles))
    return items, source


def load_slate_items(ex: dict) -> List[dict]:
    """Parse ``slate_items_json`` into a normalized list of ``{title, id}`` dicts."""

    arr = _as_list_json(ex.get("slate_items_json"))
    out: List[dict] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        vid = (item.get("id") or "").strip()
        if title or vid:
            out.append({"title": title, "id": vid})
    return out


def derive_next_from_history(ex: dict, current_id: str) -> str:
    """Infer the next video id from the watch history when explicit labels are missing."""

    vids = _as_list_json(ex.get("watched_vids_json"))
    if current_id and isinstance(vids, list) and vids:
        try:
            idx = vids.index(current_id)
            if idx + 1 < len(vids):
                nxt = vids[idx + 1]
                if isinstance(nxt, str) and nxt.strip():
                    return nxt.strip()
        except ValueError:
            pass

    detailed_history = _as_list_json(ex.get("watched_detailed_json"))
    if current_id and isinstance(detailed_history, list) and detailed_history:
        for entry_index, detail_row in enumerate(detailed_history):
            if isinstance(detail_row, dict) and (detail_row.get("id") or "").strip() == current_id:
                if entry_index + 1 < len(detailed_history):
                    nxt = (detailed_history[entry_index + 1].get("id") or "").strip()
                    if nxt:
                        return nxt
                break
    return ""


def _normalize_candidate(value: Any, current_id: str) -> str:
    """Return a cleaned gold-id candidate or ``""`` when unusable."""

    if value is None:
        return ""
    text = str(value).strip()
    if not text or text == current_id:
        return ""
    return text


def get_gold_next_id(ex: dict, sol_key: Optional[str]) -> str:
    """Return the gold next-video identifier derived from ``sol_key``.

    Historically callers relied on ``sol_key=None`` to mean "use the standard
    ``next_video_id`` column". A regression removed that fallback, which meant
    downstream helpers saw empty gold ids and dropped otherwise valid rows.
    Restoring the legacy precedence keeps ``None`` working while still allowing
    explicit overrides when provided.
    """

    current_id = _normalize_candidate(ex.get("current_video_id"), "") or ""

    if sol_key:
        key = str(sol_key).strip()
        if key and key not in {"current_video_id", "current_id"}:
            override = _normalize_candidate(ex.get(key), current_id)
            if override:
                return override

    for field_name in ("next_video_id", "clicked_id", "label", "answer"):
        candidate = _normalize_candidate(ex.get(field_name), current_id)
        if candidate:
            return candidate

    derived = derive_next_from_history(ex, current_id)
    return _normalize_candidate(derived, current_id)


def gold_index_from_items(gold: str, items: List[dict]) -> int:
    """Return 1-based index of the gold item within ``items``."""

    if not gold or not items:
        return -1
    for idx, item in enumerate(items, 1):
        if gold == (item.get("id") or ""):
            return idx
    canon = _canon(gold)
    if not canon:
        return -1
    for idx, item in enumerate(items, 1):
        if canon == _canon(item.get("title", "")):
            return idx
    return -1


__all__ = [
    "build_slate_items",
    "derive_next_from_history",
    "get_gold_next_id",
    "gold_index_from_items",
    "load_slate_items",
    "normalize_display_orders",
]
