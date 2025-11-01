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

"""Utilities for constructing Open-R1 training and evaluation examples."""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from common.opinion.models import DEFAULT_SPECS

# Import from explicit submodules so static analyzers (pylint) can resolve names.
from prompt_builder.formatters import clean_text
from prompt_builder.parsers import as_list_json, is_nanlike
from prompt_builder.profiles.render import synthesize_viewer_sentence
from prompt_builder.prompt import build_user_prompt

from .constants import DEFAULT_SYSTEM_PROMPT
from .shared import build_training_example, collect_passthrough_fields

_CANON_RE = re.compile(r"[^a-z0-9]+")
_OPINION_TOLERANCE = 1e-6
_OPINION_SPEC_LOOKUP = {
    (spec.issue.lower(), spec.key.lower()): spec for spec in DEFAULT_SPECS
}


def _normalise_direction(value: float) -> float:
    """Return ``value`` when finite, otherwise ``NaN``.

    :param value: Numeric opinion delta to normalise.
    :returns: Float value or ``NaN`` when conversion fails.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _opinion_direction_label(example: Mapping[str, Any]) -> Optional[str]:
    """Compute the opinion-direction label for ``example`` when available.

    :param example: Dataset row containing opinion fields.
    :returns: One of ``\"increase\"``, ``\"decrease\"``, ``\"no_change\"``, or ``None``.
    """

    issue = str(example.get("issue") or "").strip().lower()
    study = str(example.get("participant_study") or "").strip().lower()
    spec = _OPINION_SPEC_LOOKUP.get((issue, study))
    if spec is None:
        return None

    before = _normalise_direction(example.get(spec.before_column))
    after = _normalise_direction(example.get(spec.after_column))
    if math.isnan(before) or math.isnan(after):
        return None

    delta = after - before
    if delta > _OPINION_TOLERANCE:
        return "increase"
    if delta < -_OPINION_TOLERANCE:
        return "decrease"
    return "no_change"


def canon(value: str) -> str:
    """Return a lowercase, punctuation-stripped token for fuzzy comparisons.

    :param value: Raw text to normalise.
    :returns: Canonicalised token ready for string matching.
    """

    return _CANON_RE.sub("", (value or "").lower().strip())


def load_slate_items(example: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Extract and normalise slate metadata from ``example``.

    :param example: Dataset row containing slate metadata JSON.
    :returns: Cleaned list of slate item dictionaries with stable keys.
    """

    arr = as_list_json(example.get("slate_items_json"))
    keep_keys = {
        "title",
        "id",
        "raw_id",
        "video_id",
        "video_title",
        "channel",
        "channel_title",
        "channel_name",
        "channel_id",
        "length_seconds",
        "duration_seconds",
        "duration",
        "watch_seconds",
        "score",
        "rank",
        "position",
        "reason",
        "source",
    }
    out: List[Dict[str, Any]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        cleaned: Dict[str, Any] = {}
        for key, value in item.items():
            if key in keep_keys and not is_nanlike(value):
                cleaned[key] = value

        title_source = (
            item.get("title")
            or item.get("name")
            or item.get("video_title")
            or cleaned.get("title")
        )
        title = clean_text(title_source, limit=160)

        video_id_source = (
            item.get("id") or item.get("raw_id") or item.get("video_id") or cleaned.get("id")
        )
        video_id = clean_text(video_id_source)

        channel_source = (
            item.get("channel_title")
            or item.get("channel_name")
            or item.get("channel")
            or cleaned.get("channel_title")
        )
        channel = clean_text(channel_source, limit=120)
        if title:
            cleaned["title"] = title
        if video_id:
            cleaned["id"] = video_id
        if channel:
            cleaned["channel_title"] = channel
        if cleaned.get("id") or cleaned.get("title"):
            out.append(cleaned)
    return out


def gold_index_from_items(gold: str, items: Sequence[Mapping[str, Any]]) -> int:
    """Return the 1-based index of ``gold`` within ``items`` or ``-1`` when missing.

    :param gold: Gold identifier expected to be present in the slate.
    :param items: Slate items to search for the identifier.
    :returns: 1-based index of the gold item or ``-1`` when not found.
    """

    gold = (gold or "").strip()
    if not gold or not items:
        return -1
    for idx, item in enumerate(items, 1):
        if gold == (item.get("id") or ""):
            return idx
    canonical = canon(gold)
    if canonical:
        for idx, item in enumerate(items, 1):
            if canonical == canon(item.get("title", "")):
                return idx
    return -1


def derive_next_from_history(example: Mapping[str, Any], current_id: str) -> str:
    """Infer the next-click identifier from historical watch data.

    :param example: Dataset row containing historical watch information.
    :param current_id: Identifier of the current video in the viewer session.
    :returns: Derived next video identifier or an empty string when unavailable.
    """

    vids = as_list_json(example.get("watched_vids_json"))
    if current_id and isinstance(vids, list) and vids:
        try:
            current_index = vids.index(current_id)
        except ValueError:
            current_index = -1
        if current_index >= 0 and current_index + 1 < len(vids):
            nxt = vids[current_index + 1]
            if isinstance(nxt, str) and nxt.strip():
                return nxt.strip()

    detailed = as_list_json(example.get("watched_detailed_json"))
    if current_id and isinstance(detailed, list) and detailed:
        for pos, record in enumerate(detailed):
            if isinstance(record, dict) and (record.get("id") or "").strip() == current_id:
                if pos + 1 < len(detailed):
                    nxt = (detailed[pos + 1].get("id") or "").strip()
                    if nxt:
                        return nxt
                break
    return ""


def get_gold_next_id(example: Mapping[str, Any], solution_key: Optional[str]) -> str:
    """Return the preferred next-video identifier from ``example``.

    :param example: Dataset row containing next-video fields.
    :param solution_key: Optional override for the column holding the gold id.
    :returns: Gold identifier for the next recommendation.
    """

    if solution_key and solution_key not in {"current_video_id", "current_id"}:
        value = example.get(solution_key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("next_video_id", "clicked_id", "video_id", "label", "answer"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    current = (example.get("current_video_id") or "").strip()
    return derive_next_from_history(example, current)


ExtraFieldsFn = Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], Mapping[str, Any]]


# pylint: disable=too-many-arguments
def row_to_training_example(
    example: Mapping[str, Any],
    *,
    system_prompt: Optional[str],
    solution_key: Optional[str],
    max_history: int = 12,
    passthrough_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = collect_passthrough_fields,
    extra_fields_fn: Optional[ExtraFieldsFn] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert a raw dataset ``example`` into the GRPO training payload.

    :param example: Dataset row describing the viewer slate and metadata.
    :param system_prompt: Optional system prompt to include in the example.
    :param solution_key: Column containing the gold recommendation identifier.
    :param max_history: Maximum number of history entries to include.
    :param passthrough_fn: Callable generating passthrough fields to retain.
    :param extra_fields_fn: Optional callable returning additional fields.
    :returns: Training example dict or ``None`` when the slate/gold mapping is invalid.
    """

    items = load_slate_items(example)
    if not items:
        return None
    gold_id = get_gold_next_id(example, solution_key)
    gold_index = gold_index_from_items(gold_id, items)
    if gold_index < 1:
        return None

    user_prompt = build_user_prompt(example, max_hist=max_history)
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    slate_text = "\n".join(
        f"{idx}. {(item.get('title') or item.get('id') or '(untitled)').strip()}"
        for idx, item in enumerate(items, 1)
    )

    extra_fields: Dict[str, Any] = {}
    if passthrough_fn is not None:
        extra_fields.update(passthrough_fn(example))
    if extra_fields_fn is not None:
        extra_fields.update(extra_fields_fn(example, items))
    extra_fields["opinion_direction"] = _opinion_direction_label(example)

    payload_extra_fields = extra_fields or None

    return build_training_example(
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        gold_index=gold_index,
        gold_id=gold_id,
        n_options=int(example.get("n_options") or len(items) or 0),
        viewer_profile=str(
            example.get("viewer_profile_sentence") or synthesize_viewer_sentence(example)
        ),
        slate_items=items,
        slate_text=slate_text,
        watched_detailed_json=as_list_json(example.get("watched_detailed_json")),
        watched_vids_json=as_list_json(example.get("watched_vids_json")),
        current_video_id=str(example.get("current_video_id") or ""),
        current_video_title=str(example.get("current_video_title") or ""),
        extra_fields=payload_extra_fields,
    )


def call_row_to_training_example(
    example: Mapping[str, Any],
    *,
    system_prompt: Optional[str],
    solution_key: Optional[str],
    max_history: int = 12,
    passthrough_fn: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] = collect_passthrough_fields,
    extra_fields_fn: Optional[ExtraFieldsFn] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convenience wrapper around :func:`row_to_training_example`.

    Centralises the keyword arguments setup so higher-level modules can invoke the conversion
    without repeating identical boilerplate, keeping :mod:`pylint` duplicate-code checks quiet.

    :param example: Dataset row describing the viewer slate and metadata.
    :param system_prompt: Optional system prompt to pass through.
    :param solution_key: Column containing the gold recommendation identifier.
    :param max_history: Maximum history entries to include in prompts.
    :param passthrough_fn: Callable providing passthrough fields; optional.
    :param extra_fields_fn: Callable attaching extra fields to the training example.
    :returns: Training example dict or ``None`` when the slate/gold mapping is invalid.
    """

    kwargs: Dict[str, Any] = {
        "system_prompt": system_prompt,
        "solution_key": solution_key,
        "max_history": max_history,
    }
    if passthrough_fn is not None:
        kwargs["passthrough_fn"] = passthrough_fn
    if extra_fields_fn is not None:
        kwargs["extra_fields_fn"] = extra_fields_fn
    return row_to_training_example(example, **kwargs)


__all__ = [
    "canon",
    "derive_next_from_history",
    "get_gold_next_id",
    "gold_index_from_items",
    "load_slate_items",
    "call_row_to_training_example",
    "row_to_training_example",
]
