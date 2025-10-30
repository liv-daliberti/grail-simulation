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

"""Dataclasses and formatting helpers for recommendation tree visualisations."""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from common.opinion import float_or_none
except ImportError:  # pragma: no cover - optional dependency for linting environments
    def float_or_none(value: Any) -> Optional[float]:
        """Best-effort local fallback mirroring the upstream helper."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _wrap_text(text: str, width: Optional[int]) -> str:
    """Wrap ``text`` to the specified ``width`` using newline separators."""

    if not width or width <= 0:
        return text
    return "\n".join(textwrap.wrap(text, width))


_QUOTE_RUN_RE = re.compile(r'"{2,}')


def _clean_repeated_quotes(text: str) -> str:
    """Collapse repeated double-quote runs to a single character."""

    return _QUOTE_RUN_RE.sub('"', text)


@dataclass
class TreeEdge:
    """Directed edge connecting two nodes in the recommendation tree."""

    parent: str
    child: str
    rank: Optional[int] = None


@dataclass
class TreeData:
    """Container holding recommendation tree nodes and edges."""

    root: str
    nodes: Dict[str, Mapping[str, object]]
    edges: List[TreeEdge]


@dataclass(frozen=True)
class OpinionFieldSpec:
    """Specification describing opinion score columns used by a viewer cohort."""

    before_keys: Tuple[str, ...]
    after_keys: Tuple[str, ...]
    label: str


@dataclass
class OpinionAnnotation:
    """Resolved opinion values ready for annotation on a session graph."""

    issue: str
    before_value: Optional[float]
    after_value: Optional[float]
    label: str


@dataclass(frozen=True)
class LabelRenderOptions:
    """Configuration controlling how node labels are rendered."""

    metadata: Mapping[str, Mapping[str, object]]
    template: str
    wrap_width: Optional[int]
    append_id_if_missing: bool = True


class SafeDict(dict):
    """Dictionary that returns placeholder values for missing template keys."""

    def __missing__(self, key: str) -> str:
        return ""



DEFAULT_LABEL_TEMPLATE = "{originTitle}\n{id}"
SESSION_DEFAULT_LABEL_TEMPLATE = "{title}"


_OPINION_FIELD_MAP: Dict[str, OpinionFieldSpec] = {
    "gun_control": OpinionFieldSpec(
        before_keys=("gun_index", "gun_index_w1"),
        after_keys=("gun_index_2", "gun_index_w2"),
        label="Gun regulation support",
    ),
    "minimum_wage": OpinionFieldSpec(
        before_keys=("mw_index_w1",),
        after_keys=("mw_index_w2",),
        label="Minimum wage support",
    ),
}


def _natural_sort_key(value: str) -> Tuple[int, str, str]:
    """Return a tuple suitable for natural sorting of identifiers with numerics."""

    prefix = "".join(ch for ch in value if not ch.isdigit())
    digits = "".join(ch for ch in value if ch.isdigit())
    return (int(digits) if digits else math.inf, prefix, value)


def _first_numeric(sources: Sequence[Mapping[str, object]], keys: Sequence[str]) -> Optional[float]:
    """Return the first numeric value located across ``keys`` and ``sources``."""

    for key in keys:
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            if key in source:
                numeric = float_or_none(source[key])
                if numeric is not None:
                    return numeric
    return None


def _format_decimal(value: float) -> str:
    """Format ``value`` with up to two decimal places, trimming trailing zeros."""

    formatted = f"{value:.2f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _opinion_label(
    base_label: str,
    stage: str,
    value: Optional[float],
    *,
    delta: Optional[float] = None,
) -> Optional[str]:
    """Construct a human-readable label for an opinion score stage."""

    if value is None:
        return None
    label = f"{stage} {base_label}: {_format_decimal(value)}"
    if delta is not None and abs(delta) >= 0.005:
        signed_delta = f"+{_format_decimal(delta)}" if delta > 0 else _format_decimal(delta)
        label = f"{label} ({signed_delta})"
    return label


def _extract_opinion_annotation(
    rows: Sequence[Mapping[str, object]],
) -> Optional[OpinionAnnotation]:
    """Resolve opinion indices for the viewer represented by ``rows``."""

    if not rows:
        return None
    normalized_issue = ""
    for row in rows:
        issue_value = str(row.get("issue") or "").strip().lower()
        if issue_value:
            normalized_issue = issue_value
            break
    if not normalized_issue:
        return None
    spec = _OPINION_FIELD_MAP.get(normalized_issue)
    if spec is None:
        return None

    before_value: Optional[float] = None
    after_value: Optional[float] = None
    for row in rows:
        sources: List[Mapping[str, object]] = [row]
        selected = row.get("selected_survey_row")
        if isinstance(selected, Mapping):
            sources.append(selected)
        if before_value is None:
            before_value = _first_numeric(sources, spec.before_keys)
        if after_value is None:
            after_value = _first_numeric(sources, spec.after_keys)
        if before_value is not None and after_value is not None:
            break
    if before_value is None and after_value is None:
        return None
    return OpinionAnnotation(
        issue=normalized_issue,
        before_value=before_value,
        after_value=after_value,
        label=spec.label,
    )


def format_node_label(
    node_id: str,
    *,
    node_data: Mapping[str, object],
    options: LabelRenderOptions,
) -> str:
    """Render a node label based on the configured template."""

    context = SafeDict({"id": node_id, **options.metadata.get(node_id, {}), **node_data})
    try:
        label = options.template.format_map(context).strip()
    except KeyError:
        label = ""
    if not label:
        label = str(context.get("originTitle") or context.get("title") or node_id)
    label = _clean_repeated_quotes(label)
    label = _wrap_text(label, options.wrap_width)
    if (
        options.append_id_if_missing
        and "{id}" not in options.template
        and node_id not in label
    ):
        label = f"{label}\n({node_id})"
    return label


__all__ = [
    "LabelRenderOptions",
    "DEFAULT_LABEL_TEMPLATE",
    "OpinionAnnotation",
    "OpinionFieldSpec",
    "SESSION_DEFAULT_LABEL_TEMPLATE",
    "SafeDict",
    "TreeData",
    "TreeEdge",
    "_extract_opinion_annotation",
    "_natural_sort_key",
    "_opinion_label",
    "format_node_label",
]
