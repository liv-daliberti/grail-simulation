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
        """Best-effort local fallback mirroring the upstream helper.

        :param value: Input that may represent a floating-point number.
        :returns: Parsed ``float`` when possible, otherwise ``None``.
        """

        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _wrap_text(text: str, width: Optional[int]) -> str:
    """Wrap text to the specified width using newline separators.

    :param text: Input string to wrap.
    :param width: Maximum characters per line; set ``None`` or ``<= 0`` to disable.
    :returns: Wrapped text with newlines inserted between lines.
    """

    if not width or width <= 0:
        return text
    return "\n".join(textwrap.wrap(text, width))


_QUOTE_RUN_RE = re.compile(r'"{2,}')


def _clean_repeated_quotes(text: str) -> str:
    """Collapse repeated double-quote runs to a single character.

    :param text: Raw text that may contain repeated double quotes.
    :returns: Text with consecutive double quotes replaced by a single quote.
    """

    return _QUOTE_RUN_RE.sub('"', text)


@dataclass
class TreeEdge:
    """Directed edge connecting two nodes in the recommendation tree.

    Attributes:
        parent: Identifier of the parent node.
        child: Identifier of the child node.
        rank: Optional recommendation rank for the edge.
    """

    parent: str
    child: str
    rank: Optional[int] = None


@dataclass
class TreeData:
    """Container holding recommendation tree nodes and edges.

    Attributes:
        root: Identifier of the inferred root node.
        nodes: Mapping from node id to node attribute mappings.
        edges: List of directed edges connecting nodes.
    """

    root: str
    nodes: Dict[str, Mapping[str, object]]
    edges: List[TreeEdge]


@dataclass(frozen=True)
class OpinionFieldSpec:
    """Specification describing opinion score columns used by a viewer cohort.

    Attributes:
        before_keys: Ordered keys that may contain the initial opinion value.
        after_keys: Ordered keys that may contain the final opinion value.
        label: Human-friendly label for the opinion dimension.
    """

    before_keys: Tuple[str, ...]
    after_keys: Tuple[str, ...]
    label: str


@dataclass
class OpinionAnnotation:
    """Resolved opinion values ready for annotation on a session graph.

    Attributes:
        issue: Normalised issue identifier (e.g. ``minimum_wage``).
        before_value: Initial opinion score, if available.
        after_value: Final opinion score, if available.
        label: Human-friendly label for the opinion dimension.
    """

    issue: str
    before_value: Optional[float]
    after_value: Optional[float]
    label: str


@dataclass(frozen=True)
class LabelRenderOptions:
    """Configuration controlling how node labels are rendered.

    Attributes:
        metadata: External per-node metadata keyed by identifier.
        template: Python ``str.format`` template string for labels.
        wrap_width: Optional maximum characters per line for wrapping.
        append_id_if_missing: Whether to append the node id if the template omits it.
    """

    metadata: Mapping[str, Mapping[str, object]]
    template: str
    wrap_width: Optional[int]
    append_id_if_missing: bool = True


class SafeDict(dict):
    """Dictionary that returns placeholder values for missing template keys."""

    def __missing__(self, key: str) -> str:
        """Provide a default empty string for missing keys.

        :param key: Missing mapping key requested by the formatter.
        :returns: Empty string used as a safe placeholder.
        """

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
    """Return a tuple suitable for natural sorting of identifiers with numerics.

    :param value: Identifier to split into numeric and non-numeric components.
    :returns: Tuple ``(numeric_component_or_inf, alpha_prefix, original)`` used for sorting.
    """

    prefix = "".join(ch for ch in value if not ch.isdigit())
    digits = "".join(ch for ch in value if ch.isdigit())
    return (int(digits) if digits else math.inf, prefix, value)


def _first_numeric(
    sources: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> Optional[float]:
    """Return the first numeric value located across keys and sources.

    :param sources: Sequence of mapping-like objects to probe for values.
    :param keys: Ordered keys to try across each mapping in ``sources``.
    :returns: First successfully parsed float value, otherwise ``None``.
    """

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
    """Format a float with up to two decimal places, trimming trailing zeros.

    :param value: Numeric value to format.
    :returns: String representation with at most two decimal places.
    """

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
    """Construct a human-readable label for an opinion score stage.

    :param base_label: Base label describing the opinion dimension (e.g. policy area).
    :param stage: Stage descriptor (e.g. ``"Initial"`` or ``"Final"``).
    :param value: Opinion score at the given stage.
    :param delta: Optional change between final and initial values.
    :returns: Rendered label string or ``None`` when no value is available.
    """

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
    """Resolve opinion indices for the viewer represented by the provided rows.

    :param rows: Ordered or unordered session records containing opinion fields.
    :returns: :class:`OpinionAnnotation` with resolved values or ``None`` if unavailable.
    """

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
    """Render a node label based on the configured template.

    :param node_id: Identifier of the node being labeled.
    :param node_data: Mapping of node attributes available to the template.
    :param options: Label rendering options, including metadata and template string.
    :returns: Final label string suitable for Graphviz node attributes.
    """

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
