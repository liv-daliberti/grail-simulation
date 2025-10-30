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

"""Rendering helpers for recommendation tree visualisations."""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import (
    LabelRenderOptions,
    OpinionAnnotation,
    TreeData,
    _extract_opinion_annotation,
    _opinion_label,
    format_node_label,
)

try:  # pragma: no cover - optional dependency
    from graphviz import Digraph
except ImportError:  # pragma: no cover - optional dependency
    Digraph = None  # type: ignore[assignment]


def _require_graphviz() -> "Digraph":  # type: ignore[override]
    """Return the graphviz ``Digraph`` class when the dependency is available."""

    if Digraph is None:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "graphviz must be installed to use the recommendation tree visualisations "
            "(pip install graphviz)."
        )
    return Digraph


def compute_depths(tree: TreeData) -> Dict[str, int]:
    """Compute the depth (distance from the root) for every node in ``tree``."""

    adjacency: Dict[str, List[str]] = {}
    for edge in tree.edges:
        adjacency.setdefault(edge.parent, []).append(edge.child)
    depths: Dict[str, int] = {tree.root: 0}
    queue: deque[str] = deque([tree.root])
    while queue:
        node = queue.popleft()
        for child in adjacency.get(node, []):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)
    return depths


def aggregate_counts(sequences: Iterable[Sequence[str]]) -> Tuple[Counter, Counter]:
    """Aggregate node and edge visitation counts from viewer trajectories."""

    node_counts: Counter = Counter()
    edge_counts: Counter = Counter()
    for seq in sequences:
        if not seq:
            continue
        node_counts[seq[0]] += 1
        for parent, child in zip(seq, seq[1:]):
            node_counts[child] += 1
            edge_counts[(parent, child)] += 1
    return node_counts, edge_counts


def _find_watch_details(row: Mapping[str, object], video_id: str) -> Mapping[str, object]:
    """Find the watch metadata for a given video inside a session row."""

    entries = row.get("watched_detailed_json")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping):
                entry_id = str(entry.get("id") or entry.get("raw_id") or "")
                if entry_id == video_id:
                    return entry
    return {}


def build_graph(
    tree: TreeData,
    *,
    metadata: Mapping[str, Mapping[str, object]],
    label_template: str,
    wrap_width: Optional[int],
    highlight_path: Sequence[str],
    node_counts: Counter,
    edge_counts: Counter,
    max_depth: Optional[int],
    rankdir: str,
    engine: str,
    show_rank_labels: bool,
) -> Digraph:
    """Render a recommendation tree as a Graphviz graph."""

    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    depths = compute_depths(tree)
    highlight_nodes = set(highlight_path)
    highlight_edges = set(zip(highlight_path, highlight_path[1:]))
    graph_cls = _require_graphviz()
    graph = graph_cls(engine=engine)
    graph.attr(rankdir=rankdir)
    graph.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="white",
        color="#4c566a",
        fontname="Helvetica",
    )
    graph.attr(
        "edge",
        color="#4c566a",
        arrowsize="0.7",
        fontname="Helvetica",
    )
    allowed_nodes = set(tree.nodes)
    if max_depth is not None:
        allowed_nodes = {node for node, depth in depths.items() if depth <= max_depth}
    label_options = LabelRenderOptions(
        metadata=metadata,
        template=label_template,
        wrap_width=wrap_width,
    )
    for node_id in tree.nodes:
        if max_depth is not None and node_id not in allowed_nodes:
            continue
        label = format_node_label(
            node_id,
            node_data=tree.nodes.get(node_id, {}),
            options=label_options,
        )
        node_attrs = {}
        if node_id in highlight_nodes:
            node_attrs["fillcolor"] = "#81a1c1"
            node_attrs["color"] = "#2e3440"
            node_attrs["style"] = "rounded,filled,bold"
        count = node_counts.get(node_id)
        if count:
            node_attrs["xlabel"] = str(count)
            node_attrs["labelloc"] = "c"
        graph.node(node_id, label=label, **node_attrs)
    for edge in tree.edges:
        if max_depth is not None:
            parent_depth = depths.get(edge.parent)
            child_depth = depths.get(edge.child)
            if parent_depth is None or child_depth is None:
                continue
            if parent_depth > max_depth - 1 or child_depth > max_depth:
                continue
        edge_attrs = {}
        label_parts: List[str] = []
        if show_rank_labels and edge.rank is not None:
            label_parts.append(f"Rec {edge.rank}")
        count = edge_counts.get((edge.parent, edge.child))
        if count:
            label_parts.append(f"{count} viewers")
            edge_attrs["penwidth"] = str(1.5 + min(4, math.log2(count + 1)))
        if (edge.parent, edge.child) in highlight_edges:
            edge_attrs["color"] = "#bf616a"
            edge_attrs["penwidth"] = "3"
        if label_parts:
            edge_attrs["label"] = "\n".join(label_parts)
        graph.edge(edge.parent, edge.child, **edge_attrs)
    return graph


def build_session_graph(
    rows: Sequence[Mapping[str, object]],
    *,
    label_template: str,
    wrap_width: Optional[int],
    rankdir: str,
    engine: str,
    highlight_path: Sequence[str],
) -> Digraph:
    """Visualise a single viewer session as a Graphviz graph."""

    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    graph_cls = _require_graphviz()
    graph = graph_cls(comment="Viewer session", engine=engine, format="png")
    graph.attr(rankdir=rankdir)
    highlight_set = {vid.strip() for vid in highlight_path if vid.strip()}
    metadata: Dict[str, Mapping[str, object]] = {}

    label_options = LabelRenderOptions(
        metadata=metadata,
        template=label_template,
        wrap_width=wrap_width,
        append_id_if_missing=False,
    )

    ordered_rows: List[Mapping[str, object]] = sorted(
        rows, key=lambda r: (r.get("display_step") or r.get("step_index") or 0)
    )

    opinion_annotation: Optional[OpinionAnnotation] = _extract_opinion_annotation(ordered_rows)
    initial_opinion_node: Optional[str] = None
    final_opinion_node: Optional[str] = None
    initial_label: Optional[str] = None
    final_label: Optional[str] = None
    if opinion_annotation:
        delta = None
        if (
            opinion_annotation.before_value is not None
            and opinion_annotation.after_value is not None
        ):
            delta = opinion_annotation.after_value - opinion_annotation.before_value
        initial_label = _opinion_label(
            opinion_annotation.label,
            "Initial",
            opinion_annotation.before_value,
        )
        final_label = _opinion_label(
            opinion_annotation.label,
            "Final",
            opinion_annotation.after_value,
            delta=delta,
        )
        if initial_label:
            initial_opinion_node = "opinion_initial"
            graph.node(
                initial_opinion_node,
                label=initial_label,
                shape="box",
                style="filled,bold",
                fillcolor="#fbeaea",
                color="#bf616a",
                fontname="Helvetica",
            )
        if final_label:
            final_opinion_node = "opinion_final"
            graph.node(
                final_opinion_node,
                label=final_label,
                shape="box",
                style="filled,bold",
                fillcolor="#fbeaea",
                color="#bf616a",
                fontname="Helvetica",
            )

    current_nodes: Dict[int, str] = {}
    chosen_nodes: Dict[int, str] = {}
    chosen_option_details: Dict[int, Tuple[str, Mapping[str, object]]] = {}

    for row in ordered_rows:
        step = int(row.get("display_step") or row.get("step_index") or (len(current_nodes) + 1))
        current_id = str(row.get("current_video_id") or row.get("current_video_raw_id") or "")
        if not current_id:
            continue
        current_node_key = f"step{step}_current"
        watch_info = _find_watch_details(row, current_id)
        node_data = {
            "id": current_id,
            "originTitle": row.get("current_video_title") or "",
            "title": row.get("current_video_title") or "",
            "channel_title": row.get("current_video_channel") or "",
            "channel": row.get("current_video_channel") or "",
            "watch_seconds": watch_info.get("watch_seconds"),
            "total_length": watch_info.get("total_length"),
            "step_index": row.get("step_index"),
            "display_step": row.get("display_step"),
        }
        current_label = format_node_label(
            current_id,
            node_data=node_data,
            options=label_options,
        )
        node_attrs = {"shape": "ellipse", "style": "filled", "fillcolor": "#eef3ff"}
        if current_id in highlight_set:
            node_attrs.update({"color": "#bf616a", "penwidth": "2"})
        graph.node(current_node_key, label=current_label, **node_attrs)
        current_nodes[step] = current_node_key

        chosen_option_node: Optional[str] = None
        chosen_id = str(row.get("next_video_id") or row.get("next_video_raw_id") or "")
        options = row.get("slate_items_json") or []
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except (json.JSONDecodeError, TypeError):
                options = []

        for rank, item in enumerate(options, start=1):
            if not isinstance(item, Mapping):
                continue
            option_id = str(item.get("id") or item.get("raw_id") or "")
            if not option_id:
                continue
            option_node_key = f"step{step}_opt{rank}"
            option_data = {
                "id": option_id,
                "originTitle": item.get("title") or "",
                "title": item.get("title") or "",
                "channel_title": item.get("channel_title") or "",
                "channel": item.get("channel_title") or "",
                "rank": rank,
            }
            option_label = format_node_label(
                option_id,
                node_data=option_data,
                options=label_options,
            )
            option_attrs = {"shape": "box", "style": "rounded,filled", "fillcolor": "#ffffff"}
            if option_id == chosen_id:
                option_attrs.update({"fillcolor": "#d7f5d0", "penwidth": "2", "color": "#2d9c4a"})
                chosen_option_node = option_node_key
                chosen_option_details[step] = (option_id, dict(option_data))
            elif option_id in highlight_set:
                option_attrs.update({"color": "#bf616a", "penwidth": "2"})
            graph.node(option_node_key, label=option_label, **option_attrs)

            edge_attrs = {"label": str(rank), "color": "#888888", "fontsize": "10"}
            if option_id == chosen_id:
                edge_attrs.update({"color": "#2d9c4a", "penwidth": "2.5"})
            elif option_id in highlight_set or current_id in highlight_set:
                edge_attrs.update({"color": "#bf616a", "penwidth": "2"})
            graph.edge(current_node_key, option_node_key, **edge_attrs)

        if chosen_option_node:
            chosen_nodes[step] = chosen_option_node

    ordered_steps = sorted(current_nodes.keys())
    for idx, step in enumerate(ordered_steps[:-1]):
        next_step = ordered_steps[idx + 1]
        chosen_node = chosen_nodes.get(step)
        next_current_node = current_nodes.get(next_step)
        if chosen_node and next_current_node:
            graph.edge(
                chosen_node,
                next_current_node,
                color="#2d9c4a",
                penwidth="2.5",
                style="dashed",
                label="selected",
                fontsize="10",
            )

    if initial_opinion_node and ordered_steps:
        first_current_node = current_nodes.get(ordered_steps[0])
        if first_current_node:
            graph.edge(
                initial_opinion_node,
                first_current_node,
                color="#bf616a",
                penwidth="2.5",
            )
    terminal_opinion_source: Optional[str] = None
    if ordered_steps:
        last_step = ordered_steps[-1]
        terminal_opinion_source = current_nodes.get(last_step)
        chosen_node = chosen_nodes.get(last_step)
        chosen_detail = chosen_option_details.get(last_step)
        if chosen_node and chosen_detail:
            option_id, option_data = chosen_detail
            final_current_node = f"step{last_step}_selected"
            final_current_label = format_node_label(
                option_id,
                node_data=option_data,
                options=label_options,
            )
            final_current_attrs = {"shape": "ellipse", "style": "filled", "fillcolor": "#eef3ff"}
            if option_id in highlight_set:
                final_current_attrs.update({"color": "#bf616a", "penwidth": "2"})
            graph.node(final_current_node, label=final_current_label, **final_current_attrs)
            graph.edge(
                chosen_node,
                final_current_node,
                color="#2d9c4a",
                penwidth="2.5",
                style="dashed",
                label="selected",
                fontsize="10",
            )
            terminal_opinion_source = final_current_node
    if final_opinion_node and terminal_opinion_source:
        graph.edge(
            terminal_opinion_source,
            final_opinion_node,
            color="#bf616a",
            penwidth="2.5",
        )

    return graph


def render_graph(graph: Digraph, output_path: Path, *, output_format: Optional[str]) -> Path:
    """Render a Graphviz graph to disk, normalising the output path."""

    format_to_use = output_format or output_path.suffix.lstrip(".")
    if not format_to_use:
        format_to_use = "png"
    output_stem = output_path.with_suffix("")
    graph.format = format_to_use
    rendered_path = graph.render(filename=str(output_stem), cleanup=True)
    final_expected = output_stem.with_suffix(f".{format_to_use}")
    if output_path != final_expected and final_expected.exists():
        final_expected.rename(output_path)
        return output_path
    if rendered_path and rendered_path != str(output_path):
        Path(rendered_path).rename(output_path)
    return output_path


PUBLIC_API: Tuple[str, ...] = (
    "_require_graphviz",
    "aggregate_counts",
    "build_graph",
    "build_session_graph",
    "compute_depths",
    "render_graph",
)

__all__ = list(PUBLIC_API)
