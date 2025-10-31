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
from dataclasses import dataclass, field
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
    """Return the graphviz ``Digraph`` class when the dependency is available.

    :raises ImportError: If the ``graphviz`` package is not installed.
    :returns: The :class:`graphviz.Digraph` class for constructing graphs.
    """

    if Digraph is None:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "graphviz must be installed to use the recommendation tree visualisations "
            "(pip install graphviz)."
        )
    return Digraph


@dataclass(frozen=True)
class GraphRenderOptions:
    """Configuration parameters for rendering tree-level graphs."""

    metadata: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    label_template: str = "{originTitle}"
    wrap_width: Optional[int] = None
    highlight_path: Sequence[str] = field(default_factory=tuple)
    node_counts: Mapping[str, int] = field(default_factory=Counter)
    edge_counts: Mapping[Tuple[str, str], int] = field(default_factory=Counter)
    max_depth: Optional[int] = None
    rankdir: str = "LR"
    engine: str = "dot"
    show_rank_labels: bool = True


@dataclass(frozen=True)
class SessionGraphOptions:
    """Configuration parameters for rendering viewer session graphs."""

    label_template: str = "{originTitle}"
    wrap_width: Optional[int] = None
    rankdir: str = "LR"
    engine: str = "dot"
    highlight_path: Sequence[str] = field(default_factory=tuple)


@dataclass
class OpinionNodes:
    """Container describing the optional opinion annotation nodes."""

    initial_node: Optional[str] = None
    final_node: Optional[str] = None


@dataclass
class SessionGraphState:
    """Mutable state accumulated while wiring the per-step session graph."""

    current_nodes: Dict[int, str] = field(default_factory=dict)
    chosen_nodes: Dict[int, str] = field(default_factory=dict)
    chosen_option_details: Dict[int, Tuple[str, Mapping[str, object]]] = field(
        default_factory=dict
    )

    def ordered_steps(self) -> List[int]:
        """Return the step indices observed across the session in order."""

        return sorted(self.current_nodes.keys())


def compute_depths(tree: TreeData) -> Dict[str, int]:
    """Compute the depth (distance from the root) for every node in a tree.

    :param tree: Parsed recommendation tree data structure.
    :returns: Mapping of node identifiers to their distance from the root.
    """

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
    """Aggregate node and edge visitation counts from viewer trajectories.

    :param sequences: Iterable of viewer identifier sequences.
    :returns: Tuple ``(node_counts, edge_counts)`` with visit tallies.
    """

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
    """Return watch metadata extracted from a session row for a video.

    :param row: Session entry produced by the cleaned dataset.
    :param video_id: Identifier of the video being inspected.
    :returns: Mapping with watch details when available, otherwise an empty dict.
    """

    entries = row.get("watched_detailed_json")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping):
                entry_id = str(entry.get("id") or entry.get("raw_id") or "")
                if entry_id == video_id:
                    return entry
    return {}


def build_graph(tree: TreeData, options: GraphRenderOptions) -> Digraph:
    """Render a recommendation tree as a Graphviz graph.

    :param tree: Structured tree data assembled from CSV exports.
    :param options: Rendering configuration controlling style and filtering.
    :returns: A Graphviz graph ready to be rendered to disk.
    """

    depths = compute_depths(tree)
    highlight_nodes, highlight_edges = _highlight_sets(options.highlight_path)
    allowed_nodes = _determine_allowed_nodes(depths, options.max_depth)
    graph = _create_tree_graph(options.engine, options.rankdir)
    label_options = LabelRenderOptions(
        metadata=options.metadata,
        template=options.label_template,
        wrap_width=options.wrap_width,
    )
    _add_tree_nodes(
        graph,
        tree,
        label_options,
        highlight_nodes,
        allowed_nodes,
        options.node_counts,
    )
    _add_tree_edges(
        graph,
        tree,
        depths,
        highlight_edges,
        options.edge_counts,
        options.show_rank_labels,
        options.max_depth,
    )
    return graph


def _create_tree_graph(engine: str, rankdir: str) -> Digraph:
    """Initialise a Graphviz graph with default styling for tree renders.

    :param engine: Graphviz layout engine name (``dot``, ``neato``, etc.).
    :param rankdir: Graph orientation (``LR``, ``TB`` and friends).
    :returns: A configured :class:`graphviz.Digraph` instance.
    """

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
    return graph


def _highlight_sets(
    highlight_path: Sequence[str],
) -> Tuple[set[str], set[Tuple[str, str]]]:
    """Compute node and edge highlight sets from a viewer path.

    :param highlight_path: Sequence describing a preferred viewer journey.
    :returns: Pair ``(highlight_nodes, highlight_edges)`` for styling.
    """

    cleaned = [str(token).strip() for token in highlight_path if str(token).strip()]
    highlight_nodes = set(cleaned)
    highlight_edges = set(zip(cleaned, cleaned[1:]))
    return highlight_nodes, highlight_edges


def _determine_allowed_nodes(
    depths: Mapping[str, int],
    max_depth: Optional[int],
) -> Optional[set[str]]:
    """Return the nodes that are allowed under a ``max_depth`` limit.

    :param depths: Mapping from node identifier to depth.
    :param max_depth: Optional maximum depth; ``None`` keeps every node.
    :returns: Set of permitted node identifiers or ``None`` when unbounded.
    """

    if max_depth is None:
        return None
    return {node for node, depth in depths.items() if depth <= max_depth}


def _add_tree_nodes(
    graph: Digraph,
    tree: TreeData,
    label_options: LabelRenderOptions,
    highlight_nodes: set[str],
    allowed_nodes: Optional[set[str]],
    node_counts: Mapping[str, int],
) -> None:
    """Add nodes and their attributes to the rendered tree graph.

    :param graph: Graphviz graph being constructed.
    :param tree: Tree data describing nodes and edges.
    :param label_options: Formatting options for node labels.
    :param highlight_nodes: Nodes that should be emphasised.
    :param allowed_nodes: Optional set restricting nodes to render.
    :param node_counts: Visit counts to surface alongside nodes.
    """

    for node_id, node_data in tree.nodes.items():
        if allowed_nodes is not None and node_id not in allowed_nodes:
            continue
        label = format_node_label(node_id, node_data=node_data, options=label_options)
        node_attrs = {}
        if node_id in highlight_nodes:
            node_attrs.update(
                {"fillcolor": "#81a1c1", "color": "#2e3440", "style": "rounded,filled,bold"}
            )
        count = node_counts.get(node_id)
        if count:
            node_attrs["xlabel"] = str(count)
            node_attrs["labelloc"] = "c"
        graph.node(node_id, label=label, **node_attrs)


def _add_tree_edges(
    graph: Digraph,
    tree: TreeData,
    depths: Mapping[str, int],
    highlight_edges: set[Tuple[str, str]],
    edge_counts: Mapping[Tuple[str, str], int],
    show_rank_labels: bool,
    max_depth: Optional[int],
) -> None:
    """Add edges and associated styling to the rendered tree graph.

    :param graph: Graphviz graph being constructed.
    :param tree: Tree data describing parent-child relationships.
    :param depths: Node depth mapping used when pruning by ``max_depth``.
    :param highlight_edges: Edges that should be emphasised.
    :param edge_counts: Viewer counts for each traversed edge.
    :param show_rank_labels: Whether to include rank labels on edges.
    :param max_depth: Optional depth cut-off for pruning.
    """

    for edge in tree.edges:
        if max_depth is not None:
            parent_depth = depths.get(edge.parent)
            child_depth = depths.get(edge.child)
            if parent_depth is None or child_depth is None:
                continue
            if parent_depth > max_depth - 1 or child_depth > max_depth:
                continue
        edge_attrs = {}
        labels: List[str] = []
        if show_rank_labels and edge.rank is not None:
            labels.append(f"Rec {edge.rank}")
        count = edge_counts.get((edge.parent, edge.child), 0)
        if count:
            labels.append(f"{count} viewers")
            edge_attrs.setdefault("penwidth", _edge_penwidth(count))
        if (edge.parent, edge.child) in highlight_edges:
            edge_attrs["color"] = "#bf616a"
            edge_attrs["penwidth"] = "3"
        if labels:
            edge_attrs["label"] = "\n".join(labels)
        graph.edge(edge.parent, edge.child, **edge_attrs)


def _edge_penwidth(count: int) -> str:
    """Return a Graphviz penwidth that scales with viewer counts.

    :param count: Number of viewers traversing the edge.
    :returns: String penwidth value suitable for Graphviz attributes.
    """

    return str(1.5 + min(4, math.log2(count + 1)))


def build_session_graph(rows: Sequence[Mapping[str, object]], options: SessionGraphOptions) -> Digraph:
    """Visualise a single viewer session as a Graphviz graph.

    :param rows: Ordered dataset rows describing the viewer session.
    :param options: Rendering configuration controlling styling choices.
    :returns: A Graphviz graph representing the session timeline.
    """

    graph = _create_session_graph(options.engine, options.rankdir)
    highlight_nodes, _ = _highlight_sets(options.highlight_path)
    label_options = _session_label_options(options)
    ordered_rows = _order_session_rows(rows)
    opinion_nodes = _add_opinion_annotations(graph, ordered_rows)
    state = _add_session_steps(graph, ordered_rows, label_options, highlight_nodes)
    _link_session_steps(graph, state)
    _attach_opinion_edges(graph, opinion_nodes, state, label_options, highlight_nodes)
    return graph


def _create_session_graph(engine: str, rankdir: str) -> Digraph:
    """Initialise a Graphviz graph for session-level visualisations.

    :param engine: Graphviz layout engine name.
    :param rankdir: Graph orientation (``LR``, ``TB`` and friends).
    :returns: Configured :class:`graphviz.Digraph` instance.
    """

    graph_cls = _require_graphviz()
    graph = graph_cls(comment="Viewer session", engine=engine, format="png")
    graph.attr(rankdir=rankdir)
    return graph


def _session_label_options(options: SessionGraphOptions) -> LabelRenderOptions:
    """Build label rendering options for session visualisations.

    :param options: Session rendering configuration.
    :returns: Label rendering options tailored for session graphs.
    """

    return LabelRenderOptions(
        metadata={},
        template=options.label_template,
        wrap_width=options.wrap_width,
        append_id_if_missing=False,
    )


def _order_session_rows(rows: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    """Return session rows sorted by display and step indices.

    :param rows: Raw sequence of session entries.
    :returns: List sorted by ``display_step`` (or ``step_index`` fallback).
    """

    return sorted(rows, key=lambda r: (r.get("display_step") or r.get("step_index") or 0))


def _add_opinion_annotations(
    graph: Digraph,
    rows: Sequence[Mapping[str, object]],
) -> OpinionNodes:
    """Render optional opinion change annotations for the session.

    :param graph: Graphviz graph being constructed.
    :param rows: Ordered session records used to detect opinion shifts.
    :returns: ``OpinionNodes`` describing created annotation nodes.
    """

    opinion_annotation: Optional[OpinionAnnotation] = _extract_opinion_annotation(rows)
    nodes = OpinionNodes()
    if not opinion_annotation:
        return nodes

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
        nodes.initial_node = "opinion_initial"
        graph.node(
            nodes.initial_node,
            label=initial_label,
            shape="box",
            style="filled,bold",
            fillcolor="#fbeaea",
            color="#bf616a",
            fontname="Helvetica",
        )
    if final_label:
        nodes.final_node = "opinion_final"
        graph.node(
            nodes.final_node,
            label=final_label,
            shape="box",
            style="filled,bold",
            fillcolor="#fbeaea",
            color="#bf616a",
            fontname="Helvetica",
        )
    return nodes


def _add_session_steps(
    graph: Digraph,
    rows: Sequence[Mapping[str, object]],
    label_options: LabelRenderOptions,
    highlight_set: set[str],
) -> SessionGraphState:
    """Render session steps and recommendation options for each row.

    :param graph: Graphviz graph being constructed.
    :param rows: Ordered session rows.
    :param label_options: Label configuration for nodes.
    :param highlight_set: Identifiers to highlight within the graph.
    :returns: Mutable state capturing created nodes and selections.
    """

    state = SessionGraphState()
    for row in rows:
        step = int(row.get("display_step") or row.get("step_index") or (len(state.current_nodes) + 1))
        current_id = str(row.get("current_video_id") or row.get("current_video_raw_id") or "")
        if not current_id:
            continue
        current_node_key = _add_current_node(graph, step, current_id, row, label_options, highlight_set)
        state.current_nodes[step] = current_node_key
        chosen_node, chosen_details = _add_option_nodes(
            graph,
            row,
            step,
            current_id,
            current_node_key,
            label_options,
            highlight_set,
        )
        if chosen_node:
            state.chosen_nodes[step] = chosen_node
        if chosen_details:
            state.chosen_option_details[step] = chosen_details
    return state


def _add_current_node(
    graph: Digraph,
    step: int,
    current_id: str,
    row: Mapping[str, object],
    label_options: LabelRenderOptions,
    highlight_set: set[str],
) -> str:
    """Add the node representing the current video at a given step.

    :param graph: Graphviz graph being constructed.
    :param step: Session step index.
    :param current_id: Identifier of the currently watched video.
    :param row: Dataset row describing the step.
    :param label_options: Label configuration for nodes.
    :param highlight_set: Identifiers that should be highlighted.
    :returns: Graphviz node key for the current video.
    """

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
    current_label = format_node_label(current_id, node_data=node_data, options=label_options)
    node_attrs = {"shape": "ellipse", "style": "filled", "fillcolor": "#eef3ff"}
    if current_id in highlight_set:
        node_attrs.update({"color": "#bf616a", "penwidth": "2"})
    node_key = f"step{step}_current"
    graph.node(node_key, label=current_label, **node_attrs)
    return node_key


def _add_option_nodes(
    graph: Digraph,
    row: Mapping[str, object],
    step: int,
    current_id: str,
    current_node_key: str,
    label_options: LabelRenderOptions,
    highlight_set: set[str],
) -> Tuple[Optional[str], Optional[Tuple[str, Mapping[str, object]]]]:
    """Render recommendation options for a session step.

    :param graph: Graphviz graph being constructed.
    :param row: Dataset row describing recommendation options.
    :param step: Session step index.
    :param current_id: Identifier of the current video.
    :param current_node_key: Graphviz node representing the current video.
    :param label_options: Label configuration for option nodes.
    :param highlight_set: Identifiers to highlight within the graph.
    :returns: Tuple of the chosen option node key and its details (if any).
    """

    chosen_id = str(row.get("next_video_id") or row.get("next_video_raw_id") or "")
    options = row.get("slate_items_json") or []
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except (json.JSONDecodeError, TypeError):
            options = []

    chosen_option_node: Optional[str] = None
    chosen_option_details: Optional[Tuple[str, Mapping[str, object]]] = None
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
        option_label = format_node_label(option_id, node_data=option_data, options=label_options)
        option_attrs = {"shape": "box", "style": "rounded,filled", "fillcolor": "#ffffff"}
        if option_id == chosen_id:
            option_attrs.update({"fillcolor": "#d7f5d0", "penwidth": "2", "color": "#2d9c4a"})
            chosen_option_node = option_node_key
            chosen_option_details = (option_id, dict(option_data))
        elif option_id in highlight_set:
            option_attrs.update({"color": "#bf616a", "penwidth": "2"})
        graph.node(option_node_key, label=option_label, **option_attrs)
        edge_attrs = _option_edge_attributes(rank, option_id, chosen_id, current_id, highlight_set)
        graph.edge(current_node_key, option_node_key, **edge_attrs)
    return chosen_option_node, chosen_option_details


def _option_edge_attributes(
    rank: int,
    option_id: str,
    chosen_id: str,
    current_id: str,
    highlight_set: set[str],
) -> Dict[str, str]:
    """Return edge styling attributes for recommendation options.

    :param rank: Rank of the recommendation within the slate.
    :param option_id: Candidate video identifier.
    :param chosen_id: Selected video identifier.
    :param current_id: Currently watched video identifier.
    :param highlight_set: Identifiers to highlight within the graph.
    :returns: Mapping of Graphviz edge attributes.
    """

    edge_attrs: Dict[str, str] = {"label": str(rank), "color": "#888888", "fontsize": "10"}
    if option_id == chosen_id:
        edge_attrs.update({"color": "#2d9c4a", "penwidth": "2.5"})
    elif option_id in highlight_set or current_id in highlight_set:
        edge_attrs.update({"color": "#bf616a", "penwidth": "2"})
    return edge_attrs


def _link_session_steps(graph: Digraph, state: SessionGraphState) -> None:
    """Connect successive session steps via the selected recommendation path.

    :param graph: Graphviz graph being constructed.
    :param state: Mutable state describing current/selected nodes per step.
    """

    ordered_steps = state.ordered_steps()
    for idx, step in enumerate(ordered_steps[:-1]):
        next_step = ordered_steps[idx + 1]
        chosen_node = state.chosen_nodes.get(step)
        next_current_node = state.current_nodes.get(next_step)
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


def _attach_opinion_edges(
    graph: Digraph,
    opinion_nodes: OpinionNodes,
    state: SessionGraphState,
    label_options: LabelRenderOptions,
    highlight_set: set[str],
) -> None:
    """Connect opinion annotation nodes to the session timeline.

    :param graph: Graphviz graph being constructed.
    :param opinion_nodes: Created opinion annotation node identifiers.
    :param state: Mutable state describing session node layout.
    :param label_options: Label configuration for nodes.
    :param highlight_set: Identifiers to highlight within the graph.
    """

    ordered_steps = state.ordered_steps()
    if not ordered_steps:
        return
    if opinion_nodes.initial_node:
        first_current_node = state.current_nodes.get(ordered_steps[0])
        if first_current_node:
            graph.edge(
                opinion_nodes.initial_node,
                first_current_node,
                color="#bf616a",
                penwidth="2.5",
            )
    terminal_source: Optional[str] = state.current_nodes.get(ordered_steps[-1])
    chosen_node = state.chosen_nodes.get(ordered_steps[-1])
    chosen_detail = state.chosen_option_details.get(ordered_steps[-1])
    if chosen_node and chosen_detail:
        terminal_source = _add_terminal_selected_node(
            graph,
            ordered_steps[-1],
            chosen_node,
            chosen_detail,
            label_options,
            highlight_set,
        )
    if opinion_nodes.final_node and terminal_source:
        graph.edge(
            terminal_source,
            opinion_nodes.final_node,
            color="#bf616a",
            penwidth="2.5",
        )


def _add_terminal_selected_node(
    graph: Digraph,
    step: int,
    chosen_node: str,
    chosen_detail: Tuple[str, Mapping[str, object]],
    label_options: LabelRenderOptions,
    highlight_set: set[str],
) -> str:
    """Add a terminal node for the final selected recommendation.

    :param graph: Graphviz graph being constructed.
    :param step: Session step index associated with the final choice.
    :param chosen_node: Graphviz node key for the chosen option.
    :param chosen_detail: Pair of option identifier and metadata mapping.
    :param label_options: Label configuration for the node.
    :param highlight_set: Identifiers to highlight within the graph.
    :returns: Graphviz node key representing the terminal selection.
    """

    option_id, option_data = chosen_detail
    final_current_node = f"step{step}_selected"
    final_current_label = format_node_label(option_id, node_data=option_data, options=label_options)
    final_attrs = {"shape": "ellipse", "style": "filled", "fillcolor": "#eef3ff"}
    if option_id in highlight_set:
        final_attrs.update({"color": "#bf616a", "penwidth": "2"})
    graph.node(final_current_node, label=final_current_label, **final_attrs)
    graph.edge(
        chosen_node,
        final_current_node,
        color="#2d9c4a",
        penwidth="2.5",
        style="dashed",
        label="selected",
        fontsize="10",
    )
    return final_current_node


def render_graph(graph: Digraph, output_path: Path, *, output_format: Optional[str]) -> Path:
    """Render a Graphviz graph to disk, normalising the output path.

    :param graph: Graphviz graph to render.
    :param output_path: Destination path (suffix controls format when unspecified).
    :param output_format: Explicit Graphviz format override.
    :returns: Path to the final rendered artefact.
    """

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
    "GraphRenderOptions",
    "SessionGraphOptions",
    "_require_graphviz",
    "aggregate_counts",
    "build_graph",
    "build_session_graph",
    "compute_depths",
    "render_graph",
)

__all__ = list(PUBLIC_API)
