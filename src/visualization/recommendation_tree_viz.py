"""Render Guns and GRAIL recommendation trees as Graphviz diagrams."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from graphviz import Digraph


@dataclass
class TreeEdge:
    parent: str
    child: str
    rank: Optional[int] = None


@dataclass
class TreeData:
    root: str
    nodes: Dict[str, Mapping[str, object]]
    edges: List[TreeEdge]


class SafeDict(dict):
    """Dictionary that returns an empty string for missing keys."""

    def __missing__(self, key: str) -> str:
        return ""


def _natural_sort_key(value: str) -> Tuple[int, str, str]:
    prefix = "".join(ch for ch in value if not ch.isdigit())
    digits = "".join(ch for ch in value if ch.isdigit())
    return (int(digits) if digits else math.inf, prefix, value)


def _wrap_text(text: str, width: Optional[int]) -> str:
    if not width or width <= 0:
        return text
    import textwrap

    return "\n".join(textwrap.wrap(text, width))


def load_tree_csv(
    csv_path: Path,
    *,
    id_column: str = "originId",
    child_prefixes: Sequence[str] = ("rec",),
) -> TreeData:
    df = pd.read_csv(csv_path)
    normalized_cols = {col: col.lower() for col in df.columns}
    if id_column not in df.columns:
        lowered = id_column.lower()
        for original, lower in normalized_cols.items():
            if lower == lowered:
                id_column = original
                break
        else:
            id_column = df.columns[0]
    children_cols: List[str] = []
    for col in df.columns:
        col_lower = col.lower()
        for prefix in child_prefixes:
            if col_lower.startswith(prefix.lower()):
                children_cols.append(col)
                break
    if not children_cols:
        raise ValueError(
            "Could not identify recommendation columns. "
            "Use --child-prefixes to point to the recommendation columns."
        )
    children_cols.sort(key=_natural_sort_key)
    nodes: Dict[str, Mapping[str, object]] = {}
    edges: List[TreeEdge] = []
    seen_children = set()
    parent_order: List[str] = []
    for _, row in df.iterrows():
        parent = str(row[id_column])
        parent_order.append(parent)
        nodes.setdefault(parent, row.to_dict())
        for rank, child_col in enumerate(children_cols, start=1):
            child_val = row[child_col]
            if pd.isna(child_val) or child_val == "":
                continue
            child = str(child_val)
            seen_children.add(child)
            edges.append(TreeEdge(parent=parent, child=child, rank=rank))
            if child not in nodes:
                nodes[child] = {}
    roots = [node for node in parent_order if node not in seen_children]
    root = roots[0] if roots else parent_order[0]
    return TreeData(root=root, nodes=nodes, edges=edges)


def load_metadata(
    metadata_path: Optional[Path],
    *,
    id_column: str = "originId",
) -> Dict[str, Mapping[str, object]]:
    if metadata_path is None:
        return {}
    suffix = metadata_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        records: List[Mapping[str, object]] = []
        if suffix == ".jsonl":
            with metadata_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, Mapping):
                        records.append(obj)
        else:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(data, Mapping):
                records = list(data.values())
            elif isinstance(data, list):
                records = data
        lookup: Dict[str, Mapping[str, object]] = {}
        for row in records:
            if id_column in row:
                lookup[str(row[id_column])] = row
        return lookup
    df = pd.read_csv(metadata_path)
    if id_column not in df.columns:
        raise ValueError(
            f"Metadata file {metadata_path} is missing the identifier column '{id_column}'."
        )
    result: Dict[str, Mapping[str, object]] = {}
    for _, row in df.iterrows():
        result[str(row[id_column])] = row.to_dict()
    return result


def format_node_label(
    node_id: str,
    *,
    node_data: Mapping[str, object],
    metadata: Mapping[str, Mapping[str, object]],
    template: str,
    wrap_width: Optional[int],
) -> str:
    context = SafeDict({"id": node_id, **metadata.get(node_id, {}), **node_data})
    try:
        label = template.format_map(context).strip()
    except KeyError:
        label = ""
    if not label:
        label = str(context.get("originTitle") or context.get("title") or node_id)
    label = _wrap_text(label, wrap_width)
    if "{id}" not in template and node_id not in label:
        label = f"{label}\n({node_id})"
    return label


def _extract_sequences_from_object(obj: object) -> List[List[str]]:
    if isinstance(obj, list):
        return [
            [str(item) for item in seq if str(item)]
            for seq in obj
            if isinstance(seq, (list, tuple))
        ]
    if isinstance(obj, Mapping):
        for key in ("videos", "trajectory", "trajectories", "sequence", "sequences"):
            if key in obj:
                value = obj[key]
                if isinstance(value, (list, tuple)):
                    return _extract_sequences_from_object(value)
        return [[str(value) for value in obj.values() if str(value)]]
    if isinstance(obj, str):
        return [[token.strip() for token in obj.split(",") if token.strip()]]
    return []


def load_trajectories(
    path: Optional[Path],
    *,
    delimiter: str = ",",
) -> List[List[str]]:
    if path is None:
        return []
    suffix = path.suffix.lower()
    sequences: List[List[str]] = []
    if suffix in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as handle:
            if suffix == ".jsonl":
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    sequences.extend(_extract_sequences_from_object(obj))
            else:
                data = json.load(handle)
                if isinstance(data, list):
                    for obj in data:
                        sequences.extend(_extract_sequences_from_object(obj))
                elif isinstance(data, Mapping):
                    sequences.extend(_extract_sequences_from_object(data))
    else:
        with path.open("r", encoding="utf-8") as handle:
            if suffix in {".csv", ".tsv", ".txt"}:
                dialect = csv.excel
                if suffix == ".tsv":
                    dialect = csv.excel_tab
                reader = csv.reader(handle, dialect=dialect)
                for row in reader:
                    tokens = [token.strip() for token in row if token.strip()]
                    if tokens:
                        sequences.append(tokens)
            else:
                for line in handle:
                    line = line.strip()
                    if line:
                        parts = [segment.strip() for segment in line.split(delimiter) if segment.strip()]
                        if parts:
                            sequences.append(parts)
    normalized: List[List[str]] = []
    for seq in sequences:
        current: List[str] = []
        for item in seq:
            if item is None or item == "" or (isinstance(item, float) and math.isnan(item)):
                continue
            item_str = str(item)
            if current and current[-1] == item_str:
                continue
            current.append(item_str)
        if current:
            normalized.append(current)
    return normalized


def compute_depths(tree: TreeData) -> Dict[str, int]:
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
    depths = compute_depths(tree)
    highlight_nodes = set(highlight_path)
    highlight_edges = set(zip(highlight_path, highlight_path[1:]))
    graph = Digraph(engine=engine)
    graph.attr(rankdir=rankdir)
    graph.attr("node", shape="box", style="rounded,filled", fillcolor="white", color="#4c566a", fontname="Helvetica")
    graph.attr("edge", color="#4c566a", arrowsize="0.7", fontname="Helvetica")
    allowed_nodes = set(tree.nodes)
    if max_depth is not None:
        allowed_nodes = {node for node, depth in depths.items() if depth <= max_depth}
    for node_id in tree.nodes:
        if max_depth is not None and node_id not in allowed_nodes:
            continue
        label = format_node_label(
            node_id,
            node_data=tree.nodes.get(node_id, {}),
            metadata=metadata,
            template=label_template,
            wrap_width=wrap_width,
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True, help="Path to a tree CSV file.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path (extension selects the format, e.g. .png or .svg).",
    )
    parser.add_argument(
        "--id-column",
        default="originId",
        help="Column that holds the video identifier inside the tree CSV.",
    )
    parser.add_argument(
        "--child-prefixes",
        default="rec",
        help="Comma separated prefixes used to detect recommendation columns.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional metadata CSV/JSON with extra information for labels.",
    )
    parser.add_argument(
        "--metadata-id-column",
        default="originId",
        help="Identifier column inside the metadata file.",
    )
    parser.add_argument(
        "--label-template",
        default="{originTitle}\n{id}",
        help=(
            "Python format string used to render node labels. "
            "Available fields include any column from the tree CSV or metadata."
        ),
    )
    parser.add_argument(
        "--wrap-width",
        type=int,
        default=30,
        help="Wrap labels to this many characters per line (set 0 to disable).",
    )
    parser.add_argument(
        "--highlight",
        help="Comma separated list of video IDs describing a viewer path to highlight.",
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        help=(
            "Optional path to trajectories (CSV, TXT, JSON, JSONL). "
            "Each row or entry should describe a sequence of video IDs."
        ),
    )
    parser.add_argument(
        "--trajectory-delimiter",
        default=",",
        help="Delimiter used when parsing plain-text trajectory files.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Limit the depth of the rendered tree (root depth = 0).",
    )
    parser.add_argument(
        "--rankdir",
        default="LR",
        choices=["TB", "LR", "BT", "RL"],
        help="Graph orientation (Graphviz rankdir).",
    )
    parser.add_argument(
        "--engine",
        default="dot",
        help="Graphviz layout engine (dot, neato, fdp, etc.).",
    )
    parser.add_argument(
        "--hide-rank-labels",
        action="store_true",
        help="Do not annotate edges with their recommendation rank.",
    )
    parser.add_argument(
        "--format",
        help="Override the Graphviz output format (otherwise inferred from --output).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    child_prefixes = tuple(prefix.strip() for prefix in args.child_prefixes.split(",") if prefix.strip())
    tree = load_tree_csv(args.tree, id_column=args.id_column, child_prefixes=child_prefixes)
    metadata = load_metadata(args.metadata, id_column=args.metadata_id_column)
    sequences = load_trajectories(args.trajectories, delimiter=args.trajectory_delimiter)
    if sequences:
        node_counts, edge_counts = aggregate_counts(sequences)
    else:
        node_counts, edge_counts = Counter(), Counter()
    highlight_path: List[str] = []
    if args.highlight:
        highlight_path = [token.strip() for token in args.highlight.split(",") if token.strip()]
    graph = build_graph(
        tree,
        metadata=metadata,
        label_template=args.label_template,
        wrap_width=args.wrap_width,
        highlight_path=highlight_path,
        node_counts=node_counts,
        edge_counts=edge_counts,
        max_depth=args.max_depth,
        rankdir=args.rankdir,
        engine=args.engine,
        show_rank_labels=not args.hide_rank_labels,
    )
    output_format = args.format or args.output.suffix.lstrip(".")
    if not output_format:
        output_format = "png"
    graph.format = output_format
    graph.render(filename=str(args.output), cleanup=True)


if __name__ == "__main__":
    main()
