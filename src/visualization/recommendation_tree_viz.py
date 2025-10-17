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


def _load_cleaned_dataset(path: Path):
    from datasets import load_from_disk, load_dataset  # type: ignore
    if path.is_dir():
        return load_from_disk(str(path))
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(path))
    if suffix in {".csv", ".tsv"}:
        return load_dataset("csv", data_files=str(path), delimiter="," if suffix == ".csv" else "\t")
    raise ValueError(f"Unsupported cleaned dataset format: {path}")


def _collect_rows(dataset, split: Optional[str] = None, issue: Optional[str] = None) -> List[Dict[str, object]]:
    from datasets import DatasetDict  # type: ignore

    rows: List[Dict[str, object]] = []
    if isinstance(dataset, DatasetDict):
        for name, split_ds in dataset.items():
            if split and name != split:
                continue
            for entry in split_ds:
                if issue and str(entry.get("issue") or "").strip() != issue:
                    continue
                rows.append({k: entry[k] for k in entry.keys()})
    else:
        if split and split not in {None, "train"}:
            return []
        for entry in dataset:
            if issue and str(entry.get("issue") or "").strip() != issue:
                continue
            rows.append({k: entry[k] for k in entry.keys()})
    return rows


def _extract_session_rows(
    dataset,
    *,
    session_id: Optional[str],
    split: Optional[str],
    issue: Optional[str],
    max_steps: Optional[int],
) -> Tuple[str, List[Dict[str, object]]]:
    rows = _collect_rows(dataset, split=split, issue=issue)
    if not rows:
        raise ValueError("No rows found in the cleaned dataset with the provided filters.")

    if session_id is None:
        session_id = str(rows[0].get("session_id"))

    session_rows = [row for row in rows if str(row.get("session_id")) == session_id]
    if not session_rows:
        available = sorted({str(row.get("session_id")) for row in rows})
        raise ValueError(f"Session '{session_id}' not found. Available session IDs: {available[:10]}")

    session_rows.sort(key=lambda r: (r.get("step_index") or 0, r.get("display_step") or 0))
    if max_steps is not None and max_steps > 0:
        session_rows = session_rows[:max_steps]
    return session_id, session_rows


def _find_watch_details(row: Mapping[str, object], video_id: str) -> Mapping[str, object]:
    entries = row.get("watched_detailed_json")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping):
                entry_id = str(entry.get("id") or entry.get("raw_id") or "")
                if entry_id == video_id:
                    return entry
    return {}


def build_session_graph(
    rows: Sequence[Mapping[str, object]],
    *,
    label_template: str,
    wrap_width: Optional[int],
    rankdir: str,
    engine: str,
    highlight_path: Sequence[str],
) -> Digraph:
    graph = Digraph(comment="Viewer session", engine=engine, format="png")
    graph.attr(rankdir=rankdir)
    highlight_set = {vid.strip() for vid in highlight_path if vid.strip()}
    metadata: Dict[str, Mapping[str, object]] = {}

    ordered_rows: List[Mapping[str, object]] = sorted(
        rows, key=lambda r: (r.get("display_step") or r.get("step_index") or 0)
    )

    current_nodes: Dict[int, str] = {}
    chosen_nodes: Dict[int, str] = {}

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
            metadata=metadata,
            template=label_template,
            wrap_width=wrap_width,
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
            except Exception:
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
                metadata=metadata,
                template=label_template,
                wrap_width=wrap_width,
            )
            option_attrs = {"shape": "box", "style": "rounded,filled", "fillcolor": "#ffffff"}
            if option_id == chosen_id:
                option_attrs.update({"fillcolor": "#d7f5d0", "penwidth": "2", "color": "#2d9c4a"})
                chosen_option_node = option_node_key
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

    return graph


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, help="Path to a tree CSV file.")
    parser.add_argument(
        "--cleaned-data",
        type=Path,
        help="Path to a cleaned dataset produced by clean_data/clean_data.py (datasets.save_to_disk).",
    )
    parser.add_argument(
        "--session-id",
        help="Session ID to visualize when using --cleaned-data. Defaults to the first session found.",
    )
    parser.add_argument(
        "--split",
        help="Dataset split to read from when using --cleaned-data (defaults to every available split).",
    )
    parser.add_argument(
        "--issue",
        help="Optional issue filter when using --cleaned-data (e.g. gun_control or minimum_wage).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Limit the number of recommendation steps rendered for the selected session.",
    )
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
    args = parser.parse_args(argv)
    if not args.tree and not args.cleaned_data:
        parser.error("Please provide either --tree or --cleaned-data.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    highlight_path: List[str] = []
    if args.highlight:
        highlight_path = [token.strip() for token in args.highlight.split(",") if token.strip()]

    if args.cleaned_data:
        dataset = _load_cleaned_dataset(args.cleaned_data)
        session_id, session_rows = _extract_session_rows(
            dataset,
            session_id=args.session_id,
            split=args.split,
            issue=args.issue,
            max_steps=args.max_steps,
        )
        graph = build_session_graph(
            session_rows,
            label_template=args.label_template,
            wrap_width=args.wrap_width,
            rankdir=args.rankdir,
            engine=args.engine,
            highlight_path=highlight_path,
        )
    else:
        child_prefixes = tuple(prefix.strip() for prefix in args.child_prefixes.split(",") if prefix.strip())
        tree = load_tree_csv(args.tree, id_column=args.id_column, child_prefixes=child_prefixes)
        metadata = load_metadata(args.metadata, id_column=args.metadata_id_column)
        sequences = load_trajectories(args.trajectories, delimiter=args.trajectory_delimiter)
        if sequences:
            node_counts, edge_counts = aggregate_counts(sequences)
        else:
            node_counts, edge_counts = Counter(), Counter()
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
    output_stem = args.output.with_suffix("")
    graph.format = output_format
    rendered_path = graph.render(filename=str(output_stem), cleanup=True)
    final_expected = output_stem.with_suffix(f".{output_format}")
    if args.output != final_expected and final_expected.exists():
        final_expected.rename(args.output)
    elif rendered_path and rendered_path != str(args.output):
        Path(rendered_path).rename(args.output)


if __name__ == "__main__":
    main()
