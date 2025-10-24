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

"""Visualization helpers for rendering Grail recommendation tree diagrams."""

# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
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


@dataclass
class TreeEdge:
    """Directed edge connecting two nodes in the recommendation tree.

    :param parent: Identifier of the upstream node that produced the recommendation.
    :type parent: str
    :param child: Identifier of the downstream node that was recommended.
    :type child: str
    :param rank: One-based recommendation rank assigned by the model, when available.
    :type rank: int | None
    """

    parent: str
    child: str
    rank: Optional[int] = None


@dataclass
class TreeData:
    """Container holding recommendation tree nodes and edges.

    :param root: Identifier of the tree root, typically the entry video.
    :type root: str
    :param nodes: Mapping from node identifiers to their underlying row data.
    :type nodes: dict[str, Mapping[str, object]]
    :param edges: List of directed edges describing parent-child relationships.
    :type edges: list[TreeEdge]
    """

    root: str
    nodes: Dict[str, Mapping[str, object]]
    edges: List[TreeEdge]


class SafeDict(dict):
    """Dictionary that returns placeholder values for missing template keys.

    Instances behave like a regular ``dict`` but ensure format strings never raise a
    :class:`KeyError` by substituting empty strings for absent fields.
    """

    def __missing__(self, key: str) -> str:
        """Return an empty string when ``key`` is absent during template formatting.

        :param key: Lookup key requested by :meth:`str.format_map`.
        :type key: str
        :returns: Empty string to keep format operations resilient.
        :rtype: str
        """
        return ""


def _natural_sort_key(value: str) -> Tuple[int, str, str]:
    """Return a tuple suitable for natural sorting of identifiers with numerics.

    The returned tuple allows call sites to sort values such as ``rec1`` and ``rec10`` in
    numerical order rather than purely lexicographical order.

    :param value: String that may contain interleaved alphabetic and numeric segments.
    :type value: str
    :returns: Comparison tuple of ``(numeric_suffix, alphabetic_prefix, original_value)``.
    :rtype: tuple[int, str, str]
    """
    prefix = "".join(ch for ch in value if not ch.isdigit())
    digits = "".join(ch for ch in value if ch.isdigit())
    return (int(digits) if digits else math.inf, prefix, value)


def _wrap_text(text: str, width: Optional[int]) -> str:
    """Wrap ``text`` to the specified ``width`` using newline separators.

    :param text: Original text block that should be wrapped for readability.
    :type text: str
    :param width: Maximum line length; disables wrapping when ``None`` or ``<= 0``.
    :type width: int | None
    :returns: Wrapped text with newline separators inserted at breakpoints.
    :rtype: str
    """
    if not width or width <= 0:
        return text
    return "\n".join(textwrap.wrap(text, width))


def parse_issue_counts(spec: str) -> Dict[str, int]:
    """Parse comma-separated ``issue=count`` specifications into a mapping.

    The helper is primarily used to interpret ``--batch-issues`` CLI payloads.

    :param spec: Comma-separated string such as ``"minimum_wage=2,gun_control=1"``.
    :type spec: str
    :returns: Normalised mapping from issue identifiers to positive integer counts.
    :rtype: dict[str, int]
    :raises ValueError: If the input string does not follow the ``issue=count`` format.
    """
    result: Dict[str, int] = {}
    if not spec:
        return result
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"Invalid issue specification '{chunk}'. Expected format issue=count."
            )
        issue, count_str = chunk.split("=", 1)
        issue = issue.strip()
        if not issue:
            raise ValueError(f"Missing issue name in specification '{chunk}'.")
        try:
            count = int(count_str)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Invalid count in specification '{chunk}'.") from exc
        if count <= 0:
            raise ValueError(f"Count must be positive in specification '{chunk}'.")
        result[issue] = count
    if not result:
        raise ValueError("Provided issue specification did not contain any pairs.")
    return result


def load_tree_csv(
    csv_path: Path,
    *,
    id_column: str = "originId",
    child_prefixes: Sequence[str] = ("rec",),
) -> TreeData:
    """Load a recommendation tree from a CSV export.

    The CSV is expected to contain a unique identifier column as well as several columns
    whose names share a prefix (for example ``rec1`` ... ``rec5``). Each prefixed column
    is interpreted as a ranked recommendation edge.

    :param csv_path: Path to the CSV file generated by the recommender pipeline.
    :type csv_path: Path
    :param id_column: Column name that stores the node identifier; automatically detected
        in a case-insensitive fashion when not present verbatim.
    :type id_column: str
    :param child_prefixes: One or more column prefixes that denote recommendation targets.
    :type child_prefixes: Sequence[str]
    :returns: Populated :class:`TreeData` describing the recommendations and root node.
    :rtype: TreeData
    :raises ValueError: If no recommendation columns can be inferred from the provided CSV.
    """
    # pylint: disable=too-many-locals
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
    """Load per-node metadata to enrich node labels.

    Metadata files may be authored in CSV, JSON, or JSONL formats. The loader produces a
    dictionary keyed by node identifier, which is merged into the label template context.

    :param metadata_path: Path to the metadata file, or ``None`` to skip enrichment.
    :type metadata_path: Path | None
    :param id_column: Column containing the node identifier within the metadata file.
    :type id_column: str
    :returns: Mapping from node identifiers to metadata dictionaries.
    :rtype: dict[str, Mapping[str, object]]
    :raises ValueError: If the identifier column is missing in a CSV metadata file.
    """
    # pylint: disable=too-many-branches
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
    """Render a node label based on the configured template.

    Labels are constructed from both the raw tree row and any auxiliary metadata. Missing
    template keys are tolerated thanks to :class:`SafeDict`. When the formatted template
    results in an empty string, the function falls back to common title fields.

    :param node_id: Identifier of the node being rendered.
    :type node_id: str
    :param node_data: Raw data associated with the node sourced from the tree CSV.
    :type node_data: Mapping[str, object]
    :param metadata: Supplemental metadata keyed by node identifier.
    :type metadata: Mapping[str, Mapping[str, object]]
    :param template: Python format string used for the label body.
    :type template: str
    :param wrap_width: Optional character width used to wrap the label text.
    :type wrap_width: int | None
    :returns: Formatted label string with reasonable fallbacks and ``wrap_width`` applied.
    :rtype: str
    """
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
    """Extract sequences of identifiers from a heterogeneous object.

    The helper accepts the diverse payloads encountered in real trajectory dumps. Lists,
    tuples, dictionaries, and delimited strings are normalised into lists of string
    identifiers.

    :param obj: Candidate object containing sequence information.
    :type obj: object
    :returns: Collection of identifier sequences discovered within ``obj``.
    :rtype: list[list[str]]
    """
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
    """Load viewer trajectories from text, CSV, or JSON representations.

    The loader gracefully handles repeated identifiers, blank entries, and numeric values,
    returning deduplicated and stringified sequences suitable for downstream aggregation.

    :param path: Path to the trajectory file, or ``None`` to skip loading trajectories.
    :type path: Path | None
    :param delimiter: Delimiter used when parsing free-form text files.
    :type delimiter: str
    :returns: Normalised viewer trajectories as lists of identifiers.
    :rtype: list[list[str]]
    """
    # pylint: disable=too-many-branches,too-many-locals
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
                        parts = [
                            segment.strip()
                            for segment in line.split(delimiter)
                            if segment.strip()
                        ]
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
    """Compute the depth (distance from the root) for every node in ``tree``.

    A breadth-first traversal considers every parent-child relation once and produces a
    zero-based depth for each reachable node.

    :param tree: Recommendation tree structure.
    :type tree: TreeData
    :returns: Mapping from node identifiers to zero-based depth.
    :rtype: dict[str, int]
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

    :param sequences: Iterable of node identifier sequences, each representing a viewer.
    :type sequences: Iterable[Sequence[str]]
    :returns: Tuple of ``(node_counts, edge_counts)`` counters with visit frequencies.
    :rtype: tuple[Counter, Counter]
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


def _group_rows_by_session(
    rows: Sequence[Mapping[str, object]]
) -> Dict[str, List[Mapping[str, object]]]:
    """Group dataset rows by session identifier and sort within each session.

    :param rows: Iterable of dataset rows emitted by the cleaned dataset.
    :type rows: Sequence[Mapping[str, object]]
    :returns: Mapping from session identifiers to chronologically sorted rows.
    :rtype: dict[str, list[Mapping[str, object]]]
    """
    sessions: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        session_id = str(row.get("session_id") or "")
        if not session_id:
            continue
        sessions.setdefault(session_id, []).append(row)
    for session_rows in sessions.values():
        session_rows.sort(key=lambda r: (r.get("step_index") or 0, r.get("display_step") or 0))
    return sessions


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
    """Render a recommendation tree as a Graphviz graph.

    The resulting diagram supports optional highlighting of a user journey, overlays visit
    counts, and displays recommendation rank labels. Layout is controlled entirely through
    Graphviz attributes.

    :param tree: Tree structure to render.
    :type tree: TreeData
    :param metadata: Supplemental metadata per node used by :func:`format_node_label`.
    :type metadata: Mapping[str, Mapping[str, object]]
    :param label_template: Template used to render each node label.
    :type label_template: str
    :param wrap_width: Optional wrapping width for labels; ``None`` disables wrapping.
    :type wrap_width: int | None
    :param highlight_path: Sequence of nodes to emphasise visually.
    :type highlight_path: Sequence[str]
    :param node_counts: Aggregated node view counts sourced from viewer trajectories.
    :type node_counts: Counter
    :param edge_counts: Aggregated edge traversal counts from viewer trajectories.
    :type edge_counts: Counter
    :param max_depth: Optional depth limit controlling which nodes are rendered.
    :type max_depth: int | None
    :param rankdir: Graphviz rank direction (for example ``"LR"`` or ``"TB"``).
    :type rankdir: str
    :param engine: Graphviz layout engine name.
    :type engine: str
    :param show_rank_labels: Whether to annotate edges with their recommendation rank.
    :type show_rank_labels: bool
    :returns: Configured :class:`graphviz.Digraph` instance ready for rendering.
    :rtype: graphviz.Digraph
    """
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
    """Load a HuggingFace dataset exported by ``clean_data.py``.

    The function supports both saved-to-disk datasets as well as flat JSON/CSV exports,
    deferring heavy imports until runtime so command-line usage remains lightweight.

    :param path: Path to the dataset folder or file.
    :type path: Path
    :returns: Dataset object compatible with the HuggingFace ``datasets`` API.
    :rtype: datasets.Dataset | datasets.DatasetDict
    :raises ImportError: If the optional ``datasets`` dependency is not installed.
    :raises ValueError: If the file extension is not supported.
    """
    # pylint: disable=import-outside-toplevel
    try:  # pragma: no cover - optional dependency
        from datasets import load_dataset, load_from_disk  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Loading recommendation tree datasets requires the 'datasets' package. "
            "Install it with `pip install datasets`."
        ) from exc
    if path.is_dir():
        return load_from_disk(str(path))
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(path))
    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        return load_dataset("csv", data_files=str(path), delimiter=delimiter)
    raise ValueError(f"Unsupported cleaned dataset format: {path}")


def _collect_rows(
    dataset,
    split: Optional[str] = None,
    issue: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Collect dataset rows, optionally filtering by split name and issue.

    :param dataset: Dataset object or mapping of split names to datasets.
    :type dataset: datasets.Dataset | datasets.DatasetDict | Mapping[str, object]
    :param split: Optional split name (for example ``"train"`` or ``"validation"``).
    :type split: str | None
    :param issue: Optional issue identifier to filter rows.
    :type issue: str | None
    :returns: Rows materialised as standard dictionaries to simplify downstream use.
    :rtype: list[dict[str, object]]
    """
    rows: List[Dict[str, object]] = []
    dataset_items = None
    if hasattr(dataset, "items") and callable(getattr(dataset, "items")):
        dataset_items = dataset.items()
    else:
        if split and split not in {None, "train"}:
            return []
        dataset_items = [(None, dataset)]
    for name, split_ds in dataset_items:
        if split and name and name != split:
            continue
        for entry in split_ds:
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
    """Extract the rows that correspond to a target viewer session.

    :param dataset: HuggingFace dataset or mapping of splits.
    :type dataset: datasets.Dataset | datasets.DatasetDict | Mapping[str, object]
    :param session_id: Specific session identifier to fetch; defaults to the first available.
    :type session_id: str | None
    :param split: Optional split name for filtering.
    :type split: str | None
    :param issue: Optional issue identifier filter.
    :type issue: str | None
    :param max_steps: Optional maximum number of recommendation steps to include.
    :type max_steps: int | None
    :returns: Tuple of the resolved session identifier and its chronologically ordered rows.
    :rtype: tuple[str, list[dict[str, object]]]
    :raises ValueError: If no matching session can be located with the provided filters.
    """
    rows = _collect_rows(dataset, split=split, issue=issue)
    if not rows:
        raise ValueError("No rows found in the cleaned dataset with the provided filters.")

    if session_id is None:
        session_id = str(rows[0].get("session_id"))

    session_rows = [row for row in rows if str(row.get("session_id")) == session_id]
    if not session_rows:
        available = sorted({str(row.get("session_id")) for row in rows})
        raise ValueError(
            f"Session '{session_id}' not found. Available session IDs: {available[:10]}"
        )

    session_rows.sort(key=lambda r: (r.get("step_index") or 0, r.get("display_step") or 0))
    if max_steps is not None and max_steps > 0:
        session_rows = session_rows[:max_steps]
    return session_id, session_rows


def _find_watch_details(row: Mapping[str, object], video_id: str) -> Mapping[str, object]:
    """Find the watch metadata for a given video inside a session row.

    :param row: Session row with watch details embedded in ``watched_detailed_json``.
    :type row: Mapping[str, object]
    :param video_id: Identifier of the video to search for.
    :type video_id: str
    :returns: Matching watch information, or an empty mapping when not present.
    :rtype: Mapping[str, object]
    """
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
    """Visualise a single viewer session as a Graphviz graph.

    Nodes represent the current video and available recommendations at every step, while
    edges track both recommendation rank and the option ultimately selected by the viewer.

    :param rows: Chronologically ordered session rows.
    :type rows: Sequence[Mapping[str, object]]
    :param label_template: Template used to render node labels.
    :type label_template: str
    :param wrap_width: Optional wrapping width for labels; ``None`` and ``0`` disable wrapping.
    :type wrap_width: int | None
    :param rankdir: Graph orientation encoded using Graphviz's ``rankdir`` attribute.
    :type rankdir: str
    :param engine: Graphviz layout engine to use.
    :type engine: str
    :param highlight_path: Sequence of video identifiers to highlight visually.
    :type highlight_path: Sequence[str]
    :returns: Configured :class:`graphviz.Digraph` instance describing the session.
    :rtype: graphviz.Digraph
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    graph_cls = _require_graphviz()
    graph = graph_cls(comment="Viewer session", engine=engine, format="png")
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


def render_graph(graph: Digraph, output_path: Path, *, output_format: Optional[str]) -> Path:
    """Render a Graphviz graph to disk, normalising the output path.

    The helper mimics Graphviz CLI behaviour by inferring the format from the filename,
    setting :attr:`Digraph.format`, and renaming the intermediate output produced by
    :meth:`graphviz.Digraph.render`.

    :param graph: Graphviz graph to render.
    :type graph: graphviz.Digraph
    :param output_path: Target output file path.
    :type output_path: Path
    :param output_format: Optional format override (otherwise inferred from ``output_path``).
    :type output_format: str | None
    :returns: Path to the rendered output file on disk.
    :rtype: Path
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the recommendation visualiser.

    The parser supports both recommendation-tree mode and session-visualisation mode,
    validating that the correct combination of flags is supplied before returning.

    :param argv: Optional explicit argument vector for testing.
    :type argv: Sequence[str] | None
    :returns: Parsed arguments namespace ready for downstream consumption.
    :rtype: argparse.Namespace
    :raises SystemExit: If required arguments are missing or incompatible.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, help="Path to a tree CSV file.")
    parser.add_argument(
        "--cleaned-data",
        type=Path,
        help=(
            "Path to a cleaned dataset produced by clean_data/clean_data.py "
            "(datasets.save_to_disk)."
        ),
    )
    parser.add_argument(
        "--session-id",
        help=(
            "Session ID to visualize when using --cleaned-data. Defaults to the first "
            "session found."
        ),
    )
    parser.add_argument(
        "--split",
        help=(
            "Dataset split to read from when using --cleaned-data "
            "(defaults to every available split)."
        ),
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
        "--batch-output-dir",
        type=Path,
        help="Directory to emit multiple session visualisations (requires --cleaned-data).",
    )
    parser.add_argument(
        "--batch-issues",
        default="minimum_wage=2,gun_control=2",
        help=(
            "Comma separated list of issue=count pairs when using --batch-output-dir. "
            "Defaults to minimum_wage=2,gun_control=2."
        ),
    )
    parser.add_argument(
        "--batch-prefix",
        default="session",
        help="Filename prefix when writing batch visualisations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
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
    if not args.batch_output_dir and not args.output:
        parser.error("Please provide --output or --batch-output-dir.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the recommendation tree visualiser.

    Depending on the provided arguments the command renders either recommendation trees
    from CSV exports or viewer sessions sourced from cleaned datasets. Errors are
    communicated via :class:`SystemExit` to match typical CLI semantics.

    :param argv: Optional list of CLI arguments mirroring :data:`sys.argv[1:]`.
    :type argv: Sequence[str] | None
    :raises SystemExit: When the command arguments are inconsistent or data is invalid.
    """
    # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    args = parse_args(argv)
    highlight_path: List[str] = []
    if args.highlight:
        highlight_path = [token.strip() for token in args.highlight.split(",") if token.strip()]

    dataset = _load_cleaned_dataset(args.cleaned_data) if args.cleaned_data else None

    if args.batch_output_dir:
        if args.tree:
            raise SystemExit("--batch-output-dir currently supports --cleaned-data only.")
        if not dataset:
            raise SystemExit("--batch-output-dir requires --cleaned-data.")
        if args.issue:
            raise SystemExit(
                "--batch-output-dir is incompatible with --issue. Use --batch-issues instead."
            )
        try:
            issue_targets = parse_issue_counts(args.batch_issues)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not issue_targets:
            raise SystemExit("No issue counts provided for batch rendering.")
        output_dir = args.batch_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_format = args.format or "svg"

        emitted = 0
        for issue, count in issue_targets.items():
            if count <= 0:
                continue
            rows = _collect_rows(dataset, split=args.split, issue=issue)
            sessions = _group_rows_by_session(rows)
            if len(sessions) < count:
                raise SystemExit(
                    f"Requested {count} session(s) for issue '{issue}', "
                    f"but only {len(sessions)} found."
                )
            for idx, session_id in enumerate(sorted(sessions)[:count], start=1):
                session_rows = sessions[session_id]
                if args.max_steps and args.max_steps > 0:
                    session_rows = session_rows[: args.max_steps]
                graph = build_session_graph(
                    session_rows,
                    label_template=args.label_template,
                    wrap_width=args.wrap_width,
                    rankdir=args.rankdir,
                    engine=args.engine,
                    highlight_path=highlight_path,
                )
                filename = f"{args.batch_prefix}_{issue}_{idx}.{output_format}"
                output_path = output_dir / filename
                render_graph(graph, output_path, output_format=output_format)
                print(f"Wrote {output_path}", file=sys.stderr)
                emitted += 1
        if emitted == 0:
            raise SystemExit(
                "No sessions rendered. Check --batch-issues counts and dataset filters."
            )
        return

    if args.cleaned_data:
        assert dataset is not None  # for type checkers
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
        child_prefixes = tuple(
            prefix.strip()
            for prefix in args.child_prefixes.split(",")
            if prefix.strip()
        )
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
    output_format = args.format or (args.output.suffix.lstrip(".") if args.output else "")
    render_graph(graph, args.output, output_format=output_format)


if __name__ == "__main__":
    main()
