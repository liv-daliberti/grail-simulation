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

"""Input/output helpers for recommendation tree visualisations."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

try:  # pragma: no cover - optional dependency
    from datasets import load_dataset, load_from_disk  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore
    load_from_disk = None  # type: ignore

from .models import TreeData, TreeEdge, _natural_sort_key


def parse_issue_counts(spec: str) -> Dict[str, int]:
    """Parse comma-separated ``issue=count`` specifications into a mapping.

    :param spec: Raw specification string (e.g. ``"issue_a=2,issue_b=1"``).
    :returns: Mapping of issue names to target counts.
    :raises ValueError: If any chunk is missing an issue, count, or is invalid.
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

    :param csv_path: Path to the CSV file describing the tree.
    :param id_column: Column holding the node identifier.
    :param child_prefixes: Column prefixes pointing to recommendation children.
    :returns: Parsed :class:`TreeData` ready for rendering.
    """

    csv_frame = pd.read_csv(csv_path)
    resolved_id = _resolve_id_column(csv_frame, id_column)
    children_cols = _identify_children_columns(csv_frame, child_prefixes)
    nodes, edges, parent_order, seen_children = _build_tree_components(
        csv_frame,
        resolved_id,
        children_cols,
    )
    root = _select_root(parent_order, seen_children)
    return TreeData(root=root, nodes=nodes, edges=edges)


def _resolve_id_column(frame: pd.DataFrame, preferred: str) -> str:
    """Return the identifier column to use when parsing tree CSVs.

    :param frame: DataFrame loaded from the tree CSV.
    :param preferred: Desired identifier column name.
    :returns: Column name present in the frame that will be used as the id.
    """

    if preferred in frame.columns:
        return preferred
    lowered = preferred.lower()
    for column in frame.columns:
        if column.lower() == lowered:
            return column
    return frame.columns[0]


def _identify_children_columns(
    frame: pd.DataFrame,
    child_prefixes: Sequence[str],
) -> List[str]:
    """Detect recommendation child columns using prefix heuristics.

    :param frame: DataFrame loaded from the tree CSV.
    :param child_prefixes: Column prefixes pointing to child recommendations.
    :returns: Sorted list of child column names.
    :raises ValueError: If no matching columns are found.
    """

    prefixes = [prefix.lower() for prefix in child_prefixes]
    columns = [
        column
        for column in frame.columns
        if any(column.lower().startswith(prefix) for prefix in prefixes)
    ]
    if not columns:
        raise ValueError(
            "Could not identify recommendation columns. "
            "Use --child-prefixes to point to the recommendation columns."
        )
    columns.sort(key=_natural_sort_key)
    return columns


def _build_tree_components(
    frame: pd.DataFrame,
    id_column: str,
    children_cols: Sequence[str],
) -> Tuple[Dict[str, Mapping[str, object]], List[TreeEdge], List[str], set[str]]:
    """Populate the node and edge collections for the tree.

    :param frame: DataFrame loaded from the tree CSV.
    :param id_column: Identifier column to reference parent nodes.
    :param children_cols: Ordered list of child columns discovered earlier.
    :returns: Tuple containing node mapping, edges, parent order, and seen children.
    """

    nodes: Dict[str, Mapping[str, object]] = {}
    edges: List[TreeEdge] = []
    seen_children: set[str] = set()
    parent_order: List[str] = []
    for _, row in frame.iterrows():
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
            nodes.setdefault(child, {})
    return nodes, edges, parent_order, seen_children


def _select_root(parent_order: Sequence[str], seen_children: Sequence[str]) -> str:
    """Infer the root node for the tree by excluding seen children.

    :param parent_order: Node identifiers in the order they appear in the CSV.
    :param seen_children: Identifiers that appeared as child nodes.
    :returns: Identifier for the inferred root node.
    """

    for node in parent_order:
        if node not in seen_children:
            return node
    return parent_order[0] if parent_order else ""


def load_metadata(
    metadata_path: Optional[Path],
    *,
    id_column: str = "originId",
) -> Dict[str, Mapping[str, object]]:
    """Load per-node metadata to enrich node labels.

    :param metadata_path: Optional path to JSON/CSV metadata.
    :param id_column: Identifier column used to index rows.
    :returns: Mapping of identifier to metadata rows.
    :raises ValueError: If the identifier column is missing in tabular inputs.
    """

    if metadata_path is None:
        return {}
    suffix = metadata_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        records = _read_json_metadata(metadata_path)
        return _records_to_lookup(records, id_column)
    return _read_tabular_metadata(metadata_path, id_column)


def _read_json_metadata(path: Path) -> List[Mapping[str, object]]:
    """Read metadata records from JSON or JSONL sources.

    :param path: Path to the metadata file.
    :returns: List of mapping records extracted from the file.
    """

    if path.suffix.lower() == ".jsonl":
        return _read_jsonl_metadata(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [obj for obj in data if isinstance(obj, Mapping)]
    if isinstance(data, Mapping):
        return [value for value in data.values() if isinstance(value, Mapping)]
    return []


def _read_jsonl_metadata(path: Path) -> List[Mapping[str, object]]:
    """Parse JSON Lines metadata into a list of mapping records.

    :param path: Path to the JSONL metadata file.
    :returns: List of mapping records parsed from the file.
    """

    records: List[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, Mapping):
                records.append(obj)
    return records


def _records_to_lookup(
    records: Sequence[Mapping[str, object]],
    id_column: str,
) -> Dict[str, Mapping[str, object]]:
    """Convert metadata records into a lookup keyed by identifier.

    :param records: Iterable of metadata mapping records.
    :param id_column: Identifier key to use for the lookup.
    :returns: Mapping from identifier to metadata record.
    """

    lookup: Dict[str, Mapping[str, object]] = {}
    for row in records:
        if id_column in row:
            lookup[str(row[id_column])] = row
    return lookup


def _read_tabular_metadata(path: Path, id_column: str) -> Dict[str, Mapping[str, object]]:
    """Load metadata from CSV/TSV exports.

    :param path: Path to the CSV/TSV metadata file.
    :param id_column: Identifier column expected in the file.
    :returns: Mapping from identifier to metadata row.
    :raises ValueError: If the identifier column is absent.
    """

    metadata_frame = pd.read_csv(path)
    if id_column not in metadata_frame.columns:
        raise ValueError(
            f"Metadata file {path} is missing the identifier column '{id_column}'."
        )
    result: Dict[str, Mapping[str, object]] = {}
    for _, row in metadata_frame.iterrows():
        result[str(row[id_column])] = row.to_dict()
    return result


def _extract_sequences_from_object(obj: object) -> List[List[str]]:
    """Extract sequences of identifiers from a heterogeneous object.

    :param obj: Parsed JSON entry or primitive containing trajectory data.
    :returns: List of sequences extracted from the object.
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

    :param path: Optional path to a trajectories file.
    :param delimiter: Delimiter used for plain-text trajectory files.
    :returns: List of normalised identifier sequences.
    """

    if path is None:
        return []
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        sequences = _load_json_trajectories(path)
    else:
        sequences = _load_text_trajectories(path, suffix, delimiter)
    return _normalise_trajectories(sequences)


def _load_json_trajectories(path: Path) -> List[List[str]]:
    """Load trajectory sequences from JSON or JSONL files.

    :param path: Path to the JSON/JSONL trajectories file.
    :returns: Raw sequences extracted from the file.
    """

    suffix = path.suffix.lower()
    sequences: List[List[str]] = []
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
    return sequences


def _load_text_trajectories(path: Path, suffix: str, delimiter: str) -> List[List[str]]:
    """Load trajectory sequences from CSV, TSV, or plain-text files.

    :param path: Path to the text-based trajectories file.
    :param suffix: Lowercased file suffix used to determine parsing mode.
    :param delimiter: Delimiter when parsing generic text files.
    :returns: Raw sequences extracted from the file.
    """

    sequences: List[List[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        if suffix in {".csv", ".tsv", ".txt"}:
            dialect = csv.excel_tab if suffix == ".tsv" else csv.excel
            reader = csv.reader(handle, dialect=dialect)
            for row in reader:
                tokens = [token.strip() for token in row if token.strip()]
                if tokens:
                    sequences.append(tokens)
        else:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = [segment.strip() for segment in line.split(delimiter) if segment.strip()]
                if parts:
                    sequences.append(parts)
    return sequences


def _normalise_trajectories(sequences: Sequence[Sequence[object]]) -> List[List[str]]:
    """Normalise sequences by dropping blanks and consecutive duplicates.

    :param sequences: Raw sequences produced by loader helpers.
    :returns: Cleaned sequences ready for aggregation.
    """

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


def load_cleaned_dataset(path: Path):
    """Load a HuggingFace dataset exported by ``clean_data.py``.

    :param path: Path to the dataset directory or file.
    :returns: HuggingFace dataset object.
    :raises ImportError: If the ``datasets`` package is not available.
    :raises ValueError: If the file extension is unsupported.
    """

    if load_dataset is None or load_from_disk is None:
        raise ImportError(
            "Loading recommendation tree datasets requires the 'datasets' package. "
            "Install it with `pip install datasets`."
        )
    if path.is_dir():
        return load_from_disk(str(path))
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(path))
    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        return load_dataset("csv", data_files=str(path), delimiter=delimiter)
    raise ValueError(f"Unsupported cleaned dataset format: {path}")




def collect_rows(
    dataset,
    split: Optional[str] = None,
    issue: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Collect dataset rows, optionally filtering by split name and issue.

    :param dataset: HuggingFace dataset or mapping of splits to datasets.
    :param split: Optional split name to filter.
    :param issue: Optional issue identifier to filter.
    :returns: List of dataset rows matching the filters.
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




def group_rows_by_session(
    rows: Sequence[Mapping[str, object]]
) -> Dict[str, List[Mapping[str, object]]]:
    """Group dataset rows by session identifier and sort within each session.

    :param rows: Sequence of dataset rows.
    :returns: Mapping from session identifier to sorted row list.
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




def extract_session_rows(
    dataset,
    *,
    session_id: Optional[str],
    split: Optional[str],
    issue: Optional[str],
    max_steps: Optional[int],
) -> Tuple[str, List[Dict[str, object]]]:
    """Extract the rows that correspond to a target viewer session.

    :param dataset: HuggingFace dataset or mapping of splits to datasets.
    :param session_id: Optional specific session identifier to select.
    :param split: Optional dataset split name.
    :param issue: Optional issue filter.
    :param max_steps: Optional maximum number of steps to return.
    :returns: Tuple of resolved session identifier and the matching rows.
    :raises ValueError: If no rows or matching session can be found.
    """

    rows = collect_rows(dataset, split=split, issue=issue)
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


# Backwards compatibility for callers still using private names.
_load_cleaned_dataset = load_cleaned_dataset
_collect_rows = collect_rows
_group_rows_by_session = group_rows_by_session
_extract_session_rows = extract_session_rows


__all__ = [
    "collect_rows",
    "extract_session_rows",
    "_extract_sequences_from_object",
    "group_rows_by_session",
    "load_cleaned_dataset",
    "load_metadata",
    "load_trajectories",
    "load_tree_csv",
    "parse_issue_counts",
]
