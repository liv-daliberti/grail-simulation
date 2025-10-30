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

from .models import TreeData, TreeEdge, _natural_sort_key


def parse_issue_counts(spec: str) -> Dict[str, int]:
    """Parse comma-separated ``issue=count`` specifications into a mapping."""

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
    """Load a recommendation tree from a CSV export."""

    # pylint: disable=too-many-locals
    csv_frame = pd.read_csv(csv_path)
    normalized_cols = {col: col.lower() for col in csv_frame.columns}
    if id_column not in csv_frame.columns:
        lowered = id_column.lower()
        for original, lower in normalized_cols.items():
            if lower == lowered:
                id_column = original
                break
        else:
            id_column = csv_frame.columns[0]
    children_cols: List[str] = []
    for col in csv_frame.columns:
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
    for _, row in csv_frame.iterrows():
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
    """Load per-node metadata to enrich node labels."""

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
    metadata_frame = pd.read_csv(metadata_path)
    if id_column not in metadata_frame.columns:
        raise ValueError(
            f"Metadata file {metadata_path} is missing the identifier column '{id_column}'."
        )
    result: Dict[str, Mapping[str, object]] = {}
    for _, row in metadata_frame.iterrows():
        result[str(row[id_column])] = row.to_dict()
    return result


def _extract_sequences_from_object(obj: object) -> List[List[str]]:
    """Extract sequences of identifiers from a heterogeneous object."""

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
    """Load viewer trajectories from text, CSV, or JSON representations."""

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


def load_cleaned_dataset(path: Path):
    """Load a HuggingFace dataset exported by ``clean_data.py``."""

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




def collect_rows(
    dataset,
    split: Optional[str] = None,
    issue: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Collect dataset rows, optionally filtering by split name and issue."""

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
    """Group dataset rows by session identifier and sort within each session."""

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
    """Extract the rows that correspond to a target viewer session."""

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
