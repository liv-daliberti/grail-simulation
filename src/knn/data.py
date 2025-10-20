"""Dataset loading and issue filtering utilities for the KNN baseline."""

from __future__ import annotations

import os
from typing import Any, Dict, List

try:  # pragma: no cover - optional dependency
    from datasets import DatasetDict, load_dataset, load_from_disk  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    DatasetDict = None  # type: ignore
    load_dataset = load_from_disk = None  # type: ignore

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

DEFAULT_DATASET_SOURCE = "data/cleaned_grail"
TRAIN_SPLIT = "train"
EVAL_SPLIT = "validation"
PROMPT_COLUMN = "state_text"
SOLUTION_COLUMN = "gold_id"
PROMPT_MAX_HISTORY = int(
    os.environ.get("KNN_PROMPT_MAX_HISTORY", os.environ.get("GRAIL_MAX_HISTORY", "12"))
)


def load_dataset_source(source: str, cache_dir: str) -> DatasetDict:
    """Load a cleaned dataset from disk or from the Hugging Face Hub."""

    if load_dataset is None or load_from_disk is None:  # pragma: no cover - optional dependency
        raise ImportError("datasets must be installed to load the GRAIL dataset")
    if os.path.isdir(source):
        return load_from_disk(source)
    dataset = load_dataset(source, cache_dir=cache_dir)
    if isinstance(dataset, DatasetDict):
        return dataset
    raise ValueError(f"Dataset {source!r} did not return splits in a DatasetDict")


def issues_in_dataset(ds: DatasetDict) -> List[str]:
    """Return the list of issue labels present in the dataset."""

    train_split = ds.get(TRAIN_SPLIT) or next(iter(ds.values()))
    if "issue" not in train_split.column_names:
        return ["all"]
    issues = sorted({str(x).strip() for x in train_split["issue"] if str(x).strip()})
    return issues or ["all"]


def filter_dataset_for_issue(ds: DatasetDict, issue: str) -> DatasetDict:
    """Return a dataset filtered down to a specific issue label."""

    if issue == "all" or "issue" not in ds[TRAIN_SPLIT].column_names:
        return ds

    def _match_issue(row: Dict[str, Any]) -> bool:
        """Return ``True`` when the row's issue matches the requested label.

        :param row: Dataset example containing an ``issue`` column.
        :returns: Whether the row belongs to the requested issue slice.
        """
        value = row.get("issue")
        return str(value).strip() == issue

    filtered: Dict[str, Any] = {}
    for split_name, split_ds in ds.items():
        if "issue" not in split_ds.column_names:
            filtered[split_name] = split_ds
        else:
            filtered[split_name] = split_ds.filter(_match_issue)
    return DatasetDict(filtered)


__all__ = [
    "DEFAULT_DATASET_SOURCE",
    "EVAL_SPLIT",
    "PROMPT_COLUMN",
    "PROMPT_MAX_HISTORY",
    "SOLUTION_COLUMN",
    "TRAIN_SPLIT",
    "filter_dataset_for_issue",
    "issues_in_dataset",
    "load_dataset_source",
]
