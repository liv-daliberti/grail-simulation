"""Dataset loading and issue filtering utilities for the KNN baseline."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Sequence

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
    """
    Load a cleaned dataset from disk or from the Hugging Face Hub.

    :param source: Local directory path or Hugging Face dataset identifier.
    :type source: str
    :param cache_dir: Cache directory passed to :func:`datasets.load_dataset`.
    :type cache_dir: str
    :returns: Dataset dictionary containing train and validation splits.
    :rtype: datasets.DatasetDict
    :raises ImportError: If :mod:`datasets` is unavailable.
    :raises ValueError: When ``source`` does not yield a :class:`datasets.DatasetDict`.
    """
    if load_dataset is None or load_from_disk is None:  # pragma: no cover - optional dependency
        raise ImportError("datasets must be installed to load the GRAIL dataset")
    if os.path.isdir(source):
        return load_from_disk(source)
    dataset = load_dataset(source, cache_dir=cache_dir)
    if isinstance(dataset, DatasetDict):
        return dataset
    raise ValueError(f"Dataset {source!r} did not return splits in a DatasetDict")

def issues_in_dataset(dataset: DatasetDict) -> List[str]:
    """
    Return the list of issue labels present in the dataset.

    :param dataset: Dataset dictionary containing at least one split.
    :type dataset: datasets.DatasetDict
    :returns: Sorted list of issue identifiers or ``['all']`` when absent.
    :rtype: List[str]
    """
    train_split = dataset.get(TRAIN_SPLIT) or next(iter(dataset.values()))
    if "issue" not in train_split.column_names:
        return ["all"]
    issues = sorted({str(x).strip() for x in train_split["issue"] if str(x).strip()})
    return issues or ["all"]

def filter_dataset_for_issue(dataset: DatasetDict, issue: str) -> DatasetDict:
    """
    Filter a dataset down to a specific issue label.

    :param dataset: Dataset dictionary containing the ``issue`` column.
    :type dataset: datasets.DatasetDict
    :param issue: Issue label to retain (``all`` preserves every row).
    :type issue: str
    :returns: Dataset dictionary restricted to rows matching ``issue``.
    :rtype: datasets.DatasetDict
    """
    if issue == "all" or "issue" not in dataset[TRAIN_SPLIT].column_names:
        return dataset

    def _match_issue(row: Dict[str, Any]) -> bool:
        """
        Determine whether the row's issue matches the requested label.

        :param row: Dataset example containing an ``issue`` column.
        :type row: Dict[str, Any]
        :returns: ``True`` when the row belongs to the requested issue slice.
        :rtype: bool
        """
        value = row.get("issue")
        return str(value).strip() == issue

    filtered: Dict[str, Any] = {}
    for split_name, split_ds in dataset.items():
        if "issue" not in split_ds.column_names:
            filtered[split_name] = split_ds
        else:
            filtered[split_name] = split_ds.filter(_match_issue)
    return DatasetDict(filtered)

def _normalise_study_tokens(studies: Iterable[str]) -> set[str]:
    """
    Normalise raw participant-study tokens.

    :param studies: Iterable of study identifiers provided by the user.
    :type studies: Iterable[str]
    :returns: Lower-cased, de-duplicated set excluding ``all`` markers.
    :rtype: set[str]
    """
    normalised: set[str] = set()
    for value in studies:
        token = str(value).strip().lower()
        if not token or token == "all":
            continue
        normalised.add(token)
    return normalised

def filter_split_for_participant_studies(split_ds, studies: Sequence[str]):
    """
    Filter a dataset split to the requested ``participant_study`` tokens.

    :param split_ds: Dataset split supporting :meth:`filter`.
    :type split_ds: datasets.Dataset
    :param studies: Study identifiers to retain.
    :type studies: Sequence[str]
    :returns: Filtered dataset split covering only the requested studies.
    :rtype: datasets.Dataset
    """
    normalized = _normalise_study_tokens(studies)
    if not normalized:
        return split_ds
    if "participant_study" not in split_ds.column_names:
        return split_ds

    def _match_study(row: Dict[str, Any]) -> bool:
        """
        Determine whether the row's ``participant_study`` matches the filter.

        :param row: Dataset example retrieved from ``split_ds``.
        :type row: Dict[str, Any]
        :returns: ``True`` when the row should be retained.
        :rtype: bool
        """
        value = row.get("participant_study")
        return str(value).strip().lower() in normalized

    return split_ds.filter(_match_study)

def filter_dataset_for_participant_studies(
    dataset: DatasetDict,
    studies: Iterable[str],
) -> DatasetDict:
    """
    Filter the dataset to rows whose ``participant_study`` matches ``studies``.

    :param dataset: Dataset dictionary containing the ``participant_study`` column.
    :type dataset: datasets.DatasetDict
    :param studies: Study identifiers to retain.
    :type studies: Iterable[str]
    :returns: Dataset dictionary filtered to the requested studies.
    :rtype: datasets.DatasetDict
    """
    normalized = _normalise_study_tokens(studies)
    if not normalized:
        return dataset

    filtered: Dict[str, Any] = {}
    for split_name, split_ds in dataset.items():
        if "participant_study" not in split_ds.column_names:
            filtered[split_name] = split_ds
        else:
            filtered[split_name] = split_ds.filter(
                lambda row, norm=normalized: str(row.get("participant_study", "")).strip().lower()
                in norm
            )
    return DatasetDict(filtered)

__all__ = [
    "DEFAULT_DATASET_SOURCE",
    "EVAL_SPLIT",
    "PROMPT_COLUMN",
    "PROMPT_MAX_HISTORY",
    "SOLUTION_COLUMN",
    "TRAIN_SPLIT",
    "filter_dataset_for_issue",
    "filter_dataset_for_participant_studies",
    "filter_split_for_participant_studies",
    "issues_in_dataset",
    "load_dataset_source",
]
