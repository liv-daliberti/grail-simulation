"""Row-level filters and quick aggregate reporting utilities.

This module contains the lightweight predicates used to discard unusable
interaction rows before prompt construction as well as diagnostic helpers
that summarise the issue distribution across splits.  These functions are
intended to be composed by :mod:`clean_data.clean_data` and higher-level
entry points rather than imported by downstream consumers directly.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, Optional

from datasets import DatasetDict

from clean_data.prompting import (
    get_gold_next_id,
    gold_index_from_items,
    load_slate_items,
)

log = logging.getLogger("clean_grail")


def filter_prompt_ready(dataset: DatasetDict, sol_key: Optional[str] = None) -> DatasetDict:
    """Drop rows that cannot produce a valid prompt/example.

    :param dataset: Input dataset keyed by split.
    :param sol_key: Alternate gold-id column used to validate the target choice.
    :returns: Dataset dictionary with non-compliant rows removed.
    """

    def _ok(example: dict) -> bool:
        items = load_slate_items(example)
        if not items:
            return False
        gold = get_gold_next_id(example, sol_key)
        if not gold:
            return False
        return gold_index_from_items(gold, items) >= 1

    filtered = DatasetDict()
    for split_name, split_ds in dataset.items():
        filtered[split_name] = split_ds.filter(_ok)
    log.info("Counts after prompt filter: %s", {k: len(v) for k, v in filtered.items()})
    return filtered


def compute_issue_counts(dataset: DatasetDict) -> Dict[str, Dict[str, int]]:
    """Summarize issue/category distributions for logging.

    :param dataset: Dataset containing an ``issue`` column for each split.
    :returns: Nested mapping of split name -> issue label -> count.
    """

    counts: Dict[str, Dict[str, int]] = {}
    for split_name, split_ds in dataset.items():
        if "issue" not in split_ds.column_names:
            continue
        issue_column = split_ds["issue"]  # type: ignore[index]
        values = (str(value or "").strip() for value in issue_column)
        counter = Counter(values)
        counts[split_name] = {key or "(missing)": count for key, count in counter.items()}
    return counts


__all__ = ["filter_prompt_ready", "compute_issue_counts"]
