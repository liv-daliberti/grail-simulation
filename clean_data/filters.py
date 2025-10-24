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

"""Row-level filters and quick aggregate reporting utilities.

This module contains the lightweight predicates used to discard unusable
interaction rows before prompt construction as well as diagnostic helpers
that summarise the issue distribution across splits. These functions are
intended to be composed by :mod:`clean_data.clean_data` and higher-level
entry points rather than imported by downstream consumers directly. The
suite is provided under the repository's Apache 2.0 license; refer to the
LICENSE file for the obligations and permissions.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, Optional

try:
    from datasets import DatasetDict
except ImportError:  # pragma: no cover - optional dependency for linting
    DatasetDict = Any  # type: ignore

from clean_data.prompting import (
    get_gold_next_id,
    gold_index_from_items,
    load_slate_items,
)

log = logging.getLogger("clean_grail")


def filter_prompt_ready(
    dataset: DatasetDict,
    sol_key: Optional[str] = None,
    num_proc: Optional[int] = None,
) -> DatasetDict:
    """Drop rows that cannot produce a valid prompt/example.

    :param dataset: Input dataset keyed by split.
    :param sol_key: Alternate gold-id column used to validate the target choice.
    :param num_proc: Optional number of worker processes used by ``datasets.filter``.
    :returns: Dataset dictionary with non-compliant rows removed.
    """

    def _ok(example: dict) -> bool:
        """Return ``True`` when ``example`` has a valid slate and gold target.

        :param example: Row under evaluation from the dataset.
        :returns: ``True`` when the slate items and gold choice are present and valid.
        """

        items = load_slate_items(example)
        if not items:
            return False
        gold = get_gold_next_id(example, sol_key)
        if not gold:
            return False
        return gold_index_from_items(gold, items) >= 1

    if num_proc is not None and num_proc < 1:
        raise ValueError("num_proc must be >= 1 when provided.")

    filtered = DatasetDict()
    for split_name, split_ds in dataset.items():
        filter_kwargs: Dict[str, int] = {}
        if num_proc is not None:
            filter_kwargs["num_proc"] = num_proc
        try:
            filtered_split = split_ds.filter(_ok, **filter_kwargs)
        except TypeError:
            filtered_split = split_ds.filter(_ok)
        filtered[split_name] = filtered_split
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
