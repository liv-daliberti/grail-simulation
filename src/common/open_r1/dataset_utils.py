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

"""Dataset helpers shared by the Open-R1 evaluation scripts."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from datasets import DatasetDict
except ImportError:  # pragma: no cover - optional dependency
    DatasetDict = Any  # type: ignore


GoldLookup = Callable[[Dict[str, Any], Optional[str]], str]
SlateLoader = Callable[[Dict[str, Any]], Sequence[Dict[str, Any]]]
IndexResolver = Callable[[str, Sequence[Dict[str, Any]]], int]


def slate_has_gold(  # pylint: disable=too-many-arguments
    example: Dict[str, Any],
    solution_key: Optional[str],
    *,
    load_slate_items: SlateLoader,
    lookup_gold_id: GoldLookup,
    resolve_gold_index: IndexResolver,
    minimum_index: int = 1,
) -> bool:
    """Return ``True`` when ``example`` contains a slate and a matching gold answer.

    :param example: Dataset row containing slate information.
    :param solution_key: Optional column name holding the gold identifier.
    :param load_slate_items: Callable extracting slate items from the row.
    :param lookup_gold_id: Callable resolving the gold identifier for the row.
    :param resolve_gold_index: Callable mapping gold identifier to an index.
    :param minimum_index: Minimum index value considered valid.
    :returns: ``True`` when the slate exists and the gold identifier is on it.
    """

    items = load_slate_items(example)
    if not items:
        return False
    gold_id = lookup_gold_id(example, solution_key)
    if not gold_id:
        return False
    return resolve_gold_index(gold_id, items) >= minimum_index


def make_slate_validator(
    *,
    load_slate_items: SlateLoader,
    lookup_gold_id: GoldLookup,
    resolve_gold_index: IndexResolver,
    minimum_index: int = 1,
) -> Callable[[Dict[str, Any]], bool]:
    """Return a partial of :func:`slate_has_gold` configured for dataset filtering.

    :param load_slate_items: Callable extracting slate items from a row.
    :param lookup_gold_id: Callable producing the gold identifier for a row.
    :param resolve_gold_index: Callable converting identifiers to slate indices.
    :param minimum_index: Minimum index value considered valid.
    :returns: Validator predicate suitable for :meth:`Dataset.filter`.
    """

    return partial(
        slate_has_gold,
        load_slate_items=load_slate_items,
        lookup_gold_id=lookup_gold_id,
        resolve_gold_index=resolve_gold_index,
        minimum_index=minimum_index,
    )


def drop_marked_rows(dataset: DatasetDict, train_split: str) -> None:
    """Remove rows flagged with ``__drop__`` from every split in-place.

    :param dataset: Hugging Face dataset dict to prune.
    :param train_split: Name of the training split containing the ``__drop__`` column.
    :returns: ``None``. Mutates the dataset to exclude flagged rows.
    """

    if "__drop__" not in dataset[train_split].column_names:
        return
    for split in list(dataset.keys()):
        mask = [not flag for flag in dataset[split]["__drop__"]]
        keep_indices = [idx for idx, keep in enumerate(mask) if keep]
        dataset[split] = dataset[split].select(keep_indices)


__all__ = ["drop_marked_rows", "slate_has_gold", "make_slate_validator"]
