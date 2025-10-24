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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

from __future__ import annotations

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
    """Return ``True`` when ``example`` contains a slate and a matching gold answer."""

    items = load_slate_items(example)
    if not items:
        return False
    gold_id = lookup_gold_id(example, solution_key)
    if not gold_id:
        return False
    return resolve_gold_index(gold_id, items) >= minimum_index


def drop_marked_rows(dataset: DatasetDict, train_split: str) -> None:
    """Remove rows flagged with ``__drop__`` from every split in-place."""

    if "__drop__" not in dataset[train_split].column_names:
        return
    for split in list(dataset.keys()):
        mask = [not flag for flag in dataset[split]["__drop__"]]
        keep_indices = [idx for idx, keep in enumerate(mask) if keep]
        dataset[split] = dataset[split].select(keep_indices)


__all__ = ["drop_marked_rows", "slate_has_gold"]
