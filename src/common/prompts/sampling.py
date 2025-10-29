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

"""Shared indexing helpers for prompt preparation utilities."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, TypeVar

from numpy.random import default_rng

CollectedT = TypeVar("CollectedT")


def select_train_indices(
    dataset: Sequence[object],
    *,
    max_train: int | None,
    seed: int,
) -> list[int]:
    """
    Return deterministic training indices with optional subsampling.

    :param dataset: Training split providing random access to examples.
    :type dataset: Sequence[object]
    :param max_train: Optional cap on the number of examples to retain.
    :type max_train: int | None
    :param seed: Random seed used when subsampling.
    :type seed: int
    :returns: Ordered list of example indices to process.
    :rtype: list[int]
    :raises RuntimeError: If the training split is empty.
    """

    n_rows = len(dataset)
    if n_rows == 0:
        raise RuntimeError("Train split is empty.")

    if max_train and max_train > 0:
        rng = default_rng(seed)
        take = min(max_train, n_rows)
        return rng.permutation(n_rows)[:take].tolist()
    return list(range(n_rows))


def collect_selected_examples(
    dataset: Sequence[object],
    *,
    max_train: int | None,
    seed: int,
    collect: Callable[[int, object], CollectedT | None],
) -> Tuple[List[int], List[CollectedT]]:
    """
    Collect values produced by ``collect`` for selected training indices.

    The helper applies ``select_train_indices`` and forwards each example to
    ``collect``. ``None`` return values are skipped, allowing callers to filter
    unsuitable rows while still tracking how many indices were inspected.

    :param dataset: Training split providing random access to examples.
    :param max_train: Optional cap on the number of examples inspected.
    :param seed: Random seed controlling the subsampling order.
    :param collect: Callback receiving ``(index, example)`` and returning a value or ``None``.
    :returns: Tuple containing the processed indices and the collected values.
    """

    indices = select_train_indices(dataset, max_train=max_train, seed=seed)
    collected: List[CollectedT] = []
    for index in indices:
        example = dataset[int(index)]
        accepted = collect(int(index), example)
        if accepted is not None:
            collected.append(accepted)
    return indices, collected


__all__ = ["collect_selected_examples", "select_train_indices"]
