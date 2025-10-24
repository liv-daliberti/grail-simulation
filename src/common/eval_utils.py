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

"""Utility helpers shared by multiple evaluation pipelines."""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence, Tuple, TypeVar


_Dataset = TypeVar("_Dataset")


def safe_div(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    """Return the division result while guarding against a zero denominator.

    :param numerator: Value forming the numerator.
    :type numerator: float
    :param denominator: Value forming the denominator.
    :type denominator: float
    :param default: Fallback returned when ``denominator`` is zero.
    :type default: float
    :returns: Division result or ``default`` when the denominator is zero.
    :rtype: float
    """

    if not denominator:
        return default
    return numerator / denominator


def ensure_hf_cache(cache_dir: str) -> None:
    """Default HF cache environment variables to ``cache_dir`` if unset."""
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    os.environ.setdefault("HF_HOME", cache_dir)


def prepare_dataset(
    *,
    dataset: Optional[str],
    default_source: str,
    cache_dir: str,
    loader: Callable[[str, str], _Dataset],
    issue_lookup: Callable[[_Dataset], Sequence[str]],
) -> Tuple[str, _Dataset, Sequence[str]]:
    """Resolve the dataset source, load it, and enumerate available issues."""
    ensure_hf_cache(cache_dir)
    dataset_source = dataset or default_source
    base_ds = loader(dataset_source, cache_dir)
    available_issues = issue_lookup(base_ds)
    return dataset_source, base_ds, available_issues


def compose_issue_slug(issue: str, study_tokens: Sequence[str]) -> str:
    """



        Return a filesystem-safe slug combining ``issue`` and ``study_tokens``.



        Tokens matching ``all`` (case-insensitive) are ignored to avoid noise.



    :param issue: Value provided for ``issue``.

    :type issue: str

    :param study_tokens: Value provided for ``study_tokens``.

    :type study_tokens: Sequence[str]

    :returns: Result produced by ``compose_issue_slug``.

    :rtype: str

    """


    base_slug = issue.replace(" ", "_") if issue and issue.strip() else "all"
    suffix_parts: list[str] = []
    seen_suffix: set[str] = set()
    for token in study_tokens:
        slug = token.replace(" ", "_")
        if slug and slug.lower() != "all" and slug not in seen_suffix:
            suffix_parts.append(slug)
            seen_suffix.add(slug)
    if suffix_parts:
        return f"{base_slug}_{'_'.join(suffix_parts)}"
    return base_slug


__all__ = ["compose_issue_slug", "ensure_hf_cache", "prepare_dataset", "safe_div"]
