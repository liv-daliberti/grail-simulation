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

import os
from typing import Callable, Optional, Sequence, Tuple, TypeVar


_Dataset = TypeVar("_Dataset")


def safe_div(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    """



        Return the division result guarding against a zero denominator.



        Parameters

        ----------

        numerator:

            Value forming the numerator.

        denominator:

            Value forming the denominator.

        default:

            Fallback value returned when ``denominator`` is zero.



    :param numerator: Value provided for ``numerator``.

    :type numerator: float

    :param denominator: Value provided for ``denominator``.

    :type denominator: float

    :param default: Value provided for ``default``.

    :type default: float

    :returns: Result produced by ``safe_div``.

    :rtype: float

    """


    if not denominator:
        return default
    return numerator / denominator


def ensure_hf_cache(cache_dir: str) -> None:
    """

    Ensure Hugging Face cache directories default to ``cache_dir``.



    :param cache_dir: Value provided for ``cache_dir``.

    :type cache_dir: str

    :returns: ``None``.

    :rtype: None

    """


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
    """



        Configure the HF cache, load the dataset, and list available issues.



        Parameters

        ----------

        dataset:

            Dataset identifier supplied via CLI (``None`` uses ``default_source``).

        default_source:

            Default dataset identifier when ``dataset`` is not provided.

        cache_dir:

            Directory used for Hugging Face caching.

        loader:

            Callable that loads the dataset for ``dataset_source``.

        issue_lookup:

            Callable returning the available issue labels for ``loader``'s output.



        Returns

        -------

        tuple[str, Any, Sequence[str]]

            The dataset source string, loaded dataset object, and issue labels.



    :param dataset: Value provided for ``dataset``.

    :type dataset: Optional[str]

    :param default_source: Value provided for ``default_source``.

    :type default_source: str

    :param cache_dir: Value provided for ``cache_dir``.

    :type cache_dir: str

    :param loader: Value provided for ``loader``.

    :type loader: Callable[[str, str], _Dataset]

    :param issue_lookup: Value provided for ``issue_lookup``.

    :type issue_lookup: Callable[[_Dataset], Sequence[str]]

    :returns: Result produced by ``prepare_dataset``.

    :rtype: Tuple[str, _Dataset, Sequence[str]]

    """


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
