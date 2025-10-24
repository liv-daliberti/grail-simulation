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

from typing import Optional

try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore


def create_tfidf_vectorizer(*, max_features: Optional[int] = None) -> TfidfVectorizer:
    """

    Return a TF-IDF vectoriser with the shared configuration.



    :param max_features: Value provided for ``max_features``.

    :type max_features: Optional[int]

    :returns: Result produced by ``create_tfidf_vectorizer``.

    :rtype: TfidfVectorizer

    """


    if TfidfVectorizer is None:  # pragma: no cover - optional dependency
        raise ImportError("Install scikit-learn to use TF-IDF vectorisation.")
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None,
        token_pattern=r"(?u)\b[\w\-]{2,}\b",
        max_features=max_features,
    )


__all__ = ["create_tfidf_vectorizer", "TfidfVectorizer"]
