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

"""Factory helpers for creating shared TF-IDF vectorisers."""

from __future__ import annotations

from typing import Optional

try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore


def create_tfidf_vectorizer(*, max_features: Optional[int] = None) -> TfidfVectorizer:
    """
    Return a TF-IDF vectoriser configured with shared defaults.

    :param max_features: Optional cap on the vocabulary size used for training.
    :returns: Instance of :class:`sklearn.feature_extraction.text.TfidfVectorizer`.
    :raises ImportError: If scikit-learn is not installed.
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
