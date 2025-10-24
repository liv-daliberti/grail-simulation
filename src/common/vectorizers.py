"""Factories for shared sklearn vectorisers."""

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
