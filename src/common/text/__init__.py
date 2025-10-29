"""Aggregated exports for shared text utilities."""

from .title_index import TitleResolver
from .utils import (
    CANON_RE,
    YTID_RE,
    canon_text,
    canon_video_id,
    resolve_paths_from_env,
    split_env_list,
)
from .vectorizers import create_tfidf_vectorizer

__all__ = [
    "CANON_RE",
    "YTID_RE",
    "TitleResolver",
    "canon_text",
    "canon_video_id",
    "create_tfidf_vectorizer",
    "resolve_paths_from_env",
    "split_env_list",
]
