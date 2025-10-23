"""Shared utilities reused across multiple baselines."""

from __future__ import annotations

from .cli_args import add_comma_separated_argument, add_sentence_transformer_normalise_flags
from .eval_utils import safe_div
from .logging_utils import ensure_directory, get_logger
from .opinion import opinion_example_kwargs
from .pipeline_utils import OpinionStudySelection, merge_ordered
from .text import canon_text, canon_video_id, resolve_paths_from_env, split_env_list
from .title_index import TitleResolver
from .vectorizers import create_tfidf_vectorizer

__all__ = [
    "OpinionStudySelection",
    "TitleResolver",
    "add_comma_separated_argument",
    "add_sentence_transformer_normalise_flags",
    "canon_text",
    "canon_video_id",
    "create_tfidf_vectorizer",
    "ensure_directory",
    "get_logger",
    "merge_ordered",
    "opinion_example_kwargs",
    "resolve_paths_from_env",
    "safe_div",
    "split_env_list",
]
