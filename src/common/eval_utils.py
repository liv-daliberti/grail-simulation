"""Shared evaluation utilities reused across baseline implementations."""

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
    """

    if not denominator:
        return default
    return numerator / denominator


def ensure_hf_cache(cache_dir: str) -> None:
    """Ensure Hugging Face cache directories default to ``cache_dir``."""

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
