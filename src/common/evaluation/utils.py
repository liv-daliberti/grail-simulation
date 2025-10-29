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

from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np


_Dataset = TypeVar("_Dataset")


@dataclass(frozen=True)
class BootstrapSummaryConfig:
    """Metadata describing the bootstrap resampling configuration."""

    n_groups: int
    n_rows: int
    n_bootstrap: int
    seed: int


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
    """
    Default HF cache environment variables to ``cache_dir`` if unset.

    :param cache_dir: Directory used when populating ``HF_DATASETS_CACHE`` and ``HF_HOME``.
    :returns: ``None``.
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
    Resolve the dataset source, load it, and enumerate available issues.

    :param dataset: Explicit dataset identifier supplied by the caller.
    :param default_source: Default dataset identifier used when ``dataset`` is ``None``.
    :param cache_dir: Hugging Face cache directory forwarded to the loader.
    :param loader: Callable that loads the dataset given the source and cache path.
    :param issue_lookup: Callable returning the available issue labels for the dataset.
    :returns: Tuple of dataset source, loaded dataset object, and ordered issue labels.
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

    :param issue: Issue label used as the slug prefix.
    :param study_tokens: Sequence of participant study identifiers.
    :returns: Normalised slug suitable for filenames.
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


def group_key_for_example(example: Mapping[str, Any], fallback_index: int) -> str:
    """
    Return a stable grouping key used for bootstrap resampling.

    :param example: Dataset row containing participant/session hints.
    :param fallback_index: Fallback index used when identifiers are absent.
    :returns: Group key derived from ``urlid``, ``participant_id``, ``session_id``, or row index.
    """

    urlid = str(example.get("urlid") or "").strip()
    if urlid and urlid.lower() != "nan":
        return f"urlid::{urlid}"
    participant = str(example.get("participant_id") or "").strip()
    if participant and participant.lower() != "nan":
        return f"participant::{participant}"
    session = str(example.get("session_id") or "").strip()
    if session and session.lower() != "nan":
        return f"session::{session}"
    return f"row::{fallback_index}"


def participant_bootstrap_summary(
    *,
    model_samples: Sequence[float],
    baseline_samples: Sequence[float] | None,
    summary_config: BootstrapSummaryConfig,
) -> Dict[str, Any]:
    """Summarise participant bootstrap samples using the shared helper."""
    return summarise_bootstrap_samples(
        model_samples=model_samples,
        baseline_samples=baseline_samples,
        method="participant_bootstrap",
        config=summary_config,
    )


def build_participant_bootstrap_summary(
    *,
    model_samples: Sequence[float],
    baseline_samples: Sequence[float] | None,
    summary_config: BootstrapSummaryConfig,
) -> Dict[str, Any]:
    """
    Convenience wrapper that forwards pre-built ``BootstrapSummaryConfig`` objects.

    :param model_samples: Bootstrap samples produced for the model accuracy.
    :param baseline_samples: Optional baseline accuracy samples.
    :param summary_config: Resampling metadata describing the bootstrap parameters.
    :returns: Summary dictionary created by :func:`participant_bootstrap_summary`.
    """

    return participant_bootstrap_summary(
        model_samples=model_samples,
        baseline_samples=baseline_samples,
        summary_config=summary_config,
    )


def summarise_bootstrap_samples(
    *,
    model_samples: Sequence[float],
    baseline_samples: Sequence[float] | None,
    method: str,
    config: BootstrapSummaryConfig,
) -> Dict[str, Any]:
    """
    Return a standardised summary dictionary for bootstrap accuracy samples.

    Consolidates identical aggregation logic used across evaluation pipelines.

    :param model_samples: Bootstrap samples for the primary model accuracy.
    :param baseline_samples: Optional bootstrap samples for the baseline accuracy.
    :param method: Human-readable description of the uncertainty estimation method.
    :param config: Bootstrap resampling metadata (group count, rows, replicates, seed).
    :returns: Dictionary containing summary statistics for model and baseline samples.
    """

    if not model_samples:
        raise ValueError("model_samples must contain at least one element.")

    def _ci95(samples: Sequence[float]) -> Dict[str, float]:
        return {
            "low": float(np.percentile(samples, 2.5)),
            "high": float(np.percentile(samples, 97.5)),
        }

    summary: Dict[str, Any] = {
        "method": method,
        "n_groups": int(config.n_groups),
        "n_rows": int(config.n_rows),
        "n_bootstrap": int(config.n_bootstrap),
        "seed": int(config.seed),
        "model": {
            "mean": float(np.mean(model_samples)),
            "ci95": _ci95(model_samples),
        },
    }
    if baseline_samples:
        summary["baseline"] = {
            "mean": float(np.mean(baseline_samples)),
            "ci95": _ci95(baseline_samples),
        }
    return summary


__all__ = [
    "BootstrapSummaryConfig",
    "compose_issue_slug",
    "ensure_hf_cache",
    "group_key_for_example",
    "build_participant_bootstrap_summary",
    "participant_bootstrap_summary",
    "prepare_dataset",
    "safe_div",
    "summarise_bootstrap_samples",
]
