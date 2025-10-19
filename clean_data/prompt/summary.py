"""Summarization helpers for prompt statistics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..prompt_constants import DEMOGRAPHIC_FEATURE_KEYS, PROMPT_FEATURE_GROUPS
from .plotting import (
    plot_categorical_hist,
    plot_categorical_hist_by_issue,
    plot_numeric_hist,
    plot_numeric_hist_by_issue,
)
from .utils import (
    SeriesPair,
    categorical_summary,
    clean_viewer_profile,
    convert_numeric,
    count_prior_history,
    dedupe_participants,
    feature_label,
    is_nanlike,
    non_missing,
    numeric_summary,
    participant_stats,
    series_from_columns,
)


def summarize_feature(
    pair: SeriesPair,
    label: str,
    output_path: Path,
) -> Dict[str, Dict[str, float | int]]:
    """Render plots for a feature and return summary statistics."""

    numeric_train = convert_numeric(pair.train_series)
    numeric_val = convert_numeric(pair.val_series)

    if numeric_train is not None and numeric_val is not None:
        if pair.has_issue():
            plot_numeric_hist_by_issue(pair, title=label, output_path=output_path)
        else:
            plot_numeric_hist(numeric_train, numeric_val, title=label, output_path=output_path)
        return {
            "train": numeric_summary(numeric_train),
            "validation": numeric_summary(numeric_val),
        }

    if pair.has_issue():
        plot_categorical_hist_by_issue(pair, title=label, output_path=output_path)
    else:
        plot_categorical_hist(
            pair.train_series,
            pair.val_series,
            title=label,
            output_path=output_path,
        )
    return {
        "train": categorical_summary(pair.train_series),
        "validation": categorical_summary(pair.val_series),
    }


def summarize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    figures_dir: Path,
) -> Tuple[Dict[str, Dict[str, Dict[str, float | int]]], List[str]]:
    """Generate plots/statistics for all prompt features."""

    summary: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    skipped: List[str] = []

    for feature_name, columns in PROMPT_FEATURE_GROUPS.items():
        train_series = series_from_columns(train_df, columns)
        val_series = series_from_columns(val_df, columns)
        train_present = non_missing(train_series)
        val_present = non_missing(val_series)
        if train_present.empty and val_present.empty:
            skipped.append(feature_name)
            continue

        pair = SeriesPair(train_present, val_present, train_df, val_df)
        fig_path = figures_dir / f"{feature_name}.png"
        label = feature_label(feature_name)
        summary[feature_name] = summarize_feature(pair, label, fig_path)

    return summary, skipped


def profile_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    profile_col: Optional[str],
) -> Dict[str, Dict[str, float]]:
    """Return profile availability counts for each split."""

    if profile_col is not None:
        train_series = clean_viewer_profile(train_df.get(profile_col, pd.Series(dtype=object)))
        val_series = clean_viewer_profile(val_df.get(profile_col, pd.Series(dtype=object)))
    else:
        train_series = pd.Series(["" for _ in range(len(train_df))], dtype=str)
        val_series = pd.Series(["" for _ in range(len(val_df))], dtype=str)

    return {
        "train": {
            "rows": float(len(train_series)),
            "missing_profile": float((train_series == "").sum()),
        },
        "validation": {
            "rows": float(len(val_series)),
            "missing_profile": float((val_series == "").sum()),
        },
    }


def prior_history_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    figures_dir: Path,
) -> Tuple[Dict[str, Dict[int, int]], Path]:
    """Compute prior-history counts and render the corresponding chart."""

    train_prior = train_df.apply(count_prior_history, axis=1)
    val_prior = val_df.apply(count_prior_history, axis=1)
    summary = {
        "train": dict(Counter(train_prior)),
        "validation": dict(Counter(val_prior)),
    }
    fig_path = figures_dir / "prior_history_counts.png"
    plot_categorical_hist(
        train_prior.astype(str),
        val_prior.astype(str),
        title="Number of prior videos",
        output_path=fig_path,
    )
    return summary, fig_path


def n_options_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    figures_dir: Path,
) -> Tuple[Dict[str, Dict[int, int]], Path]:
    """Summarize slate sizes and render their distribution chart."""

    train_series = pd.to_numeric(
        train_df.get("n_options", pd.Series(dtype="float64")), errors="coerce"
    ).dropna().astype(int)
    val_series = pd.to_numeric(
        val_df.get("n_options", pd.Series(dtype="float64")), errors="coerce"
    ).dropna().astype(int)
    summary = {
        "train": dict(Counter(train_series.tolist())),
        "validation": dict(Counter(val_series.tolist())),
    }
    fig_path = figures_dir / "slate_size_counts.png"
    plot_categorical_hist(
        train_series.astype(str),
        val_series.astype(str),
        title="Slate size (n_options)",
        output_path=fig_path,
    )
    return summary, fig_path


def demographic_missing_summary(  # pylint: disable=too-many-locals
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    figures_dir: Path,
) -> Tuple[Dict[str, int], Path]:
    """Report rows missing all demographic feature columns and render a bar chart."""

    demo_columns = [
        col
        for key in DEMOGRAPHIC_FEATURE_KEYS
        for col in PROMPT_FEATURE_GROUPS.get(key, [])
    ]

    def _demo_missing_series(df: pd.DataFrame) -> pd.Series:
        existing = [col for col in demo_columns if col in df.columns]
        if not existing:
            return pd.Series([True] * len(df), index=df.index)
        return df[existing].apply(
            lambda row: all(is_nanlike(row.get(col)) for col in existing),
            axis=1,
        )

    train_missing = _demo_missing_series(train_df)
    val_missing = _demo_missing_series(val_df)
    counts = {
        "train": int(train_missing.sum()),
        "validation": int(val_missing.sum()),
    }

    fig_path = figures_dir / "demographic_missing_counts.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    splits = list(counts.keys())
    values = [counts[split] for split in splits]
    ax.bar(splits, values, color=["#2ca02c", "#d62728"], alpha=0.85)
    ax.set_title("Rows missing all demographic fields")
    ax.set_ylabel("Count")
    if values:
        offset = max(values) * 0.02 or 0.5
        for idx, val in enumerate(values):
            ax.text(idx, val + offset, str(val), ha="center")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return counts, fig_path


def unique_content_counts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """Return counts of unique content per split."""

    def _count_unique(df: pd.DataFrame, column: str) -> int:
        series = df.get(column, pd.Series(dtype=object))
        if series is None or series.empty:
            return 0
        cleaned = series.dropna().astype(str).str.strip().replace("", np.nan).dropna()
        return int(cleaned.nunique())

    def _counts_for_split(df: pd.DataFrame) -> Dict[str, int]:
        return {
            "current_video_ids": _count_unique(df, "current_video_id"),
            "gold_video_ids": _count_unique(df, "gold_id"),
            "slate_texts": _count_unique(df, "slate_text"),
            "state_texts": _count_unique(df, "state_text"),
        }

    return {
        "train": _counts_for_split(train_df),
        "validation": _counts_for_split(val_df),
    }


def participant_counts_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """Summarize participant counts for each split and overall."""

    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    return {
        "train": participant_stats(dedupe_participants(train_df)),
        "validation": participant_stats(dedupe_participants(val_df)),
        "overall": participant_stats(dedupe_participants(combined_df)),
    }


__all__ = [
    "summarize_feature",
    "summarize_features",
    "profile_summary",
    "prior_history_summary",
    "n_options_summary",
    "demographic_missing_summary",
    "unique_content_counts",
    "participant_counts_summary",
]
