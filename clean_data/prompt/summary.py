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

"""Summarization helpers for prompt statistics.

This module produces aggregate tables and plots for the cleaned prompt
dataset, delegating to plotting utilities and categorical/numeric summaries
to generate the Markdown report artifacts. All helpers are distributed
under the repository's Apache 2.0 license; refer to LICENSE for details.
"""

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
    canonical_slate_items,
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
) -> Tuple[Dict[str, Dict[str, float]], Path]:
    """Report rows missing all demographic feature columns and render a bar chart."""

    demo_columns = [
        col
        for key in DEMOGRAPHIC_FEATURE_KEYS
        for col in PROMPT_FEATURE_GROUPS.get(key, [])
    ]

    def _demo_missing_series(data_frame: pd.DataFrame) -> pd.Series:
        """Return a boolean mask indicating rows missing all demographic fields.

        :param data_frame: Dataframe to evaluate.
        :returns: Boolean series aligned with ``data_frame``.
        """

        existing = [col for col in demo_columns if col in data_frame.columns]
        if not existing:
            return pd.Series([False] * len(data_frame), index=data_frame.index)
        return data_frame[existing].apply(
            lambda row: all(is_nanlike(row.get(col)) for col in existing),
            axis=1,
        )

    train_missing = _demo_missing_series(train_df)
    val_missing = _demo_missing_series(val_df)

    def _summary(data_frame: pd.DataFrame, mask: pd.Series) -> Dict[str, float]:
        """Compute missing counts and share for a split.

        :param data_frame: Dataframe representing the split.
        :param mask: Boolean series with ``True`` indicating missing rows.
        :returns: Dictionary describing missing totals and share.
        """

        total = int(len(data_frame))
        missing = int(mask.sum())
        share = float(missing) / float(total) if total else 0.0
        return {"missing": missing, "total": total, "share": share}

    train_summary = _summary(train_df, train_missing)
    val_summary = _summary(val_df, val_missing)
    overall_total = int(train_summary["total"] + val_summary["total"])
    overall_missing = int(train_summary["missing"] + val_summary["missing"])
    overall_share = (overall_missing / overall_total) if overall_total else 0.0

    summaries = {
        "train": train_summary,
        "validation": val_summary,
        "overall": {
            "missing": overall_missing,
            "total": overall_total,
            "share": overall_share,
        },
    }

    fig_path = figures_dir / "demographic_missing_counts.png"
    fig, axes_obj = plt.subplots(figsize=(6, 4))
    splits = ["train", "validation", "overall"]
    palette = ["#2ca02c", "#d62728", "#1f77b4"]
    values = [summaries[split]["share"] * 100 for split in splits]
    bars = axes_obj.bar(splits, values, color=palette[: len(splits)], alpha=0.85)
    axes_obj.set_title("Rows missing all demographic fields")
    axes_obj.set_ylabel("Percent of rows missing")
    upper_bound = max(values) if values else 0.0
    if upper_bound == 0.0:
        axes_obj.set_ylim(0, 5)
        offset = 0.5
    else:
        offset = max(upper_bound * 0.03, 0.5)
        axes_obj.set_ylim(0, upper_bound + offset * 4)
    for idx, bar_patch in enumerate(bars):
        split = splits[idx]
        data = summaries[split]
        label = f"{values[idx]:.1f}% ({int(data['missing'])}/{int(data['total'])})"
        axes_obj.text(
            bar_patch.get_x() + bar_patch.get_width() / 2,
            bar_patch.get_height() + offset,
            label,
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return summaries, fig_path


def unique_content_counts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """Return counts of unique content per split."""

    def _count_unique(data_frame: pd.DataFrame, column: str) -> int:
        """Return the number of unique non-empty values for ``column``.

        :param data_frame: Dataframe containing the column.
        :param column: Column name to inspect.
        :returns: Count of distinct non-empty values.
        """

        series = data_frame.get(column, pd.Series(dtype=object))
        if series is None or series.empty:
            return 0
        cleaned = series.dropna().astype(str).str.strip().replace("", np.nan).dropna()
        return int(cleaned.nunique())

    def _canonical_slates(data_frame: pd.DataFrame) -> pd.Series:
        """Return a series of canonicalized slate tuples for each row.

        :param data_frame: Dataframe potentially containing slate columns.
        :returns: Series of canonical slates; empty series when unavailable.
        """

        for column_name in ("slate_items_json", "slate_items_with_meta", "slate_items"):
            column = data_frame.get(column_name)
            if column is None:
                continue
            non_null = column[column.notna()]
            if non_null.empty:
                continue
            normalized = non_null.map(canonical_slate_items).dropna()
            if not normalized.empty:
                return normalized
        return pd.Series([], dtype=object)

    def _candidate_video_count(slate_series: pd.Series) -> int:
        """Count distinct candidate video ids appearing across slates.

        :param slate_series: Series of canonical slate tuples.
        :returns: Number of unique candidate video identifiers.
        """

        candidates = set()
        for entry in slate_series:
            candidates.update(entry)
        return len(candidates)

    def _counts_for_split(split_frame: pd.DataFrame) -> Dict[str, int]:
        """Compute unique content counts for a single split dataframe.

        :param split_frame: Dataframe representing a dataset split.
        :returns: Dictionary of unique content counts for various identifiers.
        """

        slates = _canonical_slates(split_frame)
        slate_count = int(slates.nunique()) if not slates.empty else 0
        return {
            "current_video_ids": _count_unique(split_frame, "current_video_id"),
            "gold_video_ids": _count_unique(split_frame, "gold_id"),
            "candidate_video_ids": _candidate_video_count(slates) if not slates.empty else 0,
            "slate_combinations": slate_count,
            "prompt_texts": _count_unique(split_frame, "prompt"),
            "state_texts": _count_unique(split_frame, "state_text"),
        }

    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    return {
        "train": _counts_for_split(train_df),
        "validation": _counts_for_split(val_df),
        "overall": _counts_for_split(combined_df),
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
