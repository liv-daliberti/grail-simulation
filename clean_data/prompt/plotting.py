"""Plotting utilities used by prompt statistics."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import SeriesPair, convert_numeric


def _prettify_category(value: str) -> str:
    """Format a categorical value for axis tick labels."""
    if value is None:
        return ""
    text = str(value)
    if text.isnumeric() or text.upper() in {"YES", "NO"}:
        return text
    pretty = text.replace("_", " ").strip()
    if len(pretty) <= 35:
        return pretty.title()
    return pretty


def plot_numeric_hist(
    train_vals: pd.Series,
    val_vals: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    """Plot side-by-side numeric histograms for train and validation splits."""
    train_vals = train_vals.dropna()
    val_vals = val_vals.dropna()
    all_vals = pd.concat([train_vals, val_vals])
    if all_vals.empty:
        return
    bins = min(30, max(10, int(np.sqrt(len(all_vals)))))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    axes[0].hist(train_vals, bins=bins, color="#1f77b4", alpha=0.85)
    axes[0].set_title("Train")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Count")

    axes[1].hist(val_vals, bins=bins, color="#ff7f0e", alpha=0.85)
    axes[1].set_title("Validation")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Count")

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25, top=0.88)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_numeric_hist_by_issue(  # pylint: disable=too-many-locals
    pair: SeriesPair,
    title: str,
    output_path: Path,
) -> None:
    """Render per-issue numeric histograms, falling back to global plots."""

    def _clean_numeric(values: pd.Series) -> pd.Series:
        return pd.to_numeric(values.dropna().astype(str), errors="coerce").dropna()

    if not pair.has_issue():
        numeric_train = convert_numeric(pair.train_series) or pd.Series(dtype=float)
        numeric_val = convert_numeric(pair.val_series) or pd.Series(dtype=float)
        plot_numeric_hist(numeric_train, numeric_val, title, output_path)
        return

    issue_data = pair.split_by_issue(_clean_numeric)
    if not issue_data:
        numeric_train = convert_numeric(pair.train_series)
        numeric_val = convert_numeric(pair.val_series)
        if numeric_train is None and numeric_val is None:
            return
        plot_numeric_hist(
            numeric_train or pd.Series(dtype=float),
            numeric_val or pd.Series(dtype=float),
            title,
            output_path,
        )
        return

    rows = len(issue_data)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows), sharex=True)
    axes = np.atleast_2d(axes)

    for row_idx, (issue_name, train_subset, val_subset) in enumerate(issue_data):
        combined = pd.concat([train_subset, val_subset])
        bins = 10 if combined.empty else min(30, max(10, int(np.sqrt(len(combined)))))
        issue_label = issue_name.replace("_", " ").title()
        ax_train, ax_val = axes[row_idx]

        if train_subset.empty:
            ax_train.text(0.5, 0.5, "No data", ha="center", va="center")
            ax_train.axis("off")
        else:
            ax_train.hist(train_subset, bins=bins, color="#1f77b4", alpha=0.85)
            ax_train.set_ylabel("Count")
        ax_train.set_title(f"{issue_label} — Train")
        ax_train.set_xlabel("Value")

        if val_subset.empty:
            ax_val.text(0.5, 0.5, "No data", ha="center", va="center")
            ax_val.axis("off")
        else:
            ax_val.hist(val_subset, bins=bins, color="#ff7f0e", alpha=0.85)
        ax_val.set_title(f"{issue_label} — Validation")
        ax_val.set_xlabel("Value")

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_categorical_hist(  # pylint: disable=too-many-locals
    train_vals: pd.Series,
    val_vals: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    """Plot side-by-side categorical histograms for train and validation splits."""
    cleaned_train = (
        train_vals.dropna().astype(str).str.strip().replace("", np.nan).dropna()
    )
    cleaned_val = (
        val_vals.dropna().astype(str).str.strip().replace("", np.nan).dropna()
    )
    if cleaned_train.empty and cleaned_val.empty:
        return

    categories = [
        str(cat)
        for cat in sorted(set(cleaned_train.unique()) | set(cleaned_val.unique()))
    ]
    numeric_check = pd.to_numeric(pd.Series(categories), errors="coerce")
    if numeric_check.notna().all():
        numeric_unique = sorted({int(x) for x in numeric_check.dropna().astype(int)})
        if set(numeric_unique).issubset({0, 1}):
            categories = ["0", "1"]
    if len(categories) == 1:
        categories = categories + categories  # duplicate for consistent plotting

    train_counts = cleaned_train.value_counts()
    val_counts = cleaned_val.value_counts()

    fig_width = max(6, len(categories) * 0.8)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4), sharey=False)
    display_labels = [_prettify_category(cat) for cat in categories]
    for ax, counts, split in zip(axes, [train_counts, val_counts], ["train", "validation"]):
        heights = [counts.get(cat, 0) for cat in categories]
        positions = np.arange(len(categories))
        ax.bar(positions, heights, color="#1f77b4" if split == "train" else "#ff7f0e", alpha=0.85)
        ax.set_title(split.capitalize())
        ax.set_ylabel("Count")
        ax.set_xticks(positions)
        ax.set_xticklabels(display_labels, rotation=35, ha="right")

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4, top=0.88)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_categorical_hist_by_issue(  # pylint: disable=too-many-locals,cell-var-from-loop
    pair: SeriesPair,
    title: str,
    output_path: Path,
) -> None:
    """Render per-issue categorical histograms, falling back to global plots."""

    def _clean(series: pd.Series) -> pd.Series:
        return (
            series.dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
        )

    if not pair.has_issue():
        plot_categorical_hist(pair.train_series, pair.val_series, title, output_path)
        return

    issue_data: List[Tuple[str, pd.Series, pd.Series, List[str]]] = []
    for issue_name, train_subset, val_subset in pair.split_by_issue(_clean):
        combined = pd.concat([train_subset, val_subset], ignore_index=True)
        categories = sorted({str(cat) for cat in combined.unique()})
        issue_data.append((issue_name, train_subset, val_subset, categories))

    if not issue_data:
        plot_categorical_hist(pair.train_series, pair.val_series, title, output_path)
        return

    rows = len(issue_data)
    max_category_count = max((len(categories) for _, _, _, categories in issue_data), default=0)
    fig_width = max(8, max_category_count * 0.6)
    fig, axes = plt.subplots(rows, 2, figsize=(fig_width, 4 * rows), sharex=False)
    axes = np.atleast_2d(axes)

    for row_idx, (issue_name, train_subset, val_subset, categories) in enumerate(issue_data):
        if categories:
            numeric_check = pd.to_numeric(pd.Series(categories), errors="coerce")
            if numeric_check.notna().all():
                numeric_unique = sorted({int(x) for x in numeric_check.dropna().astype(int)})
                if set(numeric_unique).issubset({0, 1}):
                    categories = ["0", "1"]

        issue_label = issue_name.replace("_", " ").title()
        display_labels = [_prettify_category(cat) for cat in categories]
        ax_train, ax_val = axes[row_idx]

        positions = np.arange(len(categories))
        train_counts = train_subset.value_counts()
        train_heights = [train_counts.get(cat, 0) for cat in categories]
        ax_train.bar(positions, train_heights, color="#1f77b4", alpha=0.85)
        ax_train.set_title(f"{issue_label} — Train")
        ax_train.set_ylabel("Count")
        ax_train.set_xticks(positions)
        ax_train.set_xticklabels(display_labels, rotation=35, ha="right")

        val_counts = val_subset.value_counts()
        val_heights = [val_counts.get(cat, 0) for cat in categories]
        ax_val.bar(positions, val_heights, color="#ff7f0e", alpha=0.85)
        ax_val.set_title(f"{issue_label} — Validation")
        ax_val.set_ylabel("Count")
        ax_val.set_xticks(positions)
        ax_val.set_xticklabels(display_labels, rotation=35, ha="right")

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35, top=0.9)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


__all__ = [
    "plot_numeric_hist",
    "plot_numeric_hist_by_issue",
    "plot_categorical_hist",
    "plot_categorical_hist_by_issue",
]
