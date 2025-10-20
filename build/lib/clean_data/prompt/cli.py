"""Command-line front-end for generating prompt feature statistics.

The CLI loads a cleaned dataset, computes summary metrics, renders plots,
and writes a Markdown report describing viewer coverage, feature richness,
and participation statistics.  It is typically invoked by the main
cleaning workflow but can be used standalone for exploratory analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from collections import Counter

import pandas as pd
from datasets import DatasetDict

from .markdown import (
    ReportContext,
    ReportCounts,
    ReportFigures,
    ReportSummaries,
    build_markdown_report,
)
from .summary import (
    demographic_missing_summary,
    n_options_summary,
    participant_counts_summary,
    prior_history_summary,
    profile_summary,
    summarize_features,
    unique_content_counts,
)
from .utils import core_prompt_mask, ensure_dir, load_dataset_any


def _validate_dataset(dataset: DatasetDict, train_split: str, validation_split: str) -> None:
    if train_split not in dataset:
        raise ValueError(f"Split '{train_split}' not found in dataset")
    if validation_split not in dataset:
        raise ValueError(f"Split '{validation_split}' not found in dataset")


def _choose_profile_column(df: pd.DataFrame) -> Optional[str]:
    if "viewer_profile_sentence" in df.columns:
        return "viewer_profile_sentence"
    if "viewer_profile" in df.columns:
        return "viewer_profile"
    return None


def _write_summary_json(output_dir: Path, payload: Dict[str, Any]) -> None:
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_markdown(output_dir: Path, lines: Optional[list[str]]) -> None:
    if lines is None:
        return
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def generate_prompt_feature_report(  # pylint: disable=too-many-locals
    dataset: DatasetDict,
    output_dir: Path,
    train_split: str = "train",
    validation_split: str = "validation",
) -> None:
    """Generate exploratory prompt statistics and write plots plus summaries."""

    _validate_dataset(dataset, train_split, validation_split)

    ensure_dir(output_dir)
    figures_dir = output_dir / "figures"
    ensure_dir(figures_dir)

    train_raw = dataset[train_split].to_pandas()
    val_raw = dataset[validation_split].to_pandas()

    train_mask = core_prompt_mask(train_raw)
    val_mask = core_prompt_mask(val_raw)

    train_df = train_raw.loc[train_mask].reset_index(drop=True)
    val_df = val_raw.loc[val_mask].reset_index(drop=True)

    def _coverage_stats(df: pd.DataFrame, mask: pd.Series) -> Dict[str, Any]:
        total = int(len(df))
        included = int(mask.sum())
        excluded = total - included
        breakdown: Dict[str, int] = {}
        if excluded and "participant_study" in df.columns:
            study_series = (
                df.loc[~mask, "participant_study"]
                .fillna("unknown")
                .astype(str)
                .str.lower()
            )
            breakdown = {k: int(v) for k, v in study_series.value_counts().items()}
        return {
            "total_rows": total,
            "included_rows": included,
            "excluded_rows": excluded,
            "excluded_by_study": breakdown,
        }

    coverage_train = _coverage_stats(train_raw, train_mask)
    coverage_val = _coverage_stats(val_raw, val_mask)

    overall_breakdown = Counter(coverage_train["excluded_by_study"])
    overall_breakdown.update(coverage_val["excluded_by_study"])
    coverage_overall = {
        "total_rows": coverage_train["total_rows"] + coverage_val["total_rows"],
        "included_rows": coverage_train["included_rows"] + coverage_val["included_rows"],
        "excluded_rows": coverage_train["excluded_rows"] + coverage_val["excluded_rows"],
        "excluded_by_study": {k: int(v) for k, v in overall_breakdown.items()},
    }
    coverage_summary = {
        "train": coverage_train,
        "validation": coverage_val,
        "overall": coverage_overall,
    }

    feature_summary, skipped_features = summarize_features(train_df, val_df, figures_dir)

    profile_col = _choose_profile_column(train_df)
    profile_stats = profile_summary(train_df, val_df, profile_col)
    prior_counts, prior_fig = prior_history_summary(train_df, val_df, figures_dir)
    n_options_counts, n_options_fig = n_options_summary(train_df, val_df, figures_dir)
    demographic_counts, demographic_fig = demographic_missing_summary(train_df, val_df, figures_dir)
    unique_stats = unique_content_counts(train_df, val_df)
    participant_stats = participant_counts_summary(train_df, val_df)

    report_payload: Dict[str, Any] = {
        "feature_summary": feature_summary,
        "profile_summary": profile_stats,
        "prior_history_counts": prior_counts,
        "n_options_counts": n_options_counts,
        "demographic_missing_counts": demographic_counts,
        "unique_counts": unique_stats,
        "participant_counts": participant_stats,
        "coverage_summary": coverage_summary,
        "figures_dir": str(figures_dir),
        "missing_features": skipped_features,
    }
    _write_summary_json(output_dir, report_payload)

    counts_bundle = ReportCounts(
        prior_history=prior_counts,
        n_options=n_options_counts,
        demographic_missing=demographic_counts,
        unique_content=unique_stats,
        participant=participant_stats,
    )
    summaries = ReportSummaries(
        feature=feature_summary,
        profile=profile_stats,
        counts=counts_bundle,
        coverage=coverage_summary,
        skipped_features=skipped_features,
    )
    figures = ReportFigures(
        prior_history=prior_fig,
        n_options=n_options_fig,
        demographic=demographic_fig,
    )

    markdown_lines = build_markdown_report(
        ReportContext(
            output_dir=output_dir,
            figures_dir=figures_dir,
            summaries=summaries,
            figures=figures,
        )
    )
    _write_markdown(output_dir, markdown_lines)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the prompt statistics CLI."""
    parser = argparse.ArgumentParser(
        description="Generate prompt feature histograms and statistics.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to load_from_disk dataset or HF hub id.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for figures and summaries.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    return parser.parse_args()


def main() -> None:
    """Entrypoint for the ``prompt-stats`` command line interface."""
    args = _parse_args()
    dataset = load_dataset_any(args.dataset)
    generate_prompt_feature_report(
        dataset,
        output_dir=Path(args.output_dir),
        train_split=args.train_split,
        validation_split=args.validation_split,
    )


__all__ = ["generate_prompt_feature_report", "main"]
