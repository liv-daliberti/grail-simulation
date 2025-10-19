#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate exploratory statistics for the prompt features used in GRPO/GRAIL.

Given a cleaned dataset (the output of ``clean_data/clean_data.py``), this script:
  • Extracts every feature that feeds into ``prompt_builder.build_user_prompt``.
  • Compares training and validation distributions via histograms.
  • Reports coverage metrics such as the number of rows missing a profile sentence,
    prior-history depths, and slate-size breakdowns.

The figures and a Markdown summary are written to an output directory so they can
be versioned alongside dataset artifacts.

Example
-------

    python clean_data/prompt_stats.py \\
        --dataset data/cleaned_grail \\
        --output-dir reports/prompt_stats
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset, load_from_disk

# ---------------------------------------------------------------------------
# Feature definitions mirroring prompt_builder.py
# ---------------------------------------------------------------------------

PROMPT_FEATURE_GROUPS: Dict[str, Sequence[str]] = {
    # Demographics
    "age": ["age"],
    "gender": ["gender", "q26"],
    "race_ethnicity": ["race", "ethnicity", "q29"],
    "city": ["city", "city_name"],
    "state": ["state", "state_residence", "state_name"],
    "county": ["county", "county_name"],
    "zip3": ["zip3"],
    "education": ["education", "educ", "education_level", "college_desc"],
    "college": ["college"],
    "household_income": ["q31", "income", "household_income"],
    "employment_status": ["employment_status", "employment", "labor_force"],
    "occupation": ["occupation"],
    "marital_status": ["marital_status", "married"],
    "children_in_household": ["children_in_house", "kids_household"],
    "household_size": ["household_size"],
    "religion": ["religion", "relig_affiliation", "religious_affiliation"],
    "religious_attendance": ["relig_attend", "church_attend", "service_attendance"],
    "veteran": ["veteran", "military_service"],
    # Politics
    "party_identification": ["pid1", "party_id", "party_registration"],
    "party_lean": ["pid2", "party_id_lean", "party_lean"],
    "ideology": ["ideo1", "ideo2", "ideology"],
    "political_interest": ["pol_interest", "interest_politics", "political_interest"],
    "vote_2016": ["vote_2016"],
    "vote_2020": ["vote_2020"],
    "vote_2024": ["vote_2024", "vote_intent_2024", "vote_2024_intention"],
    "trump_approval": ["trump_approve", "trump_job_approval"],
    "biden_approval": ["biden_approve", "biden_job_approval"],
    "civic_engagement": ["civic_participation", "volunteering", "civic_activity"],
    # Minimum wage attitudes
    "minwage_text_r_w1": ["minwage_text_r_w1"],
    "minwage_text_r_w2": ["minwage_text_r_w2"],
    "minwage_text_r_w3": ["minwage_text_r_w3"],
    "minwage_text_w1": ["minwage_text_w1"],
    "minwage_text_w2": ["minwage_text_w2"],
    "mw_index_w1": ["mw_index_w1"],
    "mw_index_w2": ["mw_index_w2"],
    "minwage15_w1": ["minwage15_w1"],
    "minwage15_w2": ["minwage15_w2"],
    "mw_support_w1": ["mw_support_w1"],
    "mw_support_w2": ["mw_support_w2"],
    "minwage_importance": ["minwage_importance"],
    "minwage_priority": ["minwage_priority"],
    # Media habits
    "freq_youtube": ["freq_youtube"],
    "binge_youtube": ["binge_youtube"],
    "favorite_channels": ["q8", "fav_channels"],
    "popular_channels_followed": ["q78"],
    "media_diet": ["media_diet"],
    "news_consumption": ["news_consumption"],
    "news_sources": ["news_sources"],
    "news_sources_top": ["news_sources_top"],
    "news_frequency": ["news_frequency"],
    "platform_use": ["platform_use"],
    "social_media_use": ["social_media_use"],
    "news_trust": ["news_trust"],
    # Gun policy attitudes
    "gun_ownership": ["gun_own", "gunowner", "owns_gun"],
    "right_to_own_importance": ["right_to_own_importance"],
    "assault_ban": ["assault_ban"],
    "handgun_ban": ["handgun_ban"],
    "concealed_safe": ["concealed_safe"],
    "stricter_laws": ["stricter_laws"],
    "gun_index": ["gun_index"],
    "gun_index_2": ["gun_index_2"],
    "gun_enthusiasm": ["gun_enthusiasm"],
    "gun_importance": ["gun_importance"],
    "gun_priority": ["gun_priority"],
    "gun_policy": ["gun_policy"],
    "gun_identity": ["gun_identity"],
}

DEMOGRAPHIC_FEATURE_KEYS = [
    "age",
    "gender",
    "race_ethnicity",
    "city",
    "state",
    "county",
    "zip3",
    "education",
    "college",
    "household_income",
    "employment_status",
    "occupation",
    "marital_status",
    "children_in_household",
    "household_size",
    "religion",
    "religious_attendance",
    "veteran",
]

FEATURE_LABELS: Dict[str, str] = {
    "age": "Viewer age",
    "gender": "Gender",
    "race_ethnicity": "Race / ethnicity",
    "city": "City",
    "state": "State",
    "county": "County",
    "zip3": "ZIP3 prefix",
    "education": "Education level",
    "college": "College graduate flag",
    "household_income": "Household income bracket",
    "employment_status": "Employment status",
    "occupation": "Occupation text",
    "marital_status": "Marital status",
    "children_in_household": "Children in household flag",
    "household_size": "Household size",
    "religion": "Religious affiliation",
    "religious_attendance": "Religious service attendance",
    "veteran": "Veteran status",
    "party_identification": "Party identification",
    "party_lean": "Party lean",
    "ideology": "Political ideology",
    "political_interest": "Political interest",
    "vote_2016": "2016 vote recall",
    "vote_2020": "2020 vote recall",
    "vote_2024": "2024 vote intention",
    "trump_approval": "Trump approval",
    "biden_approval": "Biden approval",
    "civic_engagement": "Civic engagement",
    "minwage_text_r_w1": "Minimum wage stance (wave 1, inferred)",
    "minwage_text_r_w2": "Minimum wage stance (wave 2, inferred)",
    "minwage_text_r_w3": "Minimum wage stance (wave 3, inferred)",
    "minwage_text_w1": "Minimum wage stance (wave 1, survey)",
    "minwage_text_w2": "Minimum wage stance (wave 2, survey)",
    "mw_index_w1": "Minimum wage support index (wave 1)",
    "mw_index_w2": "Minimum wage support index (wave 2)",
    "minwage15_w1": "$15 minimum wage support (wave 1)",
    "minwage15_w2": "$15 minimum wage support (wave 2)",
    "mw_support_w1": "Supports wage increase (wave 1)",
    "mw_support_w2": "Supports wage increase (wave 2)",
    "minwage_importance": "Minimum wage importance",
    "minwage_priority": "Minimum wage priority",
    "freq_youtube": "YouTube watch frequency code",
    "binge_youtube": "Binge-watches YouTube flag",
    "favorite_channels": "Favorite channels text",
    "popular_channels_followed": "Popular channels followed",
    "media_diet": "Media diet description",
    "news_consumption": "News consumption description",
    "news_sources": "News sources list",
    "news_sources_top": "Top news sources",
    "news_frequency": "News consumption frequency",
    "platform_use": "Platform usage summary",
    "social_media_use": "Social media use",
    "news_trust": "News trust level",
    "gun_ownership": "Gun ownership flag",
    "right_to_own_importance": "Right to own importance",
    "assault_ban": "Supports assault weapons ban",
    "handgun_ban": "Supports handgun ban",
    "concealed_safe": "Believes concealed carry is safe",
    "stricter_laws": "Supports stricter gun laws",
    "gun_index": "Gun index",
    "gun_index_2": "Gun index (alternate)",
    "gun_enthusiasm": "Gun enthusiasm",
    "gun_importance": "Gun importance",
    "gun_priority": "Gun policy priority",
    "gun_policy": "Gun policy stance",
    "gun_identity": "Gun identity strength",
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _is_nanlike(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return True
    if isinstance(value, (list, dict)):
        return False
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "n/a"}


def _first_present(row: pd.Series, columns: Sequence[str]) -> Optional[object]:
    for col in columns:
        if col not in row:
            continue
        val = row[col]
        if _is_nanlike(val):
            continue
        return val
    return None


def _series_from_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return pd.Series([None] * len(df), index=df.index, dtype="object")
    subset = df[existing]
    values = subset.apply(lambda row: _first_present(row, existing), axis=1)
    return values


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_dataset(path_or_name: str) -> DatasetDict:
    path = Path(path_or_name)
    if path.exists():
        return load_from_disk(str(path))
    return load_dataset(path_or_name)  # type: ignore[return-value]


def _convert_numeric(series: pd.Series) -> Optional[pd.Series]:
    if series.dropna().empty:
        return None
    converted = pd.to_numeric(series.dropna().astype(str), errors="coerce")
    if converted.dropna().empty:
        return None
    return converted


def _prettify_category(value: str) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.isnumeric() or text.upper() in {"YES", "NO"}:
        return text
    pretty = text.replace("_", " ").strip()
    if len(pretty) <= 35:
        return pretty.title()
    return pretty


def _plot_numeric_hist(
    train_vals: pd.Series,
    val_vals: pd.Series,
    title: str,
    output_path: Path,
) -> None:
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


def _plot_numeric_hist_by_issue(
    train_series: pd.Series,
    val_series: pd.Series,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    if "issue" not in train_df.columns or "issue" not in val_df.columns:
        _plot_numeric_hist(
            _convert_numeric(train_series) or pd.Series(dtype=float),
            _convert_numeric(val_series) or pd.Series(dtype=float),
            title,
            output_path,
        )
        return

    def _clean_numeric(values: pd.Series) -> pd.Series:
        return pd.to_numeric(values.dropna().astype(str), errors="coerce").dropna()

    issue_candidates = pd.concat(
        [
            train_df.get("issue", pd.Series(dtype=str)),
            val_df.get("issue", pd.Series(dtype=str)),
        ],
        ignore_index=True,
    )
    issue_candidates = issue_candidates.dropna().astype(str).str.strip()
    issue_names = sorted({issue for issue in issue_candidates if issue})

    issue_data: List[Tuple[str, pd.Series, pd.Series]] = []
    for issue_name in issue_names:
        mask_train = train_df["issue"].astype(str) == issue_name
        mask_val = val_df["issue"].astype(str) == issue_name
        train_subset = _clean_numeric(train_series.loc[mask_train])
        val_subset = _clean_numeric(val_series.loc[mask_val])
        if train_subset.empty and val_subset.empty:
            continue
        issue_data.append((issue_name, train_subset, val_subset))

    if not issue_data:
        numeric_train = _convert_numeric(train_series) or pd.Series(dtype=float)
        numeric_val = _convert_numeric(val_series) or pd.Series(dtype=float)
        if numeric_train.empty and numeric_val.empty:
            return
        _plot_numeric_hist(numeric_train, numeric_val, title, output_path)
        return

    rows = len(issue_data)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows), sharex=True)
    axes = np.atleast_2d(axes)

    for row_idx, (issue_name, train_subset, val_subset) in enumerate(issue_data):
        combined = pd.concat([train_subset, val_subset])
        if combined.empty:
            bins = 10
        else:
            bins = min(30, max(10, int(np.sqrt(len(combined)))))

        issue_label = issue_name.replace("_", " ").title()
        ax_train = axes[row_idx, 0]
        ax_val = axes[row_idx, 1]

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


def _plot_categorical_hist(
    train_vals: pd.Series,
    val_vals: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    cleaned_train = train_vals.dropna().astype(str).str.strip().replace("", np.nan).dropna()
    cleaned_val = val_vals.dropna().astype(str).str.strip().replace("", np.nan).dropna()
    if cleaned_train.empty and cleaned_val.empty:
        return

    categories = [str(cat) for cat in sorted(set(cleaned_train.unique()) | set(cleaned_val.unique()))]
    # Detect binary variables (0/1) and enforce fixed ordering.
    as_numeric = pd.to_numeric(pd.Series(categories), errors="coerce")
    if as_numeric.notna().all():
        numeric_unique = sorted({int(x) for x in as_numeric.dropna().astype(int)})
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


def _plot_categorical_hist_by_issue(
    train_series: pd.Series,
    val_series: pd.Series,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    if "issue" not in train_df.columns or "issue" not in val_df.columns:
        _plot_categorical_hist(train_series, val_series, title, output_path)
        return
    issue_candidates = pd.concat(
        [
            train_df.get("issue", pd.Series(dtype=str)),
            val_df.get("issue", pd.Series(dtype=str)),
        ],
        ignore_index=True,
    )
    issue_candidates = issue_candidates.dropna().astype(str).str.strip()
    issue_names = sorted({issue for issue in issue_candidates if issue})

    def _clean(series: pd.Series) -> pd.Series:
        return (
            series.dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
        )

    issue_data: List[Tuple[str, pd.Series, pd.Series, List[str]]] = []
    for issue_name in issue_names:
        mask_train = train_df["issue"].astype(str) == issue_name
        mask_val = val_df["issue"].astype(str) == issue_name
        train_subset = _clean(train_series.loc[mask_train])
        val_subset = _clean(val_series.loc[mask_val])
        if train_subset.empty and val_subset.empty:
            continue
        combined = pd.concat([train_subset, val_subset], ignore_index=True)
        categories = sorted({str(cat) for cat in combined.unique()})
        issue_data.append((issue_name, train_subset, val_subset, categories))

    if not issue_data:
        _plot_categorical_hist(train_series, val_series, title, output_path)
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
        if len(categories) == 1:
            categories = categories + categories
        display_labels = [_prettify_category(cat) for cat in categories] if categories else []
        positions = np.arange(len(categories)) if categories else np.array([])

        issue_label = issue_name.replace("_", " ").title()
        ax_train = axes[row_idx, 0]
        ax_val = axes[row_idx, 1]

        if train_subset.empty or not categories:
            ax_train.text(0.5, 0.5, "No data", ha="center", va="center")
            ax_train.axis("off")
        else:
            counts = train_subset.value_counts()
            heights = [counts.get(cat, 0) for cat in categories]
            ax_train.bar(positions, heights, color="#1f77b4", alpha=0.85)
            ax_train.set_ylabel("Count")
            ax_train.set_xticks(positions)
            ax_train.set_xticklabels(display_labels, rotation=35, ha="right")
        ax_train.set_title(f"{issue_label} — Train")

        if val_subset.empty or not categories:
            ax_val.text(0.5, 0.5, "No data", ha="center", va="center")
            ax_val.axis("off")
        else:
            counts = val_subset.value_counts()
            heights = [counts.get(cat, 0) for cat in categories]
            ax_val.bar(positions, heights, color="#ff7f0e", alpha=0.85)
            ax_val.set_xticks(positions)
            ax_val.set_xticklabels(display_labels, rotation=35, ha="right")
        ax_val.set_title(f"{issue_label} — Validation")

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35, top=0.9)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _clean_viewer_profile(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _count_prior_history(row: pd.Series) -> int:
    def _last_index(items: Iterable, target: Optional[str]) -> Optional[int]:
        if target is None:
            return None
        idx = None
        for i, val in enumerate(items):
            if isinstance(val, str) and val.strip() == target:
                idx = i
        return idx

    current_id = str(row.get("current_video_id") or "").strip()
    watched_ids = row.get("watched_vids_json")
    watched_det = row.get("watched_detailed_json")

    def _coerce_sequence(value) -> List[str]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return []

    watched_ids = _coerce_sequence(watched_ids)
    watched_det = _coerce_sequence(watched_det)
    if not isinstance(watched_ids, list):
        watched_ids = []
    if not isinstance(watched_det, list):
        watched_det = []

    cur_idx = None
    if current_id:
        cur_idx = _last_index(watched_ids, current_id)
        if cur_idx is None and watched_det:
            for j in range(len(watched_det) - 1, -1, -1):
                entry = watched_det[j]
                if isinstance(entry, dict):
                    candidate = entry.get("id") or entry.get("video_id")
                    if isinstance(candidate, str) and candidate.strip() == current_id:
                        cur_idx = j
                        break
    if cur_idx is None:
        if watched_ids:
            cur_idx = len(watched_ids) - 1
        elif watched_det:
            cur_idx = len(watched_det) - 1
        else:
            return 0
    return max(0, cur_idx)


# ---------------------------------------------------------------------------
# Main reporting
# ---------------------------------------------------------------------------


def _feature_label(feature_name: str) -> str:
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    return feature_name.replace("_", " ").title()


def _participant_ids(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=str)
    participant_id_series = df.get("participant_id")
    urlid_series = df.get("urlid")
    session_series = df.get("session_id")
    participant_str = (
        participant_id_series.fillna("").astype(str).str.strip()
        if participant_id_series is not None
        else pd.Series([""] * len(df), index=df.index)
    )
    urlid_str = (
        urlid_series.fillna("").astype(str).str.strip()
        if urlid_series is not None
        else pd.Series([""] * len(df), index=df.index)
    )
    session_str = (
        session_series.fillna("").astype(str).str.strip()
        if session_series is not None
        else pd.Series([""] * len(df), index=df.index)
    )
    participant = participant_str
    missing_mask = participant.eq("") | participant.str.lower().isin({"nan", "none", "null"})
    participant = participant.mask(missing_mask, urlid_str)
    missing_mask = participant.eq("") | participant.str.lower().isin({"nan", "none", "null"})
    participant = participant.mask(missing_mask, session_str)
    still_missing = participant.eq("") | participant.str.lower().isin({"nan", "none", "null"})
    if still_missing.any():
        fallback_index = pd.Series(df.index.astype(str), index=df.index)
        participant = participant.mask(still_missing, fallback_index)
    return participant


def _dedupe_participants(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ids = _participant_ids(df)
    issue_series = df.get("issue")
    if issue_series is None:
        issue_series = pd.Series(["unknown"] * len(df), index=df.index)
    issue_series = issue_series.fillna("unknown").astype(str).str.strip()
    dedup = df.copy()
    dedup = dedup.assign(
        _participant_internal_id=ids,
        _participant_issue_key=ids.astype(str).str.strip() + "||" + issue_series,
    )
    dedup = dedup.drop_duplicates(subset=["_participant_issue_key"])
    dedup = dedup.drop(columns=["_participant_internal_id", "_participant_issue_key"], errors="ignore")
    dedup = dedup.reset_index(drop=True)
    return dedup


def _participant_stats(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "overall": 0,
            "by_issue": {},
            "by_study": {},
            "by_issue_study": {},
        }

    participant_ids = _participant_ids(df)
    issues = df.get("issue", pd.Series(["unknown"] * len(df), index=df.index)).fillna("unknown").astype(str)
    studies = df.get("participant_study", pd.Series(["unknown"] * len(df), index=df.index)).fillna("unknown").astype(str)

    ids_df = pd.DataFrame(
        {
            "participant": participant_ids.astype(str).str.strip(),
            "issue": issues.str.strip(),
            "study": studies.str.strip(),
        }
    )

    lower_participant = ids_df["participant"].str.lower()
    mask = lower_participant.isin({"", "nan", "none", "null"})
    ids_df = ids_df.loc[~mask]

    overall = int(ids_df["participant"].nunique())
    by_issue = {k: int(v) for k, v in ids_df.groupby("issue")["participant"].nunique().items()}
    by_study = {k: int(v) for k, v in ids_df.groupby("study")["participant"].nunique().items()}

    by_issue_study: Dict[str, Dict[str, int]] = {}
    for (issue_name, study_name), count in ids_df.groupby(["issue", "study"])["participant"].nunique().items():
        by_issue_study.setdefault(issue_name, {})[study_name] = int(count)

    return {
        "overall": overall,
        "by_issue": by_issue,
        "by_study": by_study,
        "by_issue_study": by_issue_study,
    }


def generate_prompt_feature_report(
    dataset: DatasetDict,
    output_dir: Path,
    train_split: str = "train",
    validation_split: str = "validation",
) -> None:
    _ensure_dir(output_dir)
    if train_split not in dataset:
        raise ValueError(f"Split '{train_split}' not found in dataset")
    if validation_split not in dataset:
        raise ValueError(f"Split '{validation_split}' not found in dataset")

    train_df = dataset[train_split].to_pandas()
    val_df = dataset[validation_split].to_pandas()

    train_df_unique = _dedupe_participants(train_df)
    val_df_unique = _dedupe_participants(val_df)

    figures_dir = output_dir / "figures"
    _ensure_dir(figures_dir)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    profile_col = "viewer_profile_sentence" if "viewer_profile_sentence" in train_df.columns else (
        "viewer_profile" if "viewer_profile" in train_df.columns else None
    )
    has_issue_column = "issue" in train_df.columns and "issue" in val_df.columns

    for feature_name, columns in PROMPT_FEATURE_GROUPS.items():
        train_series = _series_from_columns(train_df, columns)
        val_series = _series_from_columns(val_df, columns)

        numeric_train = _convert_numeric(train_series)
        numeric_val = _convert_numeric(val_series)

        fig_path = figures_dir / f"{feature_name}.png"
        label = _feature_label(feature_name)
        if numeric_train is not None and numeric_val is not None:
            if has_issue_column:
                _plot_numeric_hist_by_issue(
                    train_series,
                    val_series,
                    train_df,
                    val_df,
                    title=label,
                    output_path=fig_path,
                )
            else:
                _plot_numeric_hist(
                    numeric_train,
                    numeric_val,
                    title=label,
                    output_path=fig_path,
                )
            stats = {
                "train": {
                    "count": float(len(numeric_train.dropna())),
                    "mean": float(numeric_train.dropna().mean()),
                    "std": float(numeric_train.dropna().std(ddof=0)),
                },
                "validation": {
                    "count": float(len(numeric_val.dropna())),
                    "mean": float(numeric_val.dropna().mean()),
                    "std": float(numeric_val.dropna().std(ddof=0)),
                },
            }
        else:
            if has_issue_column:
                _plot_categorical_hist_by_issue(
                    train_series,
                    val_series,
                    train_df,
                    val_df,
                    title=label,
                    output_path=fig_path,
                )
            else:
                _plot_categorical_hist(
                    train_series,
                    val_series,
                    title=label,
                    output_path=fig_path,
                )
            stats = {
                "train": {k: int(v) for k, v in train_series.dropna().astype(str).value_counts().to_dict().items()},
                "validation": {k: int(v) for k, v in val_series.dropna().astype(str).value_counts().to_dict().items()},
            }
        summary[feature_name] = stats

    # Diagnostics
    if profile_col is not None:
        profile_train = _clean_viewer_profile(train_df.get(profile_col, pd.Series(dtype=object)))
        profile_val = _clean_viewer_profile(val_df.get(profile_col, pd.Series(dtype=object)))
    else:
        profile_train = pd.Series([""] * len(train_df), dtype=str)
        profile_val = pd.Series([""] * len(val_df), dtype=str)
    profile_summary = {
        "train": {
            "rows": float(len(profile_train)),
            "missing_profile": float((profile_train == "").sum()),
        },
        "validation": {
            "rows": float(len(profile_val)),
            "missing_profile": float((profile_val == "").sum()),
        },
    }

    train_prior = train_df.apply(_count_prior_history, axis=1)
    val_prior = val_df.apply(_count_prior_history, axis=1)
    prior_counts = {
        "train": dict(Counter(train_prior)),
        "validation": dict(Counter(val_prior)),
    }

    n_options_train_series = pd.to_numeric(
        train_df.get("n_options", pd.Series(dtype="float64")), errors="coerce"
    ).dropna().astype(int)
    n_options_val_series = pd.to_numeric(
        val_df.get("n_options", pd.Series(dtype="float64")), errors="coerce"
    ).dropna().astype(int)
    n_options_train = Counter(n_options_train_series.tolist())
    n_options_val = Counter(n_options_val_series.tolist())
    n_options_summary = {
        "train": dict(n_options_train),
        "validation": dict(n_options_val),
    }

    # Plot prior-history distribution
    prior_fig = figures_dir / "prior_history_counts.png"
    _plot_categorical_hist(
        train_prior.astype(str),
        val_prior.astype(str),
        title="Number of prior videos",
        output_path=prior_fig,
    )

    n_options_fig = figures_dir / "slate_size_counts.png"
    _plot_categorical_hist(
        n_options_train_series.astype(str),
        n_options_val_series.astype(str),
        title="Slate size (n_options)",
        output_path=n_options_fig,
    )

    fallback_text = "Profile information is unavailable."

    def _count_fallback(df: pd.DataFrame) -> int:
        if "state_text" in df.columns:
            state_series = df["state_text"].fillna("")
        elif "prompt" in df.columns:
            state_series = df["prompt"].fillna("").astype(str)
        else:
            return 0

        lines = state_series.astype(str)
        return int(lines.str.contains(re.escape(fallback_text), regex=True).sum())

    demo_columns = [col for key in DEMOGRAPHIC_FEATURE_KEYS for col in PROMPT_FEATURE_GROUPS.get(key, [])]

    def _demo_missing_series(df: pd.DataFrame) -> pd.Series:
        existing = [col for col in demo_columns if col in df.columns]
        if not existing:
            return pd.Series([True] * len(df), index=df.index)

        def _row_missing(row: pd.Series) -> bool:
            for col in existing:
                value = row.get(col)
                if not _is_nanlike(value):
                    return False
            return True

        return df[existing].apply(_row_missing, axis=1)

    demo_missing_train = _demo_missing_series(train_df)
    demo_missing_val = _demo_missing_series(val_df)
    demo_missing_counts = {
        "train": int(demo_missing_train.sum()),
        "validation": int(demo_missing_val.sum()),
    }

    demo_fig = figures_dir / "demographic_missing_counts.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    splits_demo = list(demo_missing_counts.keys())
    values_demo = [demo_missing_counts[split] for split in splits_demo]
    ax.bar(splits_demo, values_demo, color=["#2ca02c", "#d62728"], alpha=0.85)
    ax.set_title("Rows missing all demographic fields")
    ax.set_ylabel("Count")
    for idx, val in enumerate(values_demo):
        ax.text(idx, val + (max(values_demo) * 0.02 if values_demo else 0.5), str(val), ha="center")
    fig.tight_layout()
    fig.savefig(demo_fig, dpi=150)
    plt.close(fig)

    def _unique_counts_for_split(df: pd.DataFrame) -> Dict[str, int]:
        def _count_unique(series_name: str) -> int:
            series = df.get(series_name, pd.Series(dtype=object))
            if series is None or series.empty:
                return 0
            return int(series.dropna().astype(str).str.strip().replace("", np.nan).dropna().nunique())

        return {
            "current_video_ids": _count_unique("current_video_id"),
            "gold_video_ids": _count_unique("gold_id"),
            "slate_texts": _count_unique("slate_text"),
            "state_texts": _count_unique("state_text"),
        }

    unique_counts = {
        "train": _unique_counts_for_split(train_df),
        "validation": _unique_counts_for_split(val_df),
    }

    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    combined_unique = _dedupe_participants(combined_df)
    participant_counts = {
        "train": _participant_stats(train_df_unique),
        "validation": _participant_stats(val_df_unique),
        "overall": _participant_stats(combined_unique),
    }

    report = {
        "feature_summary": summary,
        "profile_summary": profile_summary,
        "prior_history_counts": prior_counts,
        "n_options_counts": n_options_summary,
        "demographic_missing_counts": demo_missing_counts,
        "unique_counts": unique_counts,
        "participant_counts": participant_counts,
        "figures_dir": str(figures_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    # Human-readable Markdown
    md_lines: List[str] = []
    md_lines.append("# Prompt Feature Report")
    md_lines.append("")
    md_lines.append(f"- Output directory: `{output_dir}`")
    md_lines.append(f"- Figures: `{figures_dir}`")
    md_lines.append("")
    md_lines.append("## Profile availability")
    md_lines.append("")
    md_lines.append("| Split | Rows | Missing profile | Share missing |")
    md_lines.append("|-------|------|-----------------|---------------|")
    for split in ("train", "validation"):
        rows = profile_summary[split]["rows"]
        missing = profile_summary[split]["missing_profile"]
        share = (missing / rows) if rows else 0.0
        md_lines.append(f"| {split} | {int(rows)} | {int(missing)} | {share:.2%} |")
    md_lines.append("")

    md_lines.append("## Prior video counts")
    md_lines.append("")
    md_lines.append("| Prior videos | Train | Validation |")
    md_lines.append("|--------------|-------|------------|")
    for count in sorted(set(prior_counts["train"]).union(prior_counts["validation"])):
        md_lines.append(
            f"| {count} | {prior_counts['train'].get(count, 0)} | {prior_counts['validation'].get(count, 0)} |"
        )
    md_lines.append("")

    md_lines.append("## Slate size distribution (`n_options`)")
    md_lines.append("")
    md_lines.append("| Slate size | Train | Validation |")
    md_lines.append("|------------|-------|------------|")
    for size in sorted(set(n_options_summary["train"]).union(n_options_summary["validation"])):
        md_lines.append(
            f"| {size} | {n_options_summary['train'].get(size, 0)} | {n_options_summary['validation'].get(size, 0)} |"
        )
    md_lines.append("")

    md_lines.append("## Unique content counts")
    md_lines.append("")
    md_lines.append("| Split | Current videos | Gold videos | Unique slates | Unique state texts |")
    md_lines.append("|-------|----------------|-------------|---------------|--------------------|")
    for split in ("train", "validation"):
        counts = unique_counts[split]
        md_lines.append(
            f"| {split} | {counts['current_video_ids']} | {counts['gold_video_ids']} | "
            f"{counts['slate_texts']} | {counts['state_texts']} |"
        )
    md_lines.append("")

    md_lines.append("## Unique participants per study and issue")
    md_lines.append("")
    md_lines.append("| Split | Issue | Study | Participants |")
    md_lines.append("|-------|-------|-------|--------------|")
    for split in ("train", "validation"):
        counts = participant_counts[split]
        by_issue_study = counts.get("by_issue_study", {})
        for issue_name in sorted(by_issue_study.keys(), key=lambda x: x.lower()):
            study_map = by_issue_study.get(issue_name, {})
            for study_name in sorted(study_map.keys(), key=lambda x: x.lower()):
                md_lines.append(f"| {split} | {issue_name} | {study_name} | {study_map[study_name]} |")
        md_lines.append(f"| {split} | all | all | {counts.get('overall', 0)} |")
    md_lines.append("")
    overall_counts = participant_counts["overall"]
    md_lines.append(f"- Overall participants (all issues): {overall_counts.get('overall', 0)}")
    for issue_name, value in sorted(overall_counts.get("by_issue", {}).items(), key=lambda x: x[0].lower()):
        md_lines.append(f"- Overall participants for {issue_name}: {value}")
    for study_name, value in sorted(overall_counts.get("by_study", {}).items(), key=lambda x: x[0].lower()):
        md_lines.append(f"- Overall participants in {study_name}: {value}")
    md_lines.append("")

    md_lines.append("## Dataset coverage notes")
    md_lines.append("")
    md_lines.append(
        "Builder note: rows missing all survey demographics (age, gender, race, income, etc.) are dropped "
        "during cleaning so every retained interaction has viewer context for the prompt builder. This removes "
        "roughly 22% of the ~33k raw interactions."
    )
    md_lines.append("")
    md_lines.append(
        "> \"The short answer is that sessions.json contains EVERYTHING. Every test run, every study. In addition to "
        "the studies that involved watching videos on the platform, it also contains sessions from the “First "
        "Impressions” study, which involved only rating thumbnails, and the “Shorts” study (Study 4 in the paper, "
        "I believe), which involved no user decisions (instead playing a sequence of predetermined videos that were "
        "either constant or increasing in their extremeness). All of these are differentiated by the topicId.\" — "
        "Emily Hu (University of Pennsylvania)"
    )
    md_lines.append("")
    md_lines.append(
        "- Original study participants: 1,650 (Study 1 — gun rights) and 5,326 (Studies 2–4 — minimum wage)."
    )
    md_lines.append(
        f"- Cleaned dataset participants captured here: {overall_counts.get('by_issue', {}).get('gun_control', 0)} "
        "(gun control) and {overall_counts.get('by_issue', {}).get('minimum_wage', 0)} (minimum wage)."
    )
    md_lines.append(
        "- Only gun-control and minimum-wage sessions are retained; other topic IDs from the capsule are excluded."
    )
    md_lines.append(
        "- All charts and counts above operate on unique participants per issue (a participant can appear once in gun control and once in minimum wage, but never twice within the same issue split)."
    )
    md_lines.append("")

    md_lines.append("## Feature figures")
    md_lines.append("")
    for feature_name in sorted(summary.keys()):
        fig_path = figures_dir / f"{feature_name}.png"
        if fig_path.exists():
            md_lines.append(f"- `{_feature_label(feature_name)}` → `{fig_path}`")
    md_lines.append(f"- `prior_history_counts` → `{prior_fig}`")
    md_lines.append(f"- `slate_size_counts` → `{n_options_fig}`")
    md_lines.append(f"- `demographic_missing_counts` → `{demo_fig}`")
    md_lines.append("")

    with (output_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate prompt feature histograms and statistics.")
    ap.add_argument("--dataset", required=True, help="Path to load_from_disk dataset or HF hub id.")
    ap.add_argument("--output-dir", required=True, help="Destination directory for figures and summaries.")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--validation-split", default="validation")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    ds = _load_dataset(args.dataset)
    generate_prompt_feature_report(
        ds,
        output_dir=Path(args.output_dir),
        train_split=args.train_split,
        validation_split=args.validation_split,
    )


if __name__ == "__main__":
    main()
