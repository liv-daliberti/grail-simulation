"""Utility functions and data structures shared by prompt analytics code.

This module abstracts dataset loading, Series manipulations, and summary
helpers that are reused by the plotting, Markdown, and CLI components of
the prompt reporting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset, load_from_disk

from ..helpers import _as_list_json
from ..prompt_constants import (
    CORE_PARTICIPANT_STUDIES,
    CORE_PROMPT_ISSUES,
    FEATURE_LABELS,
)

CORE_STUDIES_LOWER = {value.lower() for value in CORE_PARTICIPANT_STUDIES}
CORE_ISSUES_LOWER = {value.lower() for value in CORE_PROMPT_ISSUES}


@dataclass
class SeriesPair:
    """Bundle training/validation series alongside their originating dataframes."""

    train_series: pd.Series
    val_series: pd.Series
    train_df: pd.DataFrame
    val_df: pd.DataFrame

    def has_issue(self) -> bool:
        """Return True when both dataframes expose an 'issue' column."""
        return "issue" in self.train_df.columns and "issue" in self.val_df.columns

    def issue_names(self) -> List[str]:
        """Collect normalized issue names available in the pair."""
        if not self.has_issue():
            return []
        issue_candidates = pd.concat(
            [
                self.train_df.get("issue", pd.Series(dtype=str)),
                self.val_df.get("issue", pd.Series(dtype=str)),
            ],
            ignore_index=True,
        )
        cleaned = issue_candidates.dropna().astype(str).str.strip()
        return sorted({issue for issue in cleaned if issue})

    def split_by_issue(
        self,
        cleaner: Callable[[pd.Series], pd.Series],
    ) -> List[Tuple[str, pd.Series, pd.Series]]:
        """Return per-issue subsets using the provided cleaning function."""
        if not self.has_issue():
            return []

        data: List[Tuple[str, pd.Series, pd.Series]] = []
        for issue_name in self.issue_names():
            mask_train = self.train_df["issue"].astype(str) == issue_name
            mask_val = self.val_df["issue"].astype(str) == issue_name
            valid_train_idx = self.train_series.index.intersection(self.train_df.index[mask_train])
            valid_val_idx = self.val_series.index.intersection(self.val_df.index[mask_val])
            train_subset = cleaner(self.train_series.loc[valid_train_idx])
            val_subset = cleaner(self.val_series.loc[valid_val_idx])
            if train_subset.empty and val_subset.empty:
                continue
            data.append((issue_name, train_subset, val_subset))
        return data


def core_prompt_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask selecting rows from core studies and issues."""

    if df.empty:
        return pd.Series([], dtype=bool)

    mask = pd.Series([True] * len(df), index=df.index)

    if "participant_study" in df.columns:
        studies = df["participant_study"].fillna("").astype(str).str.lower()
        mask &= studies.isin(CORE_STUDIES_LOWER)

    if "issue" in df.columns:
        issues = df["issue"].fillna("").astype(str).str.lower()
        mask &= issues.isin(CORE_ISSUES_LOWER)

    return mask


def filter_core_prompt_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a dataframe down to the core studies/issues used in reporting."""
    if df.empty:
        return df.copy()
    mask = core_prompt_mask(df)
    return df.loc[mask].copy()


def canonical_slate_items(value: Any) -> Optional[Tuple[str, ...]]:
    """Normalize a slate representation to a tuple of video ids."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.ndarray):
        items: Any = value.tolist()
    else:
        items = value
    if not isinstance(items, list):
        items = _as_list_json(items)
    if not isinstance(items, list):
        return None

    identifiers: List[str] = []
    for entry in items:
        candidate: Any = entry
        if isinstance(entry, dict):
            candidate = entry.get("id") or entry.get("video_id")
        elif isinstance(entry, (list, tuple)) and entry:
            candidate = entry[0]
        text = str(candidate or "").strip()
        if text:
            identifiers.append(text)

    if not identifiers:
        return None
    return tuple(identifiers)


def ensure_dir(path: Path) -> None:
    """Create the directory hierarchy when it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_dataset_any(path_or_name: str) -> DatasetDict:
    """Load a dataset from disk or from the Hugging Face hub."""
    path = Path(path_or_name)
    if path.exists():
        return load_from_disk(str(path))
    return load_dataset(path_or_name)  # type: ignore[return-value]


def is_nanlike(value: Any) -> bool:
    """Return True when the value should be treated as missing."""
    if value is None:
        return True
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return True
    if isinstance(value, (list, dict)):
        return False
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "n/a"}


def first_present(row: pd.Series, columns: Sequence[str]) -> Optional[object]:
    """Return the first non-missing value from the provided columns."""
    for col in columns:
        if col not in row:
            continue
        value = row[col]
        if is_nanlike(value):
            continue
        return value
    return None


def series_from_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Collapse multiple candidate columns into a single best-effort series."""
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return pd.Series([None] * len(df), index=df.index, dtype="object")
    subset = df[existing]
    values = subset.apply(lambda row: first_present(row, existing), axis=1)
    return values


def convert_numeric(series: pd.Series) -> Optional[pd.Series]:
    """Coerce a series to numeric values, ignoring unconvertible entries."""
    if series.dropna().empty:
        return None
    converted = pd.to_numeric(series.dropna().astype(str), errors="coerce")
    if converted.dropna().empty:
        return None
    return converted


def non_missing(series: pd.Series) -> pd.Series:
    """Filter a series to entries that are not NaN-like."""
    if series.empty:
        return series
    mask = series.map(lambda value: not is_nanlike(value))
    return series[mask]


def numeric_summary(series: pd.Series) -> Dict[str, float]:
    """Compute count, mean, and std for a numeric series."""
    cleaned = series.dropna()
    if cleaned.empty:
        return {"count": 0.0, "mean": float("nan"), "std": float("nan")}
    return {
        "count": float(len(cleaned)),
        "mean": float(cleaned.mean()),
        "std": float(cleaned.std(ddof=0)),
    }


def categorical_summary(series: pd.Series) -> Dict[str, int]:
    """Return counts for each categorical value in the series."""
    if series.empty:
        return {}
    return {k: int(v) for k, v in series.astype(str).value_counts().to_dict().items()}


def clean_viewer_profile(series: pd.Series) -> pd.Series:
    """Normalize viewer profile sentences by stripping NaN-like values."""
    return series.fillna("").astype(str).str.strip()


def count_prior_history(row: pd.Series) -> int:
    """Count the number of prior videos available in the interaction history."""

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

    def _coerce_sequence(value: Any) -> List[Any]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        parsed = _as_list_json(value)
        return parsed if isinstance(parsed, list) else []

    watched_ids = _coerce_sequence(watched_ids)
    watched_det = _coerce_sequence(watched_det)

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


def feature_label(feature_name: str) -> str:
    """Return a human-readable label for the feature name."""
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    return feature_name.replace("_", " ").title()


def participant_ids(df: pd.DataFrame) -> pd.Series:
    """Return a best-effort participant identifier for each row."""
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


def dedupe_participants(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate rows so each participant/issue pair appears once."""
    if df.empty:
        return df
    ids = participant_ids(df)
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
    dedup = dedup.drop(
        columns=["_participant_internal_id", "_participant_issue_key"],
        errors="ignore",
    )
    dedup = dedup.reset_index(drop=True)
    return dedup


def participant_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute participant counts overall, by issue, and by study."""
    if df.empty:
        return {
            "overall": 0,
            "by_issue": {},
            "by_study": {},
            "by_issue_study": {},
        }

    participant_ids_series = participant_ids(df)
    issues = (
        df.get("issue", pd.Series(["unknown"] * len(df), index=df.index))
        .fillna("unknown")
        .astype(str)
    )
    studies = (
        df.get("participant_study", pd.Series(["unknown"] * len(df), index=df.index))
        .fillna("unknown")
        .astype(str)
    )

    ids_df = pd.DataFrame(
        {
            "participant": participant_ids_series.astype(str).str.strip(),
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
    grouped = ids_df.groupby(["issue", "study"])["participant"].nunique()
    for (issue_name, study_name), count in grouped.items():
        by_issue_study.setdefault(issue_name, {})[study_name] = int(count)

    return {
        "overall": overall,
        "by_issue": by_issue,
        "by_study": by_study,
        "by_issue_study": by_issue_study,
    }


__all__ = [
    "SeriesPair",
    "core_prompt_mask",
    "filter_core_prompt_rows",
    "canonical_slate_items",
    "ensure_dir",
    "load_dataset_any",
    "is_nanlike",
    "first_present",
    "series_from_columns",
    "convert_numeric",
    "non_missing",
    "numeric_summary",
    "categorical_summary",
    "clean_viewer_profile",
    "count_prior_history",
    "feature_label",
    "participant_ids",
    "dedupe_participants",
    "participant_stats",
]
