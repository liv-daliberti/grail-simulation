"""Core data wrangling helpers for the political sciences replication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from datasets import DatasetDict


@dataclass(frozen=True)
class StudySpec:
    """Configuration describing one study's pre/post opinion columns."""

    key: str
    issue: str
    label: str
    before_column: str
    after_column: str
    heatmap_filename: str


def dataframe_from_splits(dataset: DatasetDict) -> pd.DataFrame:
    """Combine all dataset splits into a single pandas dataframe."""

    frames: List[pd.DataFrame] = []
    for split in dataset.values():
        if len(split):
            frames.append(split.to_pandas())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_numeric(series: pd.Series) -> pd.Series:
    """Convert a pandas series to numeric values, preserving NaNs."""

    return pd.to_numeric(series, errors="coerce")


def prepare_study_frame(frame: pd.DataFrame, spec: StudySpec) -> pd.DataFrame:
    """Filter the combined dataframe down to the rows matching a study."""

    if frame.empty:
        return frame.copy()

    mask = (frame["participant_study"] == spec.key) & (frame["issue"] == spec.issue)
    filtered = frame.loc[mask, [spec.before_column, spec.after_column]].copy()
    filtered[spec.before_column] = to_numeric(filtered[spec.before_column])
    filtered[spec.after_column] = to_numeric(filtered[spec.after_column])
    filtered = filtered.dropna(subset=[spec.before_column, spec.after_column])
    return filtered


def histogram2d_counts(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a 2D histogram of before vs. after opinion indices."""

    if df.empty:
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        return np.zeros((bins, bins), dtype=int), bin_edges

    values_before = df[before_col].to_numpy(dtype=float)
    values_after = df[after_col].to_numpy(dtype=float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    hist, _, _ = np.histogram2d(values_before, values_after, bins=[bin_edges, bin_edges])
    return hist.astype(int), bin_edges


def summarise_shift(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
) -> Dict[str, float]:
    """Compute summary statistics describing the opinion shift."""

    if df.empty:
        return {
            "n": 0,
            "mean_before": float("nan"),
            "mean_after": float("nan"),
            "mean_change": float("nan"),
            "median_change": float("nan"),
            "std_change": float("nan"),
            "share_increase": float("nan"),
            "share_decrease": float("nan"),
            "share_small_change": float("nan"),
        }

    before = df[before_col].to_numpy(dtype=float)
    after = df[after_col].to_numpy(dtype=float)
    change = after - before
    abs_change = np.abs(change)
    epsilon = 0.05  # mirrors paper's interpretation of small shifts

    return {
        "n": float(before.size),
        "mean_before": float(before.mean()),
        "mean_after": float(after.mean()),
        "mean_change": float(change.mean()),
        "median_change": float(np.median(change)),
        "std_change": float(change.std(ddof=1)),
        "share_increase": float(np.mean(change > 0.0)),
        "share_decrease": float(np.mean(change < 0.0)),
        "share_small_change": float(np.mean(abs_change <= epsilon)),
    }


def assemble_study_specs() -> Iterable[StudySpec]:
    """Return the static study specifications handled by the report."""

    return [
        StudySpec(
            key="study1",
            issue="gun_control",
            label="Study 1 – Gun Control (MTurk)",
            before_column="gun_index",
            after_column="gun_index_2",
            heatmap_filename="heatmap_study1_gun_control.png",
        ),
        StudySpec(
            key="study2",
            issue="minimum_wage",
            label="Study 2 – Minimum Wage (MTurk)",
            before_column="mw_index_w1",
            after_column="mw_index_w2",
            heatmap_filename="heatmap_study2_minimum_wage.png",
        ),
        StudySpec(
            key="study3",
            issue="minimum_wage",
            label="Study 3 – Minimum Wage (YouGov)",
            before_column="mw_index_w1",
            after_column="mw_index_w2",
            heatmap_filename="heatmap_study3_minimum_wage.png",
        ),
    ]
