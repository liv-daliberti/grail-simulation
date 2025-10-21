"""Core data wrangling helpers for the political sciences replication."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from datasets import DatasetDict
except ImportError:  # pragma: no cover - optional dependency for linting
    DatasetDict = Any  # type: ignore

from clean_data.helpers import _MISSING_STRINGS

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
    data_frame: pd.DataFrame,
    before_col: str,
    after_col: str,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a 2D histogram of before vs. after opinion indices."""

    if data_frame.empty:
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        return np.zeros((bins, bins), dtype=int), bin_edges

    values_before = data_frame[before_col].to_numpy(dtype=float)
    values_after = data_frame[after_col].to_numpy(dtype=float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    hist, _, _ = np.histogram2d(values_before, values_after, bins=[bin_edges, bin_edges])
    return hist.astype(int), bin_edges


def summarise_shift(
    data_frame: pd.DataFrame,
    before_col: str,
    after_col: str,
) -> Dict[str, float]:
    """Compute summary statistics describing the opinion shift."""

    if data_frame.empty:
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

    before = data_frame[before_col].to_numpy(dtype=float)
    after = data_frame[after_col].to_numpy(dtype=float)
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


def _capsule_results_dir() -> Optional[Path]:
    """Return the repository-local path to the capsule intermediate data."""

    base = Path(__file__).resolve().parents[2] / "capsule-5416997" / "results" / "intermediate data"
    return base if base.exists() else None


def _normalize_series(series: pd.Series) -> pd.Series:
    """Return a trimmed string series with missing tokens replaced by blanks."""

    return series.fillna("").astype(str).str.strip()


def _nonempty_mask(series: pd.Series) -> pd.Series:
    """Boolean mask selecting entries that are not considered missing or blank."""

    normalized = _normalize_series(series)
    lowered = normalized.str.lower()
    return ~(normalized.eq("") | lowered.isin(_MISSING_STRINGS))


def _dedupe_earliest(data_frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Keep the earliest observation per identifier, mirroring survey filters."""

    if data_frame.empty or id_column not in data_frame.columns:
        return data_frame

    working = data_frame.copy()
    sort_cols: List[str] = []

    if "start_time2" in working.columns:
        working["_sort_start_time2"] = pd.to_datetime(
            working["start_time2"],
            errors="coerce",
            utc=True,
        )
        sort_cols.append("_sort_start_time2")
    if "start_time" in working.columns:
        numeric = pd.to_numeric(working["start_time"], errors="coerce")
        working["_sort_start_time"] = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
        sort_cols.append("_sort_start_time")

    if not sort_cols:
        sort_cols = [id_column]
    else:
        sort_cols.append(id_column)

    working = working.sort_values(by=sort_cols, kind="mergesort")
    deduped = working.drop_duplicates(subset=[id_column], keep="first")
    deduped = deduped.drop(columns=["_sort_start_time2", "_sort_start_time"], errors="ignore")
    return deduped


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric values with NaNs for unparsable entries."""

    return pd.to_numeric(series, errors="coerce")


def _study1_assignment_frame(base_dir: Path, spec: StudySpec) -> pd.DataFrame:
    """Return control/treatment assignments for Study 1 (gun control MTurk)."""

    wave1_path = base_dir / "gun control (issue 1)" / "guncontrol_qualtrics_w1_clean.csv"
    follow_path = base_dir / "gun control (issue 1)" / "guncontrol_qualtrics_w123_clean.csv"
    if not wave1_path.exists() or not follow_path.exists():
        return pd.DataFrame(columns=["participant_id", "before", "after", "assignment"])

    wave1 = pd.read_csv(wave1_path)
    follow = pd.read_csv(follow_path)
    if wave1.empty or follow.empty:
        return pd.DataFrame(columns=["participant_id", "before", "after", "assignment"])

    wave1["_worker_id"] = _normalize_series(wave1.get("worker_id", pd.Series(dtype=str)))
    mask = _nonempty_mask(wave1["_worker_id"])
    mask &= (
        wave1.get("q87", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("Quick and easy")
    )
    mask &= wave1.get("q89", pd.Series(dtype=str)).fillna("").astype(str).str.strip().eq("wikiHow")
    mask &= _numeric(wave1.get("survey_time", pd.Series(dtype=float))) >= 120
    mask &= _numeric(
        wave1.get(spec.before_column, pd.Series(dtype=float))
    ).between(0.05, 0.95, inclusive="both")
    valid_ids = set(wave1.loc[mask, "_worker_id"])
    valid_ids.discard("")

    follow["_worker_id"] = _normalize_series(follow.get("worker_id", pd.Series(dtype=str)))
    if valid_ids:
        follow = follow[follow["_worker_id"].isin(valid_ids)]

    follow = follow[_nonempty_mask(follow["_worker_id"])]
    follow = follow[_nonempty_mask(follow.get("treatment_arm", pd.Series(dtype=str)))]
    follow = _dedupe_earliest(follow, "_worker_id")

    follow[spec.before_column] = _numeric(follow.get(spec.before_column, pd.Series(dtype=float)))
    follow[spec.after_column] = _numeric(follow.get(spec.after_column, pd.Series(dtype=float)))
    follow = follow.dropna(subset=[spec.before_column, spec.after_column])

    assignment = (
        follow.get("treatment_arm", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    follow["assignment"] = np.where(assignment == "control", "control", "treatment")

    return follow.rename(
        columns={
            "_worker_id": "participant_id",
            spec.before_column: "before",
            spec.after_column: "after",
        }
    )[["participant_id", "before", "after", "assignment"]]


def _study2_assignment_frame(base_dir: Path, spec: StudySpec) -> pd.DataFrame:
    """Return control/treatment assignments for Study 2 (minimum wage MTurk)."""

    dataset_path = base_dir / "minimum wage (issue 2)" / "qualtrics_w12_clean.csv"
    if not dataset_path.exists():
        return pd.DataFrame(columns=["participant_id", "before", "after", "assignment"])

    survey_frame = pd.read_csv(dataset_path)
    if survey_frame.empty:
        return pd.DataFrame(columns=["participant_id", "before", "after", "assignment"])

    survey_frame["_worker_id"] = _normalize_series(
        survey_frame.get("worker_id", pd.Series(dtype=str))
    )
    mask = _nonempty_mask(survey_frame["_worker_id"])
    mask &= (
        survey_frame.get("q87", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("Quick and easy")
    )
    mask &= (
        survey_frame.get("q89", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("wikiHow")
    )
    mask &= _numeric(survey_frame.get("survey_time", pd.Series(dtype=float))) >= 120
    mw_index = _numeric(survey_frame.get(spec.before_column, pd.Series(dtype=float)))
    mask &= mw_index.between(0.025, 0.975, inclusive="both")
    survey_frame = survey_frame.loc[mask].copy()

    survey_frame = survey_frame[
        _nonempty_mask(survey_frame.get("treatment_arm", pd.Series(dtype=str)))
    ]
    survey_frame = _dedupe_earliest(survey_frame, "_worker_id")

    survey_frame[spec.before_column] = _numeric(
        survey_frame.get(spec.before_column, pd.Series(dtype=float))
    )
    survey_frame[spec.after_column] = _numeric(
        survey_frame.get(spec.after_column, pd.Series(dtype=float))
    )
    survey_frame = survey_frame.dropna(subset=[spec.before_column, spec.after_column])

    assignment = (
        survey_frame.get("treatment_arm", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    survey_frame["assignment"] = np.where(assignment == "control", "control", "treatment")

    return survey_frame.rename(
        columns={
            "_worker_id": "participant_id",
            spec.before_column: "before",
            spec.after_column: "after",
        }
    )[["participant_id", "before", "after", "assignment"]]


def _study3_assignment_frame(base_dir: Path, spec: StudySpec) -> pd.DataFrame:
    """Return control/treatment assignments for Study 3 (minimum wage YouGov)."""

    dataset_path = base_dir / "minimum wage (issue 2)" / "yg_w12_clean.csv"
    if not dataset_path.exists():
        return pd.DataFrame(columns=["participant_id", "before", "after", "assignment"])

    survey_frame = pd.read_csv(dataset_path)
    if survey_frame.empty:
        return pd.DataFrame(columns=["participant_id", "before", "after", "assignment"])

    case_col: Optional[str] = None
    if "caseid" in survey_frame.columns:
        case_col = "caseid"
    elif "CaseID" in survey_frame.columns:
        case_col = "CaseID"
    if case_col is None:
        return pd.DataFrame(columns=["participant_id", "before", "after", "assignment"])

    survey_frame["_caseid"] = _normalize_series(survey_frame[case_col])
    survey_frame = survey_frame[_nonempty_mask(survey_frame["_caseid"])]
    survey_frame = _dedupe_earliest(survey_frame, "_caseid")

    survey_frame[spec.before_column] = _numeric(
        survey_frame.get(spec.before_column, pd.Series(dtype=float))
    )
    survey_frame[spec.after_column] = _numeric(
        survey_frame.get(spec.after_column, pd.Series(dtype=float))
    )
    survey_frame = survey_frame.dropna(subset=[spec.before_column, spec.after_column])

    treatment_arm = (
        survey_frame.get("treatment_arm", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    has_explicit_control = treatment_arm.eq("control").any()

    if has_explicit_control:
        survey_frame["assignment"] = np.where(
            treatment_arm == "control", "control", "treatment"
        )
    else:
        dose = _numeric(survey_frame.get("treatment_dose", pd.Series(dtype=float)))
        survey_frame["assignment"] = np.where(
            dose <= 0,
            "control",
            np.where(dose > 0, "treatment", None),
        )
        survey_frame = survey_frame.dropna(subset=["assignment"])

    return survey_frame.rename(
        columns={
            "_caseid": "participant_id",
            spec.before_column: "before",
            spec.after_column: "after",
        }
    )[["participant_id", "before", "after", "assignment"]]


def load_assignment_frame(spec: StudySpec) -> pd.DataFrame:
    """Load a dataframe containing control/treatment assignments for a study."""

    base_dir = _capsule_results_dir()
    if base_dir is None:
        return pd.DataFrame(columns=["participant_id", "before", "after", "assignment"])

    if spec.key == "study1":
        return _study1_assignment_frame(base_dir, spec)
    if spec.key == "study2":
        return _study2_assignment_frame(base_dir, spec)
    if spec.key == "study3":
        return _study3_assignment_frame(base_dir, spec)
    return pd.DataFrame(columns=["participant_id", "before", "after", "assignment"])


def summarise_assignments(frame: pd.DataFrame) -> List[Dict[str, float]]:
    """Return summary statistics for each assignment group in a dataframe."""

    if frame.empty:
        return []

    summaries: List[Dict[str, float]] = []
    for assignment, group in frame.groupby("assignment"):
        metrics = summarise_shift(group, "before", "after")
        count = int(metrics["n"])
        stderr = float("nan")
        if count > 0 and not math.isnan(metrics["std_change"]):
            stderr = metrics["std_change"] / math.sqrt(max(count, 1))
        summaries.append(
            {
                "assignment": assignment,
                "n": count,
                "mean_before": metrics["mean_before"],
                "mean_after": metrics["mean_after"],
                "mean_change": metrics["mean_change"],
                "std_change": metrics["std_change"],
                "ci95": 1.96 * stderr if not math.isnan(stderr) else float("nan"),
            }
        )

    summaries.sort(
        key=lambda item: (
            0 if item["assignment"] == "control" else 1,
            item["assignment"],
        )
    )
    return summaries


def compute_treatment_regression(  # pylint: disable=too-many-locals
    combined_frame: pd.DataFrame,
) -> Dict[str, float]:
    """Estimate a study-adjusted treatment effect on opinion change.

    The model regresses ``after - before`` on a treatment indicator with
    study fixed-effects (Study 1 as the reference category).  Returns the
    coefficient, standard error, 95% CI, and two-sided p-value.
    """

    if combined_frame.empty:
        return {
            "coefficient": float("nan"),
            "stderr": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }

    working = combined_frame.copy()
    working["change"] = working["after"] - working["before"]
    working = working.replace([np.inf, -np.inf], np.nan).dropna(subset=["change"])
    if working.empty:
        return {
            "coefficient": float("nan"),
            "stderr": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }

    study_keys = sorted({str(label) for label in working["study_key"]})

    rows: List[List[float]] = []
    for _, row in working.iterrows():
        row_values: List[float] = [
            1.0,
            1.0 if str(row["assignment"]).lower() == "treatment" else 0.0,
        ]
        for study in study_keys[1:]:
            row_values.append(1.0 if str(row["study_key"]) == study else 0.0)
        rows.append(row_values)

    design_matrix = np.asarray(rows, dtype=float)
    change_values = working["change"].to_numpy(dtype=float)

    beta, residuals, rank, _ = np.linalg.lstsq(design_matrix, change_values, rcond=None)
    fitted = design_matrix @ beta
    rss = np.sum((change_values - fitted) ** 2) if residuals.size == 0 else residuals[0]
    dof = max(len(change_values) - rank, 1)
    sigma2 = rss / dof
    xtx_inv = np.linalg.inv(design_matrix.T @ design_matrix)
    cov_beta = sigma2 * xtx_inv
    stderr = math.sqrt(max(cov_beta[1, 1], 0.0))

    coef = beta[1]
    ci_low = coef - 1.96 * stderr
    ci_high = coef + 1.96 * stderr

    if stderr == 0.0:
        p_value = 0.0
    else:
        t_stat = coef / stderr
        p_value = 2.0 * 0.5 * math.erfc(abs(t_stat) / math.sqrt(2.0))

    return {
        "coefficient": float(coef),
        "stderr": float(stderr),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
    }
