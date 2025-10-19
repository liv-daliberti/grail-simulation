"""Survey and participant allow-list helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from clean_data.helpers import (
    _MISSING_STRINGS,
    _is_missing_value,
    _normalize_identifier,
    _normalize_urlid,
)
from clean_data.io import read_csv_if_exists

log = logging.getLogger("clean_grail")

DEMOGRAPHIC_COLUMNS = [
    "age",
    "gender",
    "q26",
    "q29",
    "race",
    "ethnicity",
    "q31",
    "income",
    "household_income",
    "pid1",
    "ideo1",
    "freq_youtube",
    "college",
]


def build_survey_index(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Create a mapping from ``urlid`` to associated survey rows.

    :param df: Raw survey dataframe loaded from the capsule.
    :returns: Mapping of normalized ``urlid`` to a list of matching rows.
    """

    index: Dict[str, List[Dict[str, Any]]] = {}
    if df.empty:
        return index
    columns = list(df.columns)
    if "urlid" not in columns:
        log.warning("Survey frame missing urlid column; columns=%s", columns)
        return index
    for _, row in df.iterrows():
        urlid = _normalize_urlid(row.get("urlid"))
        if not urlid:
            continue
        cleaned: Dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        index.setdefault(urlid, []).append(cleaned)
    return index


def select_survey_row(rows: List[Dict[str, Any]], topic_id: str) -> Dict[str, Any]:
    """Choose the most appropriate survey row for a session.

    :param rows: Candidate survey rows associated with a ``urlid``.
    :param topic_id: Topic identifier extracted from the session metadata.
    :returns: The best-matching survey row, or an empty dict when unavailable.
    """

    if not rows:
        return {}
    topic_id = (topic_id or "").strip()
    if topic_id:
        for row in rows:
            candidate_topic = str(row.get("topic_id") or row.get("topicID") or "").strip()
            if candidate_topic and candidate_topic == topic_id:
                return row
    return rows[0]


def infer_issue_from_topic(topic_id: str) -> Optional[str]:
    """Infer the policy issue based on a noisy topic identifier.

    :param topic_id: Topic string pulled from the interaction logs.
    :returns: Canonical issue label or ``None`` when detection fails.
    """

    topic = (topic_id or "").strip().lower()
    if not topic:
        return None
    if "wage" in topic:
        return "minimum_wage"
    if "gun" in topic:
        return "gun_control"
    if "pro" in topic or "anti" in topic:
        if "april" in topic or "shoot" in topic:
            return "gun_control"
    return None


def infer_participant_study(
    issue: str,
    survey_row: Optional[Dict[str, Any]],
    topic_id: str,
    session: Dict[str, Any],
) -> str:
    """Guess the originating study label for a participant.

    :param issue: Normalized issue label (e.g. ``gun_control``).
    :param survey_row: Selected survey entry for the participant.
    :param topic_id: Topic identifier associated with the session.
    :param session: Raw session payload used for additional hints.
    :returns: Study label such as ``study1`` or ``unknown``.
    """

    issue_norm = (issue or "").strip().lower()
    topic_norm = (topic_id or "").strip().lower()
    survey_keys = {str(key).strip().lower() for key in (survey_row or {}).keys()}

    if issue_norm == "gun_control":
        return "study1"

    if issue_norm == "minimum_wage":
        if survey_keys:
            if any(key in survey_keys for key in ("caseid", "sample", "weight")):
                return "study3"
            worker_keys = (
                "worker_id",
                "workerid",
                "assignment_id",
                "assignmentid",
                "hit_id",
                "hitid",
            )
            if any(key in survey_keys for key in worker_keys):
                return "study2"
        session_keys = {str(key).strip().lower() for key in session.keys()}
        if "shorts" in topic_norm or "2024" in topic_norm or "rabbit" in topic_norm:
            return "study4"
        if any(key.startswith("short") for key in session_keys):
            return "study4"

    return "unknown"


def load_participant_allowlists(capsule_root: Path) -> Dict[str, Dict[str, Set[str]]]:
    """Reconstruct participant allow-lists used by the original R preprocessing.

    :param capsule_root: Path to the CodeOcean capsule root directory.
    :returns: Nested mapping of issue -> allow-list buckets -> identifier sets.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements

    allowlists: Dict[str, Dict[str, Set[str]]] = {
        "gun_control": {"worker_ids": set(), "urlids": set()},
        "minimum_wage": {
            "study2_worker_ids": set(),
            "study2_urlids": set(),
            "study3_caseids": set(),
            "study4_worker_ids": set(),
            "study4_urlids": set(),
        },
    }

    def _normalize_series(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip()

    def _nonempty_mask(series: pd.Series) -> pd.Series:
        normalized = _normalize_series(series)
        lower = normalized.str.lower()
        return ~(normalized.eq("") | lower.isin(_MISSING_STRINGS))

    def _dedupe_earliest(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
        if df.empty or id_column not in df.columns:
            return df
        working = df.copy()
        sort_columns: List[str] = []
        if "start_time2" in working.columns:
            working["_sort_start_time2"] = pd.to_datetime(
                working["start_time2"], errors="coerce", utc=True
            )
            sort_columns.append("_sort_start_time2")
        if "start_time" in working.columns:
            numeric = pd.to_numeric(working["start_time"], errors="coerce")
            working["_sort_start_time"] = pd.to_datetime(
                numeric, unit="ms", errors="coerce", utc=True
            )
            sort_columns.append("_sort_start_time")
        if not sort_columns:
            sort_columns = [id_column]
        else:
            sort_columns.append(id_column)
        working = working.sort_values(by=sort_columns, kind="mergesort")
        deduped = working.drop_duplicates(subset=[id_column], keep="first")
        deduped = deduped.drop(columns=["_sort_start_time2", "_sort_start_time"], errors="ignore")
        return deduped

    # Gun control (Study 1)
    gun_dir = capsule_root / "results" / "intermediate data" / "gun control (issue 1)"
    gun_wave1 = read_csv_if_exists(gun_dir / "guncontrol_qualtrics_w1_clean.csv")
    gun_w123 = read_csv_if_exists(gun_dir / "guncontrol_qualtrics_w123_clean.csv")
    required_wave1_cols = {"worker_id", "q87", "q89", "survey_time", "gun_index"}
    required_followup_cols = {"worker_id", "treatment_arm", "pro", "anti"}
    if not gun_wave1.empty and not gun_w123.empty:
        has_required = required_wave1_cols.issubset(gun_wave1.columns)
        has_followup = required_followup_cols.issubset(gun_w123.columns)
        if has_required and has_followup:
            wave1 = gun_wave1.copy()
            wave1["_worker_id"] = _normalize_series(wave1["worker_id"])
            mask = _nonempty_mask(wave1["_worker_id"])
            mask &= wave1["q87"].fillna("").astype(str).str.strip().eq("Quick and easy")
            mask &= wave1["q89"].fillna("").astype(str).str.strip().eq("wikiHow")
            times = pd.to_numeric(wave1["survey_time"], errors="coerce")
            mask &= times >= 120
            gun_index = pd.to_numeric(wave1["gun_index"], errors="coerce")
            mask &= gun_index.between(0.05, 0.95, inclusive="both")
            valid_wave1_workers = set(wave1.loc[mask, "_worker_id"])
            valid_wave1_workers.discard("")

            merged = gun_w123.copy()
            merged["_worker_id"] = _normalize_series(merged["worker_id"])
            if valid_wave1_workers:
                merged = merged[merged["_worker_id"].isin(valid_wave1_workers)]
            treatment_series = (
                merged["treatment_arm"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )
            merged = merged[
                (treatment_series != "control")
                & _nonempty_mask(merged["treatment_arm"])
            ]
            merged = merged[_nonempty_mask(merged["pro"])]
            merged = merged[_nonempty_mask(merged["anti"])]
            merged = _dedupe_earliest(merged, "_worker_id")
            gun_workers = {worker for worker in merged["_worker_id"] if worker}
            allowlists["gun_control"]["worker_ids"] = gun_workers
            if "urlid" in merged.columns:
                gun_urlids = {
                    _normalize_identifier(value)
                    for value in merged["urlid"].tolist()
                    if not _is_missing_value(value)
                }
            else:
                gun_urlids = set()
            allowlists["gun_control"]["urlids"] = gun_urlids
            log.info(
                "Allow-list (gun control): %d worker_ids (urlids=%d)",
                len(gun_workers),
                len(gun_urlids),
            )
        else:
            missing = (
                required_wave1_cols.difference(gun_wave1.columns)
                | required_followup_cols.difference(gun_w123.columns)
            )
            if missing:
                log.warning(
                    "Gun control allow-list skipped missing columns: %s",
                    ", ".join(sorted(missing)),
                )
    else:
        log.warning(
            "Gun control allow-list: missing wave1 or merged dataset; skipping strict filters"
        )

    wage_dir = capsule_root / "results" / "intermediate data" / "minimum wage (issue 2)"

    # Minimum wage Study 2 (MTurk)
    wage_mt = read_csv_if_exists(wage_dir / "qualtrics_w12_clean.csv")
    required_mt_cols = {
        "worker_id",
        "q87",
        "q89",
        "survey_time",
        "mw_index_w1",
        "treatment_arm",
        "pro",
        "anti",
    }
    if not wage_mt.empty:
        if required_mt_cols.issubset(wage_mt.columns):
            mt_df = wage_mt.copy()
            mt_df["_worker_id"] = _normalize_series(mt_df["worker_id"])
            mask = _nonempty_mask(mt_df["_worker_id"])
            mask &= mt_df["q87"].fillna("").astype(str).str.strip().eq("Quick and easy")
            mask &= mt_df["q89"].fillna("").astype(str).str.strip().eq("wikiHow")
            mask &= pd.to_numeric(mt_df["survey_time"], errors="coerce") >= 120
            mw_index = pd.to_numeric(mt_df["mw_index_w1"], errors="coerce")
            mask &= mw_index.between(0.025, 0.975, inclusive="both")
            mt_df = mt_df.loc[mask]
            treatment_series = mt_df["treatment_arm"].fillna("").astype(str).str.strip().str.lower()
            mt_df = mt_df[
                (treatment_series != "control")
                & _nonempty_mask(mt_df["treatment_arm"])
            ]
            mt_df = mt_df[_nonempty_mask(mt_df["pro"])]
            mt_df = mt_df[_nonempty_mask(mt_df["anti"])]
            mt_df = _dedupe_earliest(mt_df, "_worker_id")
            study2_workers = {worker for worker in mt_df["_worker_id"] if worker}
            allowlists["minimum_wage"]["study2_worker_ids"] = study2_workers
            study2_urlids = {
                _normalize_urlid(value)
                for value in mt_df.get("urlid", [])
                if isinstance(value, str) and _normalize_urlid(value)
            }
            allowlists["minimum_wage"]["study2_urlids"] = study2_urlids
            log.info("Allow-list (minimum wage Study 2): %d worker_ids", len(study2_workers))
        else:
            missing = required_mt_cols.difference(wage_mt.columns)
            log.warning(
                "Minimum wage Study 2 allow-list skipped missing columns: %s",
                ", ".join(sorted(missing)),
            )
    else:
        log.warning("Minimum wage Study 2 allow-list: dataset missing")

    # Minimum wage Study 3 (YouGov)
    wage_yg = read_csv_if_exists(wage_dir / "yg_w12_clean.csv")
    caseid_col = None
    if "caseid" in wage_yg.columns:
        caseid_col = "caseid"
    elif "CaseID" in wage_yg.columns:
        caseid_col = "CaseID"
    required_yg_cols = {"treatment_arm", "pro", "anti"}
    if not wage_yg.empty and caseid_col:
        if required_yg_cols.issubset(wage_yg.columns):
            yg_df = wage_yg.copy()
            yg_df["_caseid"] = _normalize_series(yg_df[caseid_col])
            yg_df = yg_df[_nonempty_mask(yg_df["_caseid"])]
            treatment_series = yg_df["treatment_arm"].fillna("").astype(str).str.strip().str.lower()
            yg_df = yg_df[
                (treatment_series != "control")
                & _nonempty_mask(yg_df["treatment_arm"])
            ]
            yg_df = yg_df[_nonempty_mask(yg_df["pro"])]
            yg_df = yg_df[_nonempty_mask(yg_df["anti"])]
            yg_df = _dedupe_earliest(yg_df, "_caseid")
            study3_caseids = {caseid for caseid in yg_df["_caseid"] if caseid}
            allowlists["minimum_wage"]["study3_caseids"] = study3_caseids
            log.info("Allow-list (minimum wage Study 3): %d caseids", len(study3_caseids))
        else:
            missing = required_yg_cols.difference(wage_yg.columns)
            log.warning(
                "Minimum wage Study 3 allow-list skipped missing columns: %s",
                ", ".join(sorted(missing)),
            )
    else:
        if wage_yg.empty:
            log.warning("Minimum wage Study 3 allow-list: dataset missing")
        else:
            log.warning("Minimum wage Study 3 allow-list: missing caseid column")

    # Minimum wage Study 4 (Shorts)
    shorts_path = (
        capsule_root
        / "results"
        / "intermediate data"
        / "shorts"
        / "qualtrics_w12_clean_ytrecs_may2024.csv"
    )
    wage_shorts = read_csv_if_exists(shorts_path)
    if not wage_shorts.empty:
        if "worker_id" in wage_shorts.columns:
            shorts_df = wage_shorts.copy()
            shorts_df["_worker_id"] = _normalize_series(shorts_df["worker_id"])
            mask = _nonempty_mask(shorts_df["_worker_id"])
            if "q81" in shorts_df.columns:
                mask &= shorts_df["q81"].fillna("").astype(str).str.strip().eq("Quick and easy")
            if "q82" in shorts_df.columns:
                mask &= shorts_df["q82"].fillna("").astype(str).str.strip().eq("wikiHow")
            if "video_link" in shorts_df.columns:
                mask &= _nonempty_mask(shorts_df["video_link"])
            shorts_df = shorts_df.loc[mask]
            shorts_df = _dedupe_earliest(shorts_df, "_worker_id")
            study4_workers = {worker for worker in shorts_df["_worker_id"] if worker}
            allowlists["minimum_wage"]["study4_worker_ids"] = study4_workers
            study4_urlids = {
                _normalize_urlid(value)
                for value in shorts_df.get("urlid", [])
                if isinstance(value, str) and _normalize_urlid(value)
            }
            allowlists["minimum_wage"]["study4_urlids"] = study4_urlids
            log.info("Allow-list (minimum wage Study 4): %d worker_ids", len(study4_workers))
        else:
            log.warning("Minimum wage Study 4 allow-list: missing worker_id column")
    else:
        log.warning("Minimum wage Study 4 allow-list: dataset missing")

    return allowlists


__all__ = [
    "DEMOGRAPHIC_COLUMNS",
    "build_survey_index",
    "select_survey_row",
    "infer_issue_from_topic",
    "infer_participant_study",
    "load_participant_allowlists",
]

# Backwards-compatible aliases while the refactor is in progress.
_build_survey_index = build_survey_index
_select_survey_row = select_survey_row
_infer_issue_from_topic = infer_issue_from_topic
_infer_participant_study = infer_participant_study
_load_participant_allowlists = load_participant_allowlists
