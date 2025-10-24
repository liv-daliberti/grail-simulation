"""Functions for loading raw session and survey assets from disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from clean_data.io import (
    load_shorts_sessions,
    read_csv_if_exists,
    read_survey_with_fallback,
)
from clean_data.surveys import build_survey_index

logger = logging.getLogger("clean_grail")


def load_sessions_from_capsule(data_root: Path) -> List[dict]:
    """Return the combined longform + Shorts session logs."""

    sessions_path = data_root / "platform session data" / "sessions.json"
    with sessions_path.open("r", encoding="utf-8") as session_file:
        sessions = json.load(session_file)

    shorts_sessions = load_shorts_sessions(data_root)
    if shorts_sessions:
        sessions.extend(shorts_sessions)
        logger.info(
            "Loaded %d Shorts sessions from %s",
            len(shorts_sessions),
            data_root / "shorts" / "ytrecs_sessions_may2024.rds",
        )
    return sessions


def build_survey_index_map(capsule_root: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Load survey CSV/RDS exports and build fast lookup indexes."""

    survey_gun = read_survey_with_fallback(
        capsule_root
        / "intermediate data"
        / "gun control (issue 1)"
        / "guncontrol_qualtrics_w123_clean.csv",
        capsule_root
        / "results"
        / "intermediate data"
        / "gun control (issue 1)"
        / "guncontrol_qualtrics_w123_clean.csv",
    )
    survey_wage_sources: List[pd.DataFrame] = []

    survey_wage_qualtrics = read_survey_with_fallback(
        capsule_root
        / "intermediate data"
        / "minimum wage (issue 2)"
        / "qualtrics_w12_clean.csv",
        capsule_root
        / "results"
        / "intermediate data"
        / "minimum wage (issue 2)"
        / "qualtrics_w12_clean.csv",
    )
    if not survey_wage_qualtrics.empty:
        survey_wage_sources.append(survey_wage_qualtrics)

    for folder in [
        capsule_root / "intermediate data" / "minimum wage (issue 2)",
        capsule_root / "results" / "intermediate data" / "minimum wage (issue 2)",
    ]:
        if not folder.exists():
            continue
        for csv_path in sorted(folder.glob("yg_*_clean.csv")):
            yougov_df = read_csv_if_exists(csv_path)
            if yougov_df.empty:
                continue
            survey_wage_sources.append(yougov_df)

    shorts_survey = read_survey_with_fallback(
        capsule_root
        / "intermediate data"
        / "shorts"
        / "qualtrics_w12_clean_ytrecs_may2024.csv",
        capsule_root
        / "results"
        / "intermediate data"
        / "shorts"
        / "qualtrics_w12_clean_ytrecs_may2024.csv",
    )
    if not shorts_survey.empty:
        survey_wage_sources.append(shorts_survey)

    if survey_wage_sources:
        survey_wage = pd.concat(
            survey_wage_sources,
            ignore_index=True,
            sort=False,
        )
        if "urlid" in survey_wage.columns:
            dedupe_subset = ["urlid"]
            if "topic_id" in survey_wage.columns:
                dedupe_subset.append("topic_id")
            survey_wage = (
                survey_wage.drop_duplicates(subset=dedupe_subset, keep="first")
                .reset_index(drop=True)
            )
    else:
        survey_wage = pd.DataFrame()

    return {
        "gun_control": build_survey_index(survey_gun),
        "minimum_wage": build_survey_index(survey_wage),
    }


__all__ = [
    "build_survey_index_map",
    "load_sessions_from_capsule",
]
