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

"""Participant allowlist helpers and survey candidate resolution."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from clean_data.helpers import (
    _is_missing_value,
    _normalize_identifier,
    _parse_timestamp_ns,
)

from .models import AllowlistState, ParticipantIdentifiers


def participant_key(
    identifiers: ParticipantIdentifiers,
    fallback_counter: int,
) -> Tuple[str, int]:
    """Choose the canonical participant identifier for deduplication.

    :param identifiers: Structured participant identifiers collected from a session.
    :param fallback_counter: Monotonic counter used when no identifiers are present.
    :returns: Tuple of canonical participant token and updated fallback counter.
    """

    for candidate in (
        identifiers.worker_id,
        identifiers.case_id,
        identifiers.anon_id,
        identifiers.urlid,
        identifiers.session_id,
    ):
        val = _normalize_identifier(candidate)
        if val:
            return val, fallback_counter
    return f"anon::{fallback_counter}", fallback_counter + 1


def _candidate_start_timestamp(candidate_row: Mapping[str, Any]) -> int:
    """Return the earliest available nanosecond timestamp for the row.

    :param candidate_row: Survey row candidate under evaluation.
    :returns: Integer nanosecond timestamp used for ordering.
    """

    for field_name in ("start_time2", "start_time", "start_time_w2"):
        start_ns = _parse_timestamp_ns(candidate_row.get(field_name))
        if start_ns is not None:
            return start_ns
    return int(1e20)


def _candidate_identifiers(candidate_row: Mapping[str, Any]) -> Tuple[str, str]:
    """Extract normalized worker and case identifiers.

    :param candidate_row: Survey row candidate under evaluation.
    :returns: Tuple of normalized ``(worker_id, case_id)`` strings.
    """

    worker_candidate = _normalize_identifier(
        candidate_row.get("worker_id")
        or candidate_row.get("workerid")
        or candidate_row.get("WorkerID")
    )
    case_candidate = _normalize_identifier(
        candidate_row.get("caseid") or candidate_row.get("CaseID")
    )
    return worker_candidate, case_candidate


def _classify_candidate_topic(
    normalized_topic: str,
    urlid: str,
    allowlist: AllowlistState,
    worker_candidate: str,
    case_candidate: str,
) -> Tuple[str, Optional[str], bool]:
    """Return ``(study_label, participant_token, valid)`` for the row.

    :param normalized_topic: Canonical topic label for the session.
    :param urlid: URL identifier associated with the participant.
    :param allowlist: Allowlist state used to validate identifiers.
    :param worker_candidate: Candidate worker identifier from the survey row.
    :param case_candidate: Candidate case identifier from the survey row.
    :returns: Tuple of study label, participant token, and validity flag.
    """

    if normalized_topic == "gun_control" and allowlist.gun_workers:
        if worker_candidate and worker_candidate in allowlist.gun_workers:
            return "study1", worker_candidate, True
        return "unknown", None, False
    if normalized_topic in {"minimum_wage", "min_wage"}:
        return allowlist.classify_wage_candidate(
            urlid,
            worker_candidate,
            case_candidate,
        )
    return "unknown", None, False


def _validate_candidate_entry(
    study_label: str,
    candidate_row: Mapping[str, Any],
) -> bool:
    """Ensure the candidate row has treatment information and responses.

    :param study_label: Study label assigned to the candidate row.
    :param candidate_row: Survey row to validate for completeness.
    :returns: ``True`` when the row satisfies validation rules.
    """

    if study_label not in {"study1", "study2", "study3"}:
        return False
    treat_val = candidate_row.get("treatment_arm")
    if _is_missing_value(treat_val):
        return False
    if str(treat_val).strip().lower() == "control":
        return False
    if _is_missing_value(candidate_row.get("pro")):
        return False
    if _is_missing_value(candidate_row.get("anti")):
        return False
    return True


def _candidate_entry(
    normalized_topic: str,
    urlid: str,
    candidate_row: Mapping[str, Any],
    allowlist: AllowlistState,
) -> Optional[Tuple[int, str, str, str, str, Dict[str, Any]]]:
    """Return a candidate tuple when the survey row is eligible.

    :param normalized_topic: Canonical topic label for the session.
    :param urlid: URL identifier associated with the participant.
    :param candidate_row: Survey row candidate under evaluation.
    :param allowlist: Allowlist state used to validate the candidate.
    :returns: Candidate tuple containing ordering key, identifiers, and survey row or ``None``.
    """

    worker_candidate, case_candidate = _candidate_identifiers(candidate_row)
    start_ns = _candidate_start_timestamp(candidate_row)
    study_label, participant_token, valid = _classify_candidate_topic(
        normalized_topic,
        urlid,
        allowlist,
        worker_candidate,
        case_candidate,
    )
    if valid:
        valid = _validate_candidate_entry(study_label, candidate_row)
    if not valid:
        return None
    return (
        start_ns,
        participant_token or "",
        worker_candidate,
        case_candidate,
        study_label,
        dict(candidate_row),
    )


def candidate_entries_for_survey(
    topic: str,
    urlid: str,
    survey_rows: List[Dict[str, Any]],
    allowlist: AllowlistState,
) -> List[Tuple[int, str, str, str, str, Dict[str, Any]]]:
    """Create candidate tuples ordered by survey timestamp.

    :param topic: Topic identifier associated with the session.
    :param urlid: URL identifier for the session participant.
    :param survey_rows: Survey rows returned for the participant.
    :param allowlist: Allowlist configuration used to validate candidates.
    :returns: Sorted list of candidate tuples ranked by survey timestamp.
    """

    normalized_topic = topic.lower()
    entries = [
        candidate
        for candidate_row in survey_rows
        for candidate in [
            _candidate_entry(
                normalized_topic,
                urlid,
                candidate_row,
                allowlist,
            )
        ]
        if candidate is not None
    ]
    entries.sort(key=lambda entry: entry[0])
    return entries


__all__ = [
    "candidate_entries_for_survey",
    "participant_key",
]
