"""Data models used when constructing interaction rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple


@dataclass(frozen=True)
class ParticipantIdentifiers:
    """Structured container for participant identifier fields."""

    worker_id: str = ""
    case_id: str = ""
    anon_id: str = ""
    urlid: str = ""
    session_id: str = ""


@dataclass(frozen=True)
class SessionTiming:
    """Per-session timing metadata keyed by raw/canonical video ids."""

    start: Dict[str, Any]
    end: Dict[str, Any]
    watch: Dict[str, Any]
    total: Dict[str, Any]
    delay: Dict[str, Any]


@dataclass(frozen=True)
class SessionInfo:
    """Normalized identifiers carried across per-session rows."""

    session_id: str
    anon_id: str
    topic: str
    urlid: str
    trajectory_json: str


@dataclass(frozen=True)
class AllowlistState:
    """Participant allowlist configuration derived from survey exports."""

    gun_workers: Set[str]
    wage_study2_workers: Set[str]
    wage_study3_caseids: Set[str]
    wage_study4_workers: Set[str]
    wage_study2_urlids: Set[str]
    wage_study4_urlids: Set[str]

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Dict[str, Set[str]]]) -> "AllowlistState":
        """Create an allowlist state from the nested mapping used on disk."""

        wage = mapping.get("minimum_wage", {})
        return cls(
            gun_workers=mapping.get("gun_control", {}).get("worker_ids", set()),
            wage_study2_workers=wage.get("study2_worker_ids", set()),
            wage_study3_caseids=wage.get("study3_caseids", set()),
            wage_study4_workers=wage.get("study4_worker_ids", set()),
            wage_study2_urlids=wage.get("study2_urlids", set()),
            wage_study4_urlids=wage.get("study4_urlids", set()),
        )

    def requires_enforcement(self, topic: str) -> bool:
        """Return ``True`` when the given topic should be filtered by allowlists."""

        topic_lower = topic.lower()
        if topic_lower == "gun_control":
            return bool(self.gun_workers)
        if topic_lower in {"minimum_wage", "min_wage"}:
            return any(
                (
                    self.wage_study2_workers,
                    self.wage_study3_caseids,
                    self.wage_study4_workers,
                )
            )
        return False

    def classify_wage_candidate(
        self,
        urlid: str,
        worker_candidate: str,
        case_candidate: str,
    ) -> Tuple[str, Optional[str], bool]:
        """Return ``(study_label, participant_token, valid)`` for minimum wage topics."""

        urlid_norm = urlid or ""
        if (
            self.wage_study4_urlids
            and urlid_norm
            and urlid_norm in self.wage_study4_urlids
            and worker_candidate
            and worker_candidate in self.wage_study4_workers
        ):
            return "study4", worker_candidate, True
        if (
            self.wage_study3_caseids
            and case_candidate
            and case_candidate in self.wage_study3_caseids
        ):
            return "study3", case_candidate, True
        if (
            self.wage_study2_urlids
            and urlid_norm
            and urlid_norm in self.wage_study2_urlids
            and worker_candidate
            and worker_candidate in self.wage_study2_workers
        ):
            return "study2", worker_candidate, True
        if self.wage_study2_workers and worker_candidate in self.wage_study2_workers:
            return "study2", worker_candidate, True
        if self.wage_study4_workers and worker_candidate in self.wage_study4_workers:
            return "study4", worker_candidate, True
        return "unknown", None, False


__all__ = [
    "AllowlistState",
    "ParticipantIdentifiers",
    "SessionInfo",
    "SessionTiming",
]
