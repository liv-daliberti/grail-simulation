"""Session log parsing and feature engineering for CodeOcean exports.

Here we translate the raw capsule sessions into the intermediate dataframe
used by :mod:`clean_data.clean_data`: loading survey allow-lists, merging
per-video metadata, enforcing participant filters, and emitting one row per
viewer decision.  It is the heaviest module in the package and underpins
both the CLI and the high-level build functions.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from clean_data.helpers import (
    _as_list_json,
    _canon,
    _coerce_session_value as _helpers_coerce_session_value,
    _is_missing_value,
    _normalize_identifier,
    _parse_timestamp_ns,
    _strip_session_video_id,
)
from clean_data.io import (
    load_recommendation_tree_metadata,
    load_shorts_sessions,
    load_video_metadata,
    read_csv_if_exists,
    read_survey_with_fallback,
)
from clean_data.surveys import (
    DEMOGRAPHIC_COLUMNS,
    build_survey_index,
    infer_issue_from_topic,
    infer_participant_study,
    load_participant_allowlists,
    select_survey_row,
)

# pylint: disable=duplicate-code

log = logging.getLogger("clean_grail")


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
        """Create an allowlist state from the nested mapping used on disk.

        :param mapping: Nested allow-list mapping produced by :func:`load_participant_allowlists`.
        :returns: Fully populated :class:`AllowlistState` instance.
        """
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
        """Return ``True`` when the given topic should be filtered by allowlists.

        :param topic: Issue/topic identifier from the session logs.
        :returns: ``True`` when allow-lists must be enforced for the topic.
        """
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
        """Return ``(study_label, participant_token, valid)`` for minimum wage topics.

        :param urlid: Normalized URL identifier for the session.
        :param worker_candidate: Candidate worker identifier extracted from surveys.
        :param case_candidate: Candidate case identifier extracted from surveys.
        :returns: Tuple containing the resolved study label, identity token, and validity flag.
        """

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

def participant_key(
    identifiers: ParticipantIdentifiers,
    fallback_counter: int,
) -> Tuple[str, int]:
    """Choose the canonical participant identifier for deduplication.

    :param identifiers: Collection of candidate identifiers for a participant.
    :param fallback_counter: Counter used to synthesize ids when all candidates are missing.
    :returns: Tuple of (participant_id, next_fallback_counter).
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


def _load_sessions_from_capsule(data_root: Path) -> List[dict]:
    """Return the combined longform + Shorts session logs.

    :param data_root: Capsule ``data`` directory containing the session exports.
    :returns: List of raw session payloads.
    """

    sessions_path = data_root / "platform session data" / "sessions.json"
    with sessions_path.open("r", encoding="utf-8") as fp:
        sessions = json.load(fp)

    shorts_sessions = load_shorts_sessions(data_root)
    if shorts_sessions:
        sessions.extend(shorts_sessions)
        log.info(
            "Loaded %d Shorts sessions from %s",
            len(shorts_sessions),
            data_root / "shorts" / "ytrecs_sessions_may2024.rds",
        )
    return sessions


def _build_survey_index_map(capsule_root: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Load survey CSV/RDS exports and build fast lookup indexes.

    :param capsule_root: Path to the CodeOcean capsule root directory.
    :returns: Mapping of issue -> urlid -> list of survey rows.
    """

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


def _build_session_timings(sess: dict, raw_vids: List[str], base_vids: List[str]) -> SessionTiming:
    """Normalize per-video timing dictionaries for a session.

    :param sess: Raw session payload.
    :param raw_vids: Sequence of raw video identifiers.
    :param base_vids: Sequence of canonicalized video identifiers.
    :returns: Structured :class:`SessionTiming` container.
    """

    return SessionTiming(
        start=normalize_session_mapping(sess.get("vidStartTimes"), raw_vids, base_vids),
        end=normalize_session_mapping(sess.get("vidEndTimes"), raw_vids, base_vids),
        watch=normalize_session_mapping(sess.get("vidWatchTimes"), raw_vids, base_vids),
        total=normalize_session_mapping(sess.get("vidTotalLengths"), raw_vids, base_vids),
        delay=normalize_session_mapping(sess.get("contentStartDelay"), raw_vids, base_vids),
    )


def _video_meta(
    base_id: str,
    raw_id: str,
    tree_meta: Dict[str, Any],
    fallback_titles: Dict[str, Any],
    tree_issue_map: Dict[str, str],
) -> Dict[str, Any]:
    """Return augmented metadata for a watched video."""

    info = dict(tree_meta.get(base_id) or {})
    if raw_id and raw_id != base_id:
        raw_ids = list(info.get("raw_ids") or [])
        if raw_id not in raw_ids:
            raw_ids.append(raw_id)
        if raw_ids:
            info["raw_ids"] = raw_ids
    if not info.get("title"):
        title = fallback_titles.get(base_id) or fallback_titles.get(raw_id)
        if title:
            info["title"] = title
    if not info.get("issue"):
        issue_guess = tree_issue_map.get(base_id)
        if issue_guess:
            info["issue"] = issue_guess
    return info


def _resolve_title(meta: Dict[str, Any], fallback_id: str) -> Tuple[str, bool]:
    """Return a title for the candidate along with a missing flag."""

    title = str(meta.get("title") or "").strip()
    if title:
        return title, False
    return f"(title missing for {fallback_id})", True


def _resolve_channel(meta: Dict[str, Any]) -> Tuple[str, bool]:
    """Return the channel title with a missing flag."""

    channel = str(meta.get("channel_title") or "").strip()
    if channel:
        return channel, False
    return "(channel missing)", True


def _build_watched_details(
    raw_vids: List[str],
    base_vids: List[str],
    tree_meta: Dict[str, Any],
    fallback_titles: Dict[str, Any],
    tree_issue_map: Dict[str, str],
    timings: SessionTiming,
) -> List[Dict[str, Any]]:
    """Return per-video metadata entries for the watched sequence."""

    # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments

    details: List[Dict[str, Any]] = []
    for idx, raw_vid in enumerate(raw_vids):
        base = base_vids[idx]
        meta = _video_meta(base, raw_vid, tree_meta, fallback_titles, tree_issue_map)
        title_val, title_missing = _resolve_title(meta, base)
        channel_val, channel_missing = _resolve_channel(meta)
        entry: Dict[str, Any] = {
            "id": base,
            "raw_id": raw_vid,
            "idx": idx,
            "title": title_val,
            "title_missing": title_missing,
            "channel_title": channel_val,
            "channel_missing": channel_missing,
        }
        if meta.get("channel_id"):
            entry["channel_id"] = meta["channel_id"]
        timing_fields = {
            "start_delay_ms": timings.delay,
            "start_ms": timings.start,
            "end_ms": timings.end,
            "watch_ms": timings.watch,
            "total_length_ms": timings.total,
        }
        for field_name, timing_map in timing_fields.items():
            value = lookup_session_value(timing_map, raw_vid, base)
            if value is not None:
                entry[field_name] = value

        recs = meta.get("recs")
        if isinstance(recs, list):
            entry["recommendations"] = [
                dict(rec) for rec in recs if isinstance(rec, dict)
            ]

        details.append(entry)

    return details


def normalize_display_orders(display_orders: Any) -> Dict[int, List[str]]:
    """Convert display order mappings into canonical id sequences."""

    normalized: Dict[int, List[str]] = {}
    if not isinstance(display_orders, dict):
        return normalized
    for key, value in display_orders.items():
        if not isinstance(value, (list, tuple)):
            continue

        step_idx: Optional[int] = None
        if isinstance(key, int):
            step_idx = key
        elif isinstance(key, str):
            stripped = key.strip()
            lowered = stripped.lower()
            if "recs" not in lowered and not stripped.isdigit():
                continue

            digits = []
            for char in stripped:
                if char.isdigit():
                    digits.append(char)
                elif digits:
                    break
            if not digits:
                continue
            raw_idx = int("".join(digits))
            if "recs" in lowered and raw_idx >= 2:
                step_idx = raw_idx - 2
            else:
                step_idx = raw_idx

        if step_idx is None or step_idx < 0:
            continue

        vids: List[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            stripped = _strip_session_video_id(item)
            if stripped:
                vids.append(stripped)
        if vids:
            normalized[step_idx] = vids
    return normalized


def build_slate_items(  # pylint: disable=too-many-locals,too-many-branches
    step_index: int,
    display_orders: Dict[int, List[str]],
    recommendations: Optional[List[Dict[str, Any]]],
    tree_meta: Dict[str, Dict[str, Any]],
    fallback_titles: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    """Derive the slate items for the given interaction step."""

    candidates: List[Tuple[str, Optional[str]]] = []
    source = "display_orders"

    order_ids = display_orders.get(step_index, [])
    if order_ids:
        for vid in order_ids:
            base_vid = _strip_session_video_id(vid)
            if base_vid:
                raw_id = vid if vid != base_vid else None
                candidates.append((base_vid, raw_id))
    else:
        source = "tree_metadata"
        if not isinstance(recommendations, list):
            recommendations = []
        for rec in recommendations:
            if not isinstance(rec, dict):
                continue
            rec_id = rec.get("id") or rec.get("video_id") or rec.get("raw_id")
            rec_id = _strip_session_video_id(str(rec_id or ""))
            if not rec_id:
                continue
            raw_id = rec.get("raw_id")
            if raw_id:
                raw_id = _strip_session_video_id(str(raw_id))
            candidates.append((rec_id, raw_id if raw_id != rec_id else None))

    seen: set[str] = set()
    items: List[Dict[str, Any]] = []
    for base_id, raw_id in candidates:
        if base_id in seen:
            continue
        seen.add(base_id)
        meta = tree_meta.get(base_id, {})
        title_candidates = (
            meta.get("title"),
            fallback_titles.get(base_id),
            fallback_titles.get(raw_id) if raw_id else None,
        )
        title = next(
            (
                str(candidate).strip()
                for candidate in title_candidates
                if isinstance(candidate, str) and str(candidate).strip()
            ),
            "",
        )
        if not title and raw_id:
            title = raw_id
        if not title:
            title = base_id
        item: Dict[str, Any] = {"id": base_id, "title": title}
        if raw_id:
            item["raw_id"] = raw_id
        if meta.get("channel_title"):
            item["channel_title"] = meta["channel_title"]
        if meta.get("channel_id"):
            item["channel_id"] = meta["channel_id"]
        for stat_key in (
            "view_count",
            "like_count",
            "dislike_count",
            "favorite_count",
            "comment_count",
        ):
            stat_value = meta.get(stat_key)
            if stat_value not in (None, ""):
                item[stat_key] = stat_value
        duration_value = meta.get("duration")
        if duration_value not in (None, ""):
            item["duration"] = duration_value
        items.append(item)
    return items, source


def _session_info(sess: dict, watched_details: List[Dict[str, Any]]) -> SessionInfo:
    """Build reusable identifiers and payloads for a session.

    :param sess: Raw session payload.
    :param watched_details: Enriched metadata entries for each watched video.
    :returns: Structured :class:`SessionInfo` object.
    """

    session_id = str(sess.get("sessionID") or sess.get("session_id") or "").strip()
    anon_id = str(sess.get("anonymousFirebaseAuthUID") or "").strip()
    topic = str(sess.get("topicId") or sess.get("topicID") or "").strip()
    urlid = str(sess.get("urlid") or "").strip()
    trajectory_payload = {
        "session_id": session_id,
        "urlid": urlid,
        "topic_id": topic,
        "order": [dict(item) for item in watched_details],
    }
    trajectory_json = json.dumps(trajectory_payload)
    return SessionInfo(
        session_id=session_id,
        anon_id=anon_id,
        topic=topic,
        urlid=urlid,
        trajectory_json=trajectory_json,
    )


def _candidate_entries_for_survey(
    topic: str,
   urlid: str,
    survey_rows: List[Dict[str, Any]],
    allowlist: AllowlistState,
) -> List[Tuple[int, str, str, str, str, Dict[str, Any]]]:
    """Create candidate tuples ordered by survey timestamp.

    :param topic: Issue/topic label associated with the session.
    :param urlid: URL identifier for the session.
    :param survey_rows: Survey rows linked to the participant.
    :param allowlist: Active allow-list state for filtering participants.
    :returns: List of tuples containing candidate metadata and validity flags.
    """

    candidates: List[Tuple[int, str, str, str, str, Dict[str, Any]]] = []
    normalized_topic = topic.lower()
    for candidate_row in survey_rows:
        worker_candidate = _normalize_identifier(
            candidate_row.get("worker_id")
            or candidate_row.get("workerid")
            or candidate_row.get("WorkerID")
        )
        case_candidate = _normalize_identifier(
            candidate_row.get("caseid") or candidate_row.get("CaseID")
        )
        start_ns = _parse_timestamp_ns(candidate_row.get("start_time2"))
        if start_ns is None:
            start_ns = _parse_timestamp_ns(candidate_row.get("start_time"))
        if start_ns is None:
            start_ns = _parse_timestamp_ns(candidate_row.get("start_time_w2"))
        if start_ns is None:
            start_ns = int(1e20)

        study_label = "unknown"
        participant_token: Optional[str] = None
        valid = False

        if normalized_topic == "gun_control" and allowlist.gun_workers:
            if worker_candidate and worker_candidate in allowlist.gun_workers:
                study_label = "study1"
                participant_token = worker_candidate
                valid = True
        elif normalized_topic in {"minimum_wage", "min_wage"}:
            study_label, participant_token, valid = allowlist.classify_wage_candidate(
                urlid,
                worker_candidate,
                case_candidate,
            )

        if valid and study_label in {"study1", "study2", "study3"}:
            treat_val = candidate_row.get("treatment_arm")
            if (
                _is_missing_value(treat_val)
                or str(treat_val).strip().lower() == "control"
            ):
                valid = False
            if _is_missing_value(candidate_row.get("pro")) or _is_missing_value(
                candidate_row.get("anti")
            ):
                valid = False

        if valid:
            candidates.append(
                (
                    start_ns,
                    participant_token or "",
                    worker_candidate,
                    case_candidate,
                    study_label,
                    candidate_row,
                )
            )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates



def coerce_session_value(value: Any) -> Any:
    """Convert session log values to numeric scalars when possible.

    :param value: Raw value extracted from the session payload.
    :returns: Coerced scalar suitable for downstream serialization.
    """

    return _helpers_coerce_session_value(value)


def normalize_session_mapping(
    values: Any,
    raw_vids: List[str],
    base_vids: List[str],
) -> Dict[str, Any]:
    """Convert per-video session arrays/dicts into a standard lookup dict.

    :param values: Session metric stored as a dict or list.
    :param raw_vids: Sequence of raw video identifiers.
    :param base_vids: Sequence of canonical video identifiers.
    :returns: Mapping keyed by raw id, canonical id, and index.
    """

    mapping: Dict[str, Any] = {}
    if isinstance(values, dict):
        for key, val in values.items():
            if key is None:
                continue
            mapping[str(key)] = coerce_session_value(val)
        return mapping
    if isinstance(values, list):
        for idx, val in enumerate(values):
            coerced = coerce_session_value(val)
            if idx < len(raw_vids):
                mapping.setdefault(raw_vids[idx], coerced)
            if idx < len(base_vids):
                mapping.setdefault(base_vids[idx], coerced)
            mapping.setdefault(str(idx), coerced)
        return mapping
    return mapping


def lookup_session_value(
    mapping: Dict[str, Any],
    raw_id: str,
    base_id: str,
) -> Any:
    """Retrieve a session metric for either the raw or canonical video id.

    :param mapping: Normalized mapping produced by :func:`normalize_session_mapping`.
    :param raw_id: Raw video identifier.
    :param base_id: Canonical video identifier.
    :returns: Stored metric value or ``None`` when absent.
    """

    if not mapping:
        return None
    for key in (raw_id, base_id, str(raw_id), str(base_id)):
        if key and key in mapping:
            return mapping[key]
    return None


def load_slate_items(ex: dict) -> List[dict]:
    """Parse ``slate_items_json`` into a normalized list of ``{title, id}`` dicts.

    :param ex: Session row containing the slate metadata.
    :returns: Ordered list of slate entries suitable for prompt construction.
    """

    arr = _as_list_json(ex.get("slate_items_json"))
    out: List[dict] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        vid = (item.get("id") or "").strip()
        if title or vid:
            out.append({"title": title, "id": vid})
    return out


def derive_next_from_history(ex: dict, current_id: str) -> str:
    """Infer the next video id from the watch history when explicit labels are missing.

    :param ex: Session row containing watch history arrays.
    :param current_id: Canonical id of the current video.
    :returns: Candidate next-video id or an empty string when unavailable.
    """

    vids = _as_list_json(ex.get("watched_vids_json"))
    if current_id and isinstance(vids, list) and vids:
        try:
            i = vids.index(current_id)
            if i + 1 < len(vids):
                nxt = vids[i + 1]
                if isinstance(nxt, str) and nxt.strip():
                    return nxt.strip()
        except ValueError:
            pass
    detailed = _as_list_json(ex.get("watched_detailed_json"))
    if current_id and isinstance(detailed, list) and detailed:
        for j, record in enumerate(detailed):
            if isinstance(record, dict) and (record.get("id") or "").strip() == current_id:
                if j + 1 < len(detailed):
                    nxt = (detailed[j + 1].get("id") or "").strip()
                    if nxt:
                        return nxt
                break
    return ""


def get_gold_next_id(ex: dict, sol_key: Optional[str]) -> str:
    """Resolve the gold next-video id for a session step.

    :param ex: Session row containing selection metadata.
    :param sol_key: Alternate column containing the gold identifier.
    :returns: Canonical gold video id or an empty string when unresolved.
    """

    current = (ex.get("current_video_id") or "").strip()
    if sol_key and sol_key not in {"current_video_id", "current_id"}:
        value = ex.get(sol_key)
        if isinstance(value, str) and value.strip() and value.strip() != current:
            return value.strip()
    candidate_fields = ("next_video_id", "clicked_id", "label", "answer")
    for field in candidate_fields:
        value = ex.get(field)
        if isinstance(value, str) and value.strip() and value.strip() != current:
            return value.strip()
    return derive_next_from_history(ex, current)


def gold_index_from_items(gold: str, items: List[dict]) -> int:
    """Locate the 1-based index of ``gold`` inside the slate items list.

    :param gold: Canonical gold id to search for.
    :param items: Slate entries emitted by :func:`load_slate_items`.
    :returns: 1-based index of the gold item, or ``-1`` when not found.
    """

    gold = (gold or "").strip()
    if not gold or not items:
        return -1
    for idx, item in enumerate(items, 1):
        if gold == (item.get("id") or ""):
            return idx
    canon = _canon(gold)
    if not canon:
        return -1
    for idx, item in enumerate(items, 1):
        if canon == _canon(item.get("title", "")):
            return idx
    return -1


def build_codeocean_rows(data_root: Path) -> pd.DataFrame:  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """Construct the full interaction dataframe from raw CodeOcean assets.

    :param data_root: Capsule ``data`` directory.
    :returns: Dataframe containing per-decision rows suitable for prompt building.
    """

    sessions = _load_sessions_from_capsule(data_root)
    capsule_root = data_root.parent
    surveys = _build_survey_index_map(capsule_root)
    tree_meta, tree_issue_map = load_recommendation_tree_metadata(data_root)
    fallback_titles = load_video_metadata(data_root)
    allowlist = AllowlistState.from_mapping(load_participant_allowlists(capsule_root))

    rows: List[Dict[str, Any]] = []
    interaction_stats: Counter[str] = Counter()
    seen_participant_issue: set[Tuple[str, str]] = set()
    fallback_participant_counter = 0

    for sess in sessions:
        interaction_stats["sessions_total"] += 1
        raw_vids = [
            str(v).strip()
            for v in (sess.get("vids") or [])
            if isinstance(v, str) and str(v).strip()
        ]
        if len(raw_vids) < 2:
            interaction_stats["sessions_too_short"] += 1
            continue

        base_vids = [_strip_session_video_id(v) for v in raw_vids]
        if not all(base_vids):
            interaction_stats["sessions_invalid_ids"] += 1
            continue

        interaction_stats["pairs_total"] += max(0, len(base_vids) - 1)
        timings = _build_session_timings(sess, raw_vids, base_vids)
        watched_details = _build_watched_details(
            raw_vids,
            base_vids,
            tree_meta,
            fallback_titles,
            tree_issue_map,
            timings,
        )
        info = _session_info(sess, watched_details)

        canonical_issue: Optional[str] = None
        survey_rows: List[Dict[str, Any]] = []
        for issue_key, lookup in surveys.items():
            rows_for_url = lookup.get(info.urlid, [])
            if rows_for_url:
                canonical_issue = issue_key
                survey_rows = rows_for_url
                break

        issue_name = canonical_issue or infer_issue_from_topic(info.topic)
        if not issue_name and watched_details:
            for detail in watched_details:
                candidate_issue = detail.get("issue")
                if isinstance(candidate_issue, str) and candidate_issue.strip():
                    issue_name = candidate_issue.strip().lower()
                    break
        if not issue_name:
            raw_issue = str(sess.get("issue") or "").strip().lower()
            if raw_issue in {"gun_control", "minimum_wage", "min_wage"}:
                issue_name = "minimum_wage" if raw_issue == "min_wage" else raw_issue

        canonical_issue = issue_name or info.topic.lower()
        if not survey_rows:
            survey_lookup = surveys.get(canonical_issue, {})
            survey_rows = survey_lookup.get(info.urlid, [])
        candidate_entries = _candidate_entries_for_survey(
            canonical_issue,
            info.urlid,
            survey_rows,
            allowlist,
        )

        enforce_allowlist = allowlist.requires_enforcement(canonical_issue)

        display_orders = normalize_display_orders(sess.get("displayOrders"))
        watched_vids_json = list(base_vids)
        watched_detailed_json = deepcopy(watched_details)

        for idx in range(len(base_vids) - 1):
            current_base = base_vids[idx]
            next_base = base_vids[idx + 1]
            if not next_base:
                continue

            row: Dict[str, Any] = {
                "session_id": info.session_id,
                "step_index": idx,
                "display_step": idx + 1,
                "current_video_id": current_base,
                "current_video_raw_id": raw_vids[idx],
                "current_video_title": watched_details[idx]["title"],
                "current_video_channel": watched_details[idx]["channel_title"],
                "current_video_channel_id": watched_details[idx].get("channel_id"),
                "start_time_ms": lookup_session_value(
                    timings.start,
                    raw_vids[idx],
                    current_base,
                ),
                "end_time_ms": lookup_session_value(
                    timings.end,
                    raw_vids[idx],
                    current_base,
                ),
                "percent_visible": sess.get("percentVisible"),
                "session_finished": sess.get("sessionFinished"),
                "trajectory_json": info.trajectory_json,
            }

            selected_survey_row: Dict[str, Any] = {}
            worker_id_value = ""
            case_id_value = ""
            participant_study_label = "unknown"

            if candidate_entries:
                (
                    _,
                    _,
                    worker_id_value,
                    case_id_value,
                    participant_study_label,
                    selected_survey_row,
                ) = candidate_entries[0]
            elif survey_rows:
                selected_survey_row = select_survey_row(survey_rows, info.topic or "")
                worker_id_value = _normalize_identifier(
                    selected_survey_row.get("worker_id")
                    or selected_survey_row.get("workerid")
                    or selected_survey_row.get("WorkerID")
                )
                case_id_value = _normalize_identifier(
                    selected_survey_row.get("caseid")
                    or selected_survey_row.get("CaseID")
                )
                participant_study_label = infer_participant_study(
                    info.topic,
                    selected_survey_row or {},
                    info.topic,
                    sess,
                )

            if enforce_allowlist and not candidate_entries:
                interaction_stats["sessions_filtered_allowlist"] += 1
                continue

            recommendations = (
                watched_details[idx].get("recommendations")
                if idx < len(watched_details)
                else []
            )
            slate_items, slate_source = build_slate_items(
                idx,
                display_orders,
                recommendations,
                tree_meta,
                fallback_titles,
            )
            if not slate_items:
                interaction_stats["pairs_missing_slates"] += 1
                continue

            if (
                slate_source == "tree_metadata"
                and next_base
                and not any(item.get("id") == next_base for item in slate_items)
            ):
                fallback_title = ""
                if idx + 1 < len(watched_details):
                    fallback_title = watched_details[idx + 1].get("title") or ""
                if not fallback_title:
                    fallback_title = next_base
                slate_items = slate_items + [{"id": next_base, "title": fallback_title}]
                row["slate_items_json"] = slate_items
                row["n_options"] = len(slate_items)
            else:
                row["slate_items_json"] = slate_items
                row["n_options"] = len(slate_items)

            participant_identifier, fallback_participant_counter = participant_key(
                ParticipantIdentifiers(
                    worker_id=worker_id_value,
                    case_id=case_id_value,
                    anon_id=info.anon_id,
                    urlid=info.urlid,
                    session_id=info.session_id,
                ),
                fallback_participant_counter,
            )
            normalized_study = (participant_study_label or "").strip().lower()
            if normalized_study not in {"study1", "study2", "study3"}:
                interaction_stats["sessions_filtered_non_core_study"] += 1
                continue
            participant_issue_key = (participant_identifier, canonical_issue)
            if participant_issue_key in seen_participant_issue:
                interaction_stats["sessions_duplicate_participant_issue"] += 1
            else:
                seen_participant_issue.add(participant_issue_key)

            row["participant_id"] = participant_identifier
            row["participant_study"] = participant_study_label or "unknown"
            row["issue"] = canonical_issue
            row["urlid"] = info.urlid
            row["topic_id"] = info.topic
            row["selected_survey_row"] = selected_survey_row
            row["slate_items_json"] = slate_items
            row["slate_source"] = slate_source
            row["next_video_id"] = next_base
            row["next_video_raw_id"] = raw_vids[idx + 1]
            row["next_video_title"] = (
                watched_details[idx + 1]["title"] if idx + 1 < len(watched_details) else ""
            )
            row["watched_vids_json"] = watched_vids_json
            row["watched_detailed_json"] = watched_detailed_json

            for col in DEMOGRAPHIC_COLUMNS:
                if col in selected_survey_row:
                    row[col] = selected_survey_row.get(col)

            rows.append(row)

    df = pd.DataFrame(rows)
    log.info(
        "Interaction stats: %s",
        {k: int(v) for k, v in interaction_stats.items()},
    )
    return df

def split_dataframe(df: pd.DataFrame, validation_ratio: float = 0.1) -> Dict[str, pd.DataFrame]:
    """Split dataframe into train/validation partitions grouped by participant.

    :param df: Interaction dataframe to split.
    :param validation_ratio: Fraction of participant groups assigned to validation.
    :returns: Mapping containing ``train`` and optional ``validation`` dataframes.
    """

    if df.empty:
        return {"train": df}

    def _pick_group(row_idx: int) -> str:
        """Return a partition key for the participant/session of ``row_idx``.

        :param row_idx: Dataframe row index.
        :returns: Group key ensuring participants remain in a single split.
        """

        urlid = str(df.iloc[row_idx].get("urlid") or "").strip()
        session = str(df.iloc[row_idx].get("session_id") or "").strip()
        if urlid and urlid.lower() != "nan":
            return f"urlid::{urlid}"
        if session and session.lower() != "nan":
            return f"session::{session}"
        return f"row::{row_idx}"

    group_keys = [_pick_group(i) for i in range(len(df))]
    unique_groups = list(dict.fromkeys(group_keys))
    rng = random.Random(2024)
    rng.shuffle(unique_groups)

    val_group_count = (
        max(1, int(len(unique_groups) * validation_ratio))
        if len(unique_groups) > 1
        else 0
    )
    val_groups = set(unique_groups[:val_group_count]) if val_group_count else set()
    is_val = pd.Series(group_keys).isin(val_groups)

    splits: Dict[str, pd.DataFrame] = {
        "train": df.loc[~is_val].reset_index(drop=True),
    }
    if val_groups:
        splits["validation"] = df.loc[is_val].reset_index(drop=True)
    return splits


__all__ = [
    "AllowlistState",
    "ParticipantIdentifiers",
    "SessionInfo",
    "SessionTiming",
    "participant_key",
    "coerce_session_value",
    "normalize_session_mapping",
    "lookup_session_value",
    "load_slate_items",
    "derive_next_from_history",
    "get_gold_next_id",
    "gold_index_from_items",
    "normalize_display_orders",
    "build_slate_items",
    "build_codeocean_rows",
    "split_dataframe",
]

# Backwards-compatible aliases
_participant_key = participant_key
_coerce_session_value = coerce_session_value
_normalize_session_mapping = normalize_session_mapping
_lookup_session_value = lookup_session_value
_load_slate_items = load_slate_items
_derive_next_from_history = derive_next_from_history
_get_gold_next_id = get_gold_next_id
_gold_index_from_items = gold_index_from_items
_normalize_display_orders = normalize_display_orders
_build_slate_items = build_slate_items
_build_codeocean_rows = build_codeocean_rows
_split_dataframe = split_dataframe
