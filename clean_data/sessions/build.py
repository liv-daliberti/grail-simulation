"""High-level builders that assemble interaction dataframes from session logs."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from clean_data.helpers import (
    _normalize_identifier,
    _strip_session_video_id,
)
from clean_data.io import (
    load_recommendation_tree_metadata,
    load_video_metadata,
)
from clean_data.surveys import (
    DEMOGRAPHIC_COLUMNS,
    infer_issue_from_topic,
    infer_participant_study,
    load_participant_allowlists,
    select_survey_row,
)

from .io_utils import build_survey_index_map, load_sessions_from_capsule
from .models import AllowlistState, ParticipantIdentifiers, SessionInfo, SessionTiming
from .participants import candidate_entries_for_survey, participant_key
from .slates import build_slate_items, normalize_display_orders
from .watch import (
    VideoMetadataSources,
    build_session_timings,
    build_watched_details,
    lookup_session_value,
)

logger = logging.getLogger("clean_grail")


@dataclass(frozen=True)
class _SessionResources:
    """Reusable datasets needed to build session interaction rows."""

    surveys: Dict[str, Dict[str, List[Dict[str, Any]]]]
    tree_meta: Dict[str, Dict[str, Any]]
    tree_issue_map: Dict[str, str]
    fallback_titles: Dict[str, Any]
    allowlist: AllowlistState


@dataclass
class _SessionBuildState:
    """Mutable state tracking rows, counters, and participant ids."""

    rows: List[Dict[str, Any]] = field(default_factory=list)
    interaction_stats: Counter[str] = field(default_factory=Counter)
    seen_participant_issue: set[Tuple[str, str]] = field(default_factory=set)
    fallback_participant_counter: int = 0


@dataclass(frozen=True)
class _WatchContext:
    """Per-session watch sequence artefacts shared across builders."""

    raw_vids: List[str]
    base_vids: List[str]
    details: List[Dict[str, Any]]
    timings: SessionTiming
    display_orders: Dict[int, List[str]]


@dataclass(frozen=True)
class _SessionContext:
    """Derived per-session inputs consumed when creating interaction rows."""

    session_payload: Mapping[str, Any]
    info: SessionInfo
    watch: _WatchContext
    canonical_issue: str
    survey_rows: List[Dict[str, Any]]
    candidate_entries: List[Tuple[int, str, str, str, str, Dict[str, Any]]]
    enforce_allowlist: bool

    @property
    def raw_vids(self) -> List[str]:
        """Return the raw session video identifiers.

        :return: Raw video ids in their original order.
        :rtype: List[str]
        """
        return self.watch.raw_vids

    @property
    def base_vids(self) -> List[str]:
        """Return the base video identifiers without session prefixes.

        :return: Base video ids in watch order.
        :rtype: List[str]
        """
        return self.watch.base_vids

    @property
    def watched_details(self) -> List[Dict[str, Any]]:
        """Return the detailed metadata captured for each watched video.

        :return: Per-video metadata dictionaries.
        :rtype: List[Dict[str, Any]]
        """
        return self.watch.details

    @property
    def timings(self) -> SessionTiming:
        """Return the timing metadata associated with the session.

        :return: Structured timing information.
        :rtype: SessionTiming
        """
        return self.watch.timings

    @property
    def display_orders(self) -> Dict[int, List[str]]:
        """Return the mapping of sequence step to display order ids.

        :return: Step-indexed lists of display ids.
        :rtype: Dict[int, List[str]]
        """
        return self.watch.display_orders

    @property
    def watched_vids_json(self) -> List[str]:
        """Return the list of watched base ids suitable for JSON serialization.

        :return: List of base ids for JSON output.
        :rtype: List[str]
        """
        return list(self.watch.base_vids)

    @property
    def watched_detailed_json(self) -> List[Dict[str, Any]]:
        """Return the watched video metadata deep-copied for JSON serialization.

        :return: JSON-ready metadata entries.
        :rtype: List[Dict[str, Any]]
        """
        return deepcopy(self.watch.details)


@dataclass(frozen=True)
class _ParticipantSessionMeta:
    """Participant identifiers resolved for a session."""

    selected_row: Dict[str, Any]
    worker_id: str
    case_id: str
    study_label: str


def _session_info(sess: dict, watched_details: List[Dict[str, Any]]) -> SessionInfo:
    """Build reusable identifiers and payloads for a session."""

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


def _resolve_issue_and_surveys(
    sess: Mapping[str, Any],
    info: SessionInfo,
    watched_details: List[Dict[str, Any]],
    surveys: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Derive the canonical issue and survey rows associated with the session."""

    canonical_issue: Optional[str] = None
    survey_rows: List[Dict[str, Any]] = []
    for issue_key, lookup in surveys.items():
        rows_for_url = lookup.get(info.urlid, [])
        if rows_for_url:
            canonical_issue = issue_key
            survey_rows = list(rows_for_url)
            break

    issue_name = canonical_issue or infer_issue_from_topic(info.topic)
    if not issue_name:
        for detail in watched_details:
            candidate_issue = detail.get("issue")
            if isinstance(candidate_issue, str) and candidate_issue.strip():
                issue_name = candidate_issue.strip().lower()
                break
    if not issue_name:
        raw_issue = str(sess.get("issue") or "").strip().lower()
        if raw_issue in {"gun_control", "minimum_wage", "min_wage"}:
            issue_name = "minimum_wage" if raw_issue == "min_wage" else raw_issue

    canonical_issue_value = ((issue_name or info.topic) or "").strip().lower()
    if not survey_rows:
        survey_rows = list(surveys.get(canonical_issue_value, {}).get(info.urlid, []))
    return canonical_issue_value, survey_rows


def _build_session_context(
    sess: Dict[str, Any],
    resources: _SessionResources,
    state: _SessionBuildState,
) -> Optional[_SessionContext]:
    """Normalize the session payload into reusable components."""

    state.interaction_stats["sessions_total"] += 1
    raw_vids = [
        str(v).strip()
        for v in (sess.get("vids") or [])
        if isinstance(v, str) and str(v).strip()
    ]
    if len(raw_vids) < 2:
        state.interaction_stats["sessions_too_short"] += 1
        return None

    base_vids = [_strip_session_video_id(v) for v in raw_vids]
    if not all(base_vids):
        state.interaction_stats["sessions_invalid_ids"] += 1
        return None

    state.interaction_stats["pairs_total"] += max(0, len(base_vids) - 1)
    timings = build_session_timings(sess, raw_vids, base_vids)
    metadata_sources = VideoMetadataSources(
        tree_meta=resources.tree_meta,
        fallback_titles=resources.fallback_titles,
        tree_issue_map=resources.tree_issue_map,
    )
    watched_details = build_watched_details(
        raw_vids,
        base_vids,
        metadata_sources,
        timings=timings,
    )
    info = _session_info(sess, watched_details)
    canonical_issue, survey_rows = _resolve_issue_and_surveys(
        sess,
        info,
        watched_details,
        resources.surveys,
    )
    candidate_entries = candidate_entries_for_survey(
        canonical_issue,
        info.urlid,
        survey_rows,
        resources.allowlist,
    )
    enforce_allowlist = resources.allowlist.requires_enforcement(canonical_issue)
    display_orders = normalize_display_orders(sess.get("displayOrders"))

    watch_context = _WatchContext(
        raw_vids=raw_vids,
        base_vids=base_vids,
        details=watched_details,
        timings=timings,
        display_orders=display_orders,
    )

    return _SessionContext(
        session_payload=sess,
        info=info,
        watch=watch_context,
        canonical_issue=canonical_issue,
        survey_rows=survey_rows,
        candidate_entries=candidate_entries,
        enforce_allowlist=enforce_allowlist,
    )


def _participant_metadata(context: _SessionContext) -> _ParticipantSessionMeta:
    """Return participant identifiers and study metadata for the session."""

    if context.candidate_entries:
        _, _, worker_value, case_value, study_label, survey_row = context.candidate_entries[0]
        return _ParticipantSessionMeta(
            selected_row=dict(survey_row),
            worker_id=worker_value,
            case_id=case_value,
            study_label=study_label,
        )

    if context.survey_rows:
        selected = select_survey_row(context.survey_rows, context.info.topic or "") or {}
        worker_value = _normalize_identifier(
            selected.get("worker_id")
            or selected.get("workerid")
            or selected.get("WorkerID")
        )
        case_value = _normalize_identifier(
            selected.get("caseid") or selected.get("CaseID")
        )
        study_label = infer_participant_study(
            context.info.topic,
            selected,
            context.info.topic,
            context.session_payload,
        )
        return _ParticipantSessionMeta(
            selected_row=dict(selected),
            worker_id=worker_value,
            case_id=case_value,
            study_label=study_label,
        )

    return _ParticipantSessionMeta(
        selected_row={},
        worker_id="",
        case_id="",
        study_label="unknown",
    )


def _fallback_title_for_next(
    context: _SessionContext,
    idx: int,
    next_base: str,
) -> str:
    """Return a fallback title for the next video when slate metadata omits it."""

    if idx + 1 < len(context.watched_details):
        candidate = context.watched_details[idx + 1].get("title") or ""
        if candidate:
            return candidate
    return next_base


def _build_step_row(
    idx: int,
    context: _SessionContext,
    participant_meta: _ParticipantSessionMeta,
    state: _SessionBuildState,
    resources: _SessionResources,
) -> Optional[Dict[str, Any]]:
    """Construct a single interaction row for the given step."""

    next_base = context.base_vids[idx + 1]
    if not next_base:
        return None

    current_base = context.base_vids[idx]
    detail = context.watched_details[idx]
    row: Dict[str, Any] = {
        "session_id": context.info.session_id,
        "step_index": idx,
        "display_step": idx + 1,
        "current_video_id": current_base,
        "current_video_raw_id": context.raw_vids[idx],
        "current_video_title": detail["title"],
        "current_video_channel": detail["channel_title"],
        "current_video_channel_id": detail.get("channel_id"),
        "start_time_ms": lookup_session_value(
            context.timings.start,
            context.raw_vids[idx],
            current_base,
        ),
        "end_time_ms": lookup_session_value(
            context.timings.end,
            context.raw_vids[idx],
            current_base,
        ),
        "percent_visible": context.session_payload.get("percentVisible"),
        "session_finished": context.session_payload.get("sessionFinished"),
        "trajectory_json": context.info.trajectory_json,
    }

    slate_items, slate_source = build_slate_items(
        idx,
        context.display_orders,
        detail.get("recommendations") if isinstance(detail.get("recommendations"), list) else [],
        resources.tree_meta,
        resources.fallback_titles,
    )
    if not slate_items:
        state.interaction_stats["pairs_missing_slates"] += 1
        return None

    if (
        slate_source == "tree_metadata"
        and next_base
        and not any(item.get("id") == next_base for item in slate_items)
    ):
        slate_items = slate_items + [
            {"id": next_base, "title": _fallback_title_for_next(context, idx, next_base)}
        ]

    row["slate_items_json"] = slate_items
    row["slate_source"] = slate_source
    row["n_options"] = len(slate_items)
    row["next_video_id"] = next_base
    row["next_video_raw_id"] = context.raw_vids[idx + 1]
    row["next_video_title"] = (
        context.watched_details[idx + 1]["title"]
        if idx + 1 < len(context.watched_details)
        else ""
    )
    row["watched_vids_json"] = context.watched_vids_json
    row["watched_detailed_json"] = context.watched_detailed_json

    participant_identifier, state.fallback_participant_counter = participant_key(
        ParticipantIdentifiers(
            worker_id=participant_meta.worker_id,
            case_id=participant_meta.case_id,
            anon_id=context.info.anon_id,
            urlid=context.info.urlid,
            session_id=context.info.session_id,
        ),
        state.fallback_participant_counter,
    )

    if (participant_meta.study_label or "").strip().lower() not in {"study1", "study2", "study3"}:
        state.interaction_stats["sessions_filtered_non_core_study"] += 1
        return None

    participant_issue_key = (participant_identifier, context.canonical_issue)
    if participant_issue_key in state.seen_participant_issue:
        state.interaction_stats["sessions_duplicate_participant_issue"] += 1
    else:
        state.seen_participant_issue.add(participant_issue_key)

    row["participant_id"] = participant_identifier
    row["participant_study"] = participant_meta.study_label or "unknown"
    row["issue"] = context.canonical_issue
    row["urlid"] = context.info.urlid
    row["topic_id"] = context.info.topic
    row["selected_survey_row"] = participant_meta.selected_row
    for col in DEMOGRAPHIC_COLUMNS:
        if col in participant_meta.selected_row:
            row[col] = participant_meta.selected_row.get(col)
    return row


def _collect_session_rows(
    sess: Dict[str, Any],
    resources: _SessionResources,
    state: _SessionBuildState,
) -> None:
    """Append interaction rows for the session into ``state.rows``."""

    context = _build_session_context(sess, resources, state)
    if context is None:
        return

    if context.enforce_allowlist and not context.candidate_entries:
        state.interaction_stats["sessions_filtered_allowlist"] += 1
        return

    participant_meta = _participant_metadata(context)
    for idx in range(len(context.base_vids) - 1):
        row = _build_step_row(
            idx,
            context,
            participant_meta,
            state,
            resources,
        )
        if row is not None:
            state.rows.append(row)


def build_codeocean_rows(data_root: Path) -> pd.DataFrame:
    """Construct the full interaction dataframe from raw CodeOcean assets."""

    data_root = Path(data_root)
    sessions = load_sessions_from_capsule(data_root)
    capsule_root = data_root.parent
    surveys = build_survey_index_map(capsule_root)
    tree_meta, tree_issue_map = load_recommendation_tree_metadata(data_root)
    fallback_titles = load_video_metadata(data_root)
    allowlist = AllowlistState.from_mapping(load_participant_allowlists(capsule_root))

    resources = _SessionResources(
        surveys=surveys,
        tree_meta=tree_meta,
        tree_issue_map=tree_issue_map,
        fallback_titles=fallback_titles,
        allowlist=allowlist,
    )
    state = _SessionBuildState()

    for sess in sessions:
        _collect_session_rows(sess, resources, state)

    interaction_frame = pd.DataFrame(state.rows)
    logger.info(
        "Interaction stats: %s",
        {k: int(v) for k, v in state.interaction_stats.items()},
    )
    return interaction_frame


def split_dataframe(
    interaction_frame: pd.DataFrame, validation_ratio: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """Split dataframe into train/validation partitions grouped by participant."""

    if interaction_frame.empty:
        return {"train": interaction_frame}

    def _pick_group(row_idx: int) -> str:
        """Return a partition key for the participant/session of ``row_idx``."""

        urlid = str(interaction_frame.iloc[row_idx].get("urlid") or "").strip()
        session = str(interaction_frame.iloc[row_idx].get("session_id") or "").strip()
        if urlid and urlid.lower() != "nan":
            return f"urlid::{urlid}"
        if session and session.lower() != "nan":
            return f"session::{session}"
        return f"row::{row_idx}"

    group_keys = [_pick_group(i) for i in range(len(interaction_frame))]
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
        "train": interaction_frame.loc[~is_val].reset_index(drop=True),
    }
    if val_groups:
        splits["validation"] = interaction_frame.loc[is_val].reset_index(drop=True)
    return splits
