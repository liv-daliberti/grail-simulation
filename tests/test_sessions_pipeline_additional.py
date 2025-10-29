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

"""Additional unit tests for the sessions pipeline utilities."""

from __future__ import annotations

from typing import Dict, List

import pytest

import clean_data.sessions.build as build_module
import clean_data.sessions.participants as participants_module
import clean_data.sessions.slates as slates_module
import clean_data.sessions.watch as watch_module
from clean_data.sessions.models import AllowlistState, SessionTiming


# ---------------------------------------------------------------------------
# build.py helpers
# ---------------------------------------------------------------------------


def _base_resources(allowlist: AllowlistState | None = None) -> build_module._SessionResources:
    return build_module._SessionResources(
        surveys={},
        tree_meta={},
        tree_issue_map={},
        fallback_titles={},
        allowlist=allowlist or AllowlistState.from_mapping({}),
    )


def test_build_session_context_short_session_returns_none():
    resources = _base_resources()
    state = build_module._SessionBuildState()
    session = {"vids": ["only_vid"]}
    context = build_module._build_session_context(session, resources, state)
    assert context is None
    assert state.interaction_stats["sessions_too_short"] == 1


def test_collect_session_rows_allowlist_filters():
    allowlist = AllowlistState.from_mapping({"gun_control": {"worker_ids": {"w1"}}})
    tree_meta = {
        "video00001": {"title": "Title 1", "channel_title": "Chan 1"},
        "video00002": {"title": "Title 2", "channel_title": "Chan 2"},
    }
    resources = build_module._SessionResources(
        surveys={},
        tree_meta=tree_meta,
        tree_issue_map={},
        fallback_titles={},
        allowlist=allowlist,
    )
    session = {
        "vids": ["video00001", "video00002"],
        "anonymousFirebaseAuthUID": "anon",
        "sessionID": "session-1",
        "topicId": "gun_control",
        "urlid": "url-1",
    }
    state = build_module._SessionBuildState()
    build_module._collect_session_rows(session, resources, state)
    assert state.interaction_stats["sessions_filtered_allowlist"] == 1
    assert not state.rows


def _tree_resources_with_surveys() -> tuple[build_module._SessionResources, Dict[str, Dict[str, List[Dict[str, str]]]]]:
    allowlist = AllowlistState.from_mapping({"gun_control": {"worker_ids": {"worker-1"}}})
    tree_meta = {
        "video_base": {
            "title": "Base title",
            "channel_title": "Base channel",
            "recs": [{"id": "other_video"}],
        },
        "video_next": {"channel_title": "Next channel"},
    }
    fallback_titles = {"video_next": "Generated Title"}
    surveys = {
        "gun_control": {
            "url-allow": [
                {
                    "worker_id": "worker-1",
                    "treatment_arm": "treatment",
                    "pro": "yes",
                    "anti": "no",
                }
            ]
        }
    }
    resources = build_module._SessionResources(
        surveys=surveys,
        tree_meta=tree_meta,
        tree_issue_map={},
        fallback_titles=fallback_titles,
        allowlist=allowlist,
    )
    return resources, surveys


def _session_payload():
    return {
        "vids": ["video_base", "video_next"],
        "anonymousFirebaseAuthUID": "anon",
        "sessionID": "sess-1",
        "topicId": "gun_control",
        "urlid": "url-allow",
        "displayOrders": {},
    }


def test_collect_session_rows_records_duplicates_and_fallback_title():
    resources, _ = _tree_resources_with_surveys()
    session = _session_payload()
    state = build_module._SessionBuildState()

    build_module._collect_session_rows(session, resources, state)
    assert len(state.rows) == 1
    row = state.rows[0]
    # Because the recommendation lacked the next video, the fallback should add it with generated title.
    titles = [item["title"] for item in row["slate_items_json"]]
    assert "Generated Title" in titles

    # Reprocessing the same session increments duplicate participant/issue counter.
    build_module._collect_session_rows(session, resources, state)
    assert state.interaction_stats["sessions_duplicate_participant_issue"] == 1


def test_fallback_title_for_next_uses_next_detail_title():
    resources, surveys = _tree_resources_with_surveys()
    session = _session_payload()
    state = build_module._SessionBuildState()
    context = build_module._build_session_context(session, resources, state)
    assert context is not None
    fallback = build_module._fallback_title_for_next(context, 0, "video_next")
    assert fallback == "Generated Title"


# ---------------------------------------------------------------------------
# participants.py helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "issue,row,expected",
    [
        (
            "gun_control",
            {"worker_id": "worker-1", "treatment_arm": "treatment", "pro": "yes", "anti": "no"},
            ("study1", "worker-1"),
        ),
        (
            "gun_control",
            {"worker_id": "worker-1", "treatment_arm": "control", "pro": "yes", "anti": "no"},
            None,
        ),
        (
            "minimum_wage",
            {
                "worker_id": "worker-2",
                "treatment_arm": "treatment",
                "pro": None,
                "anti": "no",
            },
            None,
        ),
        (
            "minimum_wage",
            {
                "worker_id": "worker-2",
                "treatment_arm": "treatment",
                "pro": "yes",
                "anti": "no",
            },
            ("study2", "worker-2"),
        ),
        (
            "minimum_wage",
            {
                "caseid": "case-1",
                "treatment_arm": "treatment",
                "pro": "yes",
                "anti": "no",
            },
            ("study3", "case-1"),
        ),
    ],
)
def test_candidate_entry_validation(issue, row, expected):
    allowlist = AllowlistState.from_mapping(
        {
            "gun_control": {"worker_ids": {"worker-1"}},
            "minimum_wage": {
                "study2_worker_ids": {"worker-2"},
                "study3_caseids": {"case-1"},
            },
        }
    )
    entry = participants_module._candidate_entry(
        issue,
        "url",
        row,
        allowlist,
    )
    if expected is None:
        assert entry is None
    else:
        assert entry is not None
        _, participant_token, worker_candidate, case_candidate, study_label, _ = entry
        exp_study, exp_token = expected
        assert study_label == exp_study
        assert participant_token or worker_candidate or case_candidate
        assert exp_token in (participant_token, worker_candidate, case_candidate)


# ---------------------------------------------------------------------------
# slates.py helpers
# ---------------------------------------------------------------------------


def test_build_slate_items_from_display_orders_deduplicates():
    display_orders = {0: ["VIDEOCANDID1", "VIDEOCANDID1"]}
    items, source = slates_module.build_slate_items(
        0,
        display_orders,
        recommendations=None,
        tree_meta={},
        fallback_titles={},
    )
    assert source == "display_orders"
    assert len(items) == 1


def test_build_slate_items_tree_metadata_appends_fallback():
    tree_meta = {"video_base": {"title": "base", "channel_title": "channel", "recs": [{"id": "other"}]}}
    fallback_titles = {"video_next": "Next Title"}
    context_items, source = slates_module.build_slate_items(
        0,
        {},
        tree_meta["video_base"]["recs"],
        tree_meta,
        fallback_titles,
    )
    assert source == "tree_metadata"
    assert all(item["id"] != "video_next" for item in context_items)

    # When next video isn't in recommendations, fallback title should be used.
    context = build_module._SessionContext(  # type: ignore[attr-defined]
        session_payload={},
        info=build_module.SessionInfo("sess", "anon", "topic", "url", "traj"),  # type: ignore[arg-type]
        watch=build_module._WatchContext(  # type: ignore[arg-type]
            raw_vids=["video_base", "video_next"],
            base_vids=["video_base", "video_next"],
            details=[
                {"title": "first", "channel_title": "chan", "recommendations": [{"id": "other"}]},
                {"title": "Next Title"},
            ],
            timings=SessionTiming({}, {}, {}, {}, {}),
            display_orders={},
        ),
        canonical_issue="topic",
        survey_rows=[],
        candidate_entries=[],
        enforce_allowlist=False,
    )
    new_items = context_items + [
        {"id": "video_next", "title": build_module._fallback_title_for_next(context, 0, "video_next")}
    ]
    assert any(item["title"] == "Next Title" and item["id"] == "video_next" for item in new_items)


def test_dervive_next_from_history_prefers_detailed_history():
    example = {
        "watched_vids_json": ["vid1", "vid2", "vid3"],
        "watched_detailed_json": [
            {"id": "vid1"},
            {"id": "vid2"},
            {"id": "vid3"},
        ],
    }
    assert slates_module.derive_next_from_history(example, "vid1") == "vid2"
    assert slates_module.derive_next_from_history(example, "vid3") == ""


# ---------------------------------------------------------------------------
# watch.py helpers
# ---------------------------------------------------------------------------


def test_build_watched_details_includes_fallback_metadata():
    metadata_sources = watch_module.VideoMetadataSources(
        tree_meta={
            "vid1": {"title": "", "channel_title": "", "recs": [{"id": "vid2"}]},
        },
        fallback_titles={"vid1": "Fallback 1", "vid2": "Fallback 2"},
        tree_issue_map={"vid1": "gun_control"},
    )
    timings = SessionTiming(
        start={"vid1": 1},
        end={"vid1": 2},
        watch={"vid1": 3},
        total={"vid1": 4},
        delay={"vid1": 0},
    )
    details = watch_module.build_watched_details(
        raw_vids=["vid1"],
        base_vids=["vid1"],
        metadata=metadata_sources,
        timings=timings,
    )
    entry = details[0]
    assert entry["title"] == "Fallback 1"
    assert entry["channel_title"] == "(channel missing)"
    assert entry["recommendations"][0]["id"] == "vid2"
