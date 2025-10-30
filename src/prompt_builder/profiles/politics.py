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

"""Political sentence builders for viewer profiles."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..profile_helpers import collect_labeled_fields, sentencize

POLITICS_FIELD_SPECS: Sequence[tuple[Sequence[str], str]] = (
    (("pid1", "party_id", "party_registration", "partyid"), "Party identification"),
    (("pid2", "party_id_lean", "party_lean"), "Party lean"),
    (("ideo1", "ideo2", "ideology", "ideology_text"), "Ideology"),
    (("pol_interest", "interest_politics", "political_interest"), "Political interest"),
    (("vote_2016", "presvote16post"), "Voted in 2016"),
    (("vote_2020", "presvote20post"), "Voted in 2020"),
    (("vote_2024", "vote_intent_2024", "vote_2024_intention"), "Vote intention 2024"),
    (
        (
            "trump_approve",
            "trump_job_approval",
            "q5_2",
            "Q5_a",
            "Q5_a_W2",
            "political_lead_feels_2",
        ),
        "Trump approval",
    ),
    (
        (
            "biden_approve",
            "biden_job_approval",
            "q5_5",
            "Q5_b",
            "Q5_b_W2",
            "political_lead_feels_5",
        ),
        "Biden approval",
    ),
    (("civic_participation", "volunteering", "civic_activity"), "Civic engagement"),
)


def _politics_sentences(ex: Dict[str, Any], selected: Dict[str, Any]) -> List[str]:
    """Return sentences covering political views and affiliations."""

    politics = collect_labeled_fields(ex, selected, POLITICS_FIELD_SPECS)
    sentence = sentencize("Politics include", politics)
    return [sentence] if sentence else []
