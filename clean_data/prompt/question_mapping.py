"""Reference mapping between Qualtrics question IDs and friendly feature names.

This table helps us keep track of where each prompt feature originates in the
CodeOcean survey exports.  The ``question_ids`` list contains the raw column
names observed across the intermediate CSVs for Studies 1–4.  When the builder
renames or aggregates features, refer back here to understand which Qualtrics
items are involved.
"""

from __future__ import annotations

from typing import Dict, List, TypedDict


class QuestionMapping(TypedDict, total=False):
    """Structure describing how a friendly feature relates to raw survey IDs."""

    question_ids: List[str]
    description: str
    notes: str


QUESTION_ID_MAPPING: Dict[str, QuestionMapping] = {
    "biden_approval": {
        "question_ids": ["q5_5", "Q5_b", "Q5_b_W2", "political_lead_feels_5"],
        "description": (
            "Feeling thermometer / approval rating for President Joe Biden "
            "(0-100)."
        ),
        "notes": (
            "Appears in MTurk waves (q5_5), YouGov panels (Q5_b / Q5_b_W2), "
            "and Shorts follow-up surveys (political_lead_feels_5)."
        ),
    },
    "trump_approval": {
        "question_ids": ["q5_2", "Q5_a", "Q5_a_W2", "political_lead_feels_2"],
        "description": (
            "Feeling thermometer / approval rating for former President "
            "Donald Trump (0-100)."
        ),
        "notes": "Parallel structure to Biden approval across all studies.",
    },
    "freq_youtube": {
        "question_ids": ["freq_youtube", "q77", "Q77", "youtube_freq", "youtube_freq_v2"],
        "description": "Self-reported YouTube viewing frequency.",
        "notes": (
            "Shorts study also records `youtube_time`; see `binge_youtube` "
            "for that variant."
        ),
    },
    "binge_youtube": {
        "question_ids": ["binge_youtube", "youtube_time"],
        "description": "Indicator for binge-watching YouTube or total minutes watched.",
        "notes": (
            "Only the Shorts survey surfaces `youtube_time`; other studies omit "
            "an explicit binge field."
        ),
    },
    "favorite_channels": {
        "question_ids": ["q8", "fav_channels"],
        "description": "Open-ended list of favourite YouTube channels.",
        "notes": "Qualtrics export stores the raw text under q8.",
    },
    "popular_channels_followed": {
        "question_ids": ["q78", "popular_channels"],
        "description": "Multiple-choice indicator of popular channels watched recently.",
    },
    "children_in_household": {
        "question_ids": ["children_in_house", "kids_household", "child18"],
        "description": "Presence of children in the household.",
        "notes": "`child18` is provided by the YouGov panel export.",
    },
    "city": {
        "question_ids": [],
        "description": "Participant city/city-name.",
        "notes": (
            "City is not present in the released intermediate CSVs; only "
            "latitude/longitude are available."
        ),
    },
    "civic_engagement": {
        "question_ids": ["civic_participation", "volunteering", "civic_activity"],
        "description": "Civic participation / volunteering frequency.",
        "notes": (
            "The published CSVs do not expose civic-engagement questions; "
            "included here for completeness."
        ),
    },
}


__all__ = ["QUESTION_ID_MAPPING"]
