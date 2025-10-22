"""Dataset helpers reused by the XGBoost baseline."""

from __future__ import annotations

from knn.data import (  # Re-export shared dataset utilities
    DEFAULT_DATASET_SOURCE,
    EVAL_SPLIT,
    PROMPT_COLUMN,
    PROMPT_MAX_HISTORY,
    SOLUTION_COLUMN,
    TRAIN_SPLIT,
    filter_dataset_for_issue,
    filter_dataset_for_participant_studies,
    filter_split_for_participant_studies,
    issues_in_dataset,
    load_dataset_source,
)

__all__ = [
    "DEFAULT_DATASET_SOURCE",
    "EVAL_SPLIT",
    "PROMPT_COLUMN",
    "PROMPT_MAX_HISTORY",
    "SOLUTION_COLUMN",
    "TRAIN_SPLIT",
    "filter_dataset_for_issue",
    "filter_dataset_for_participant_studies",
    "filter_split_for_participant_studies",
    "issues_in_dataset",
    "load_dataset_source",
]
