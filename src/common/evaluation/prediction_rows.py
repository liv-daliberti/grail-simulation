#!/usr/bin/env python
"""Shared helpers for parsing evaluation prediction rows.

Centralises logic used by next-video and pipeline loaders to coerce fields
from prediction JSONL rows and to build ``slate_eval.Observation`` objects.
"""

from __future__ import annotations

from typing import Mapping, Tuple

from . import slate_eval


def _safe_int(value: object, default: int) -> int:
    """Return ``int(value)`` or ``default`` on failure."""

    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def parse_common_row_fields(
    row: Mapping[str, object]
) -> Tuple[str, str, int, int, int, int | None, bool, bool]:
    """Extract common fields from a prediction row.

    :param row: Mapping representing a single prediction JSONL row.
    :returns: Tuple of ``(issue, study, n_options, position_index, gold_index,
        parsed_index, eligible, is_correct)`` with sane defaults on errors.
    :rtype: tuple[str, str, int, int, int, int | None, bool, bool]
    """

    issue = str(row.get("issue") or "unspecified")
    study = str(row.get("participant_study") or "unspecified")
    n_options = _safe_int(row.get("n_options"), 0)
    position_index = _safe_int(row.get("position_index"), -1)
    gold_index = _safe_int(row.get("gold_index"), 0)
    parsed_raw = row.get("parsed_index")
    try:
        parsed_index = int(parsed_raw) if parsed_raw is not None else None
    except (TypeError, ValueError):
        parsed_index = None
    eligible = bool(row.get("eligible"))
    is_correct = bool(row.get("correct"))
    return (
        issue,
        study,
        n_options,
        position_index,
        gold_index,
        parsed_index,
        eligible,
        is_correct,
    )


def observation_from_row(
    row: Mapping[str, object], *, is_formatted: bool
) -> slate_eval.Observation:
    """Build a ``slate_eval.Observation`` from a prediction row.

    The ``is_formatted`` flag is computed by the caller to avoid coupling this
    helper to any specific tag/regex implementation.
    """

    (
        issue,
        study,
        n_options,
        position_index,
        gold_index,
        parsed_index,
        eligible,
        is_correct,
    ) = parse_common_row_fields(row)

    return slate_eval.Observation(
        issue_label=issue,
        study_label=study,
        position_bucket=slate_eval.bucket_from_position(position_index),
        option_bucket=slate_eval.bucket_from_options(n_options),
        option_count=n_options,
        gold_index=gold_index,
        parsed_index=parsed_index,
        is_formatted=is_formatted,
        eligible=eligible,
        is_correct=is_correct,
    )
