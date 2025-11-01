#!/usr/bin/env python
"""Utilities and helpers for GRPO opinion evaluation."""

from __future__ import annotations
import logging
import re
from typing import List, Mapping, MutableMapping, Sequence

from common.opinion import OpinionSpec
from common.opinion.baselines import baseline_metrics as _baseline_metrics_shared

LOGGER = logging.getLogger("grpo.opinion")

ANSWER_PATTERN = re.compile(r"(?si)<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>")
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _clip_prediction(value: float) -> float:
    """Clamp predictions to the 1–7 opinion scale."""

    return min(7.0, max(1.0, value))


def _compute_baseline_metrics(
    *, truth_before: Sequence[float], truth_after: Sequence[float]
) -> Mapping[str, object]:
    """Return baseline opinion metrics via the shared implementation."""
    return _baseline_metrics_shared(truth_before=truth_before, truth_after=truth_after)


def _parse_prediction(raw_output: str) -> float:
    """Parse a numeric prediction from the model output."""

    match = ANSWER_PATTERN.search(raw_output)
    candidate = match.group(1) if match else None
    if not candidate:
        fallback = NUMBER_PATTERN.search(raw_output)
        candidate = fallback.group(0) if fallback else ""
    try:
        return _clip_prediction(float(candidate))
    except (TypeError, ValueError):
        LOGGER.warning("Unable to parse opinion prediction from output: %r", raw_output)
        return float("nan")


def _extract_user_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    """Return the most recent user message from ``messages``."""

    for message in reversed(messages):
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"]).strip()
    return ""


def _select_examples_by_participant(
    rows: Sequence[Mapping[str, object]], *, spec: OpinionSpec
) -> List[Mapping[str, object]]:
    """Return the latest example per participant for ``spec``."""

    per_participant: MutableMapping[str, tuple[int, Mapping[str, object]]] = {}
    for row in rows:
        issue = str(row.get("issue") or "").strip().lower()
        study = str(row.get("participant_study") or "").strip().lower()
        if issue != spec.issue.lower() or study != spec.key.lower():
            continue
        participant_id = str(row.get("participant_id") or "").strip()
        if not participant_id:
            continue
        before = row.get(spec.before_column)
        after = row.get(spec.after_column)
        if before is None or after is None:
            continue
        try:
            before_value = float(before)
            after_value = float(after)
        except (TypeError, ValueError):
            continue
        try:
            step_index = int(row.get("step_index") or -1)
        except (TypeError, ValueError):
            step_index = -1
        payload = dict(row)
        payload["_participant_id"] = participant_id
        payload["_before"] = before_value
        payload["_after"] = after_value
        payload["_step_index"] = step_index
        existing = per_participant.get(participant_id)
        if existing is None or step_index >= existing[0]:
            per_participant[participant_id] = (step_index, payload)
    selected = [payload for _, payload in per_participant.values()]
    selected.sort(key=lambda item: (item["_participant_id"], item["_step_index"]))
    return selected


__all__ = [
    "_clip_prediction",
    "_compute_baseline_metrics",
    "_extract_user_prompt",
    "_parse_prediction",
    "_select_examples_by_participant",
]
