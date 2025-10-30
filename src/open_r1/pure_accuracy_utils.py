#!/usr/bin/env python
"""Utilities supporting the pure accuracy reward computation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

_ANS_PAT = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
_INDEX_ONLY_RE = re.compile(r"^\s*(?:option\s*)?(\d+)\s*$", re.I)
_OPINION_PATTERNS = (
    re.compile(r"(?si)<opinion>\s*([^<\n]+?)\s*</opinion>"),
    re.compile(r"(?si)<opinion_change>\s*([^<\n]+?)\s*</opinion_change>"),
    re.compile(r"(?si)<opiniondirection>\s*([^<\n]+?)\s*</opiniondirection>"),
)
_OPINION_INCREASE = {"increase", "increased", "higher", "rise", "rising", "up"}
_OPINION_DECREASE = {"decrease", "decreased", "lower", "drop", "falling", "down"}
_OPINION_NO_CHANGE = {
    "same",
    "nochange",
    "nochanges",
    "nochg",
    "unchanged",
    "steady",
    "equal",
    "no_difference",
}


@dataclass
class PureAccuracyStats:
    """Aggregate counters for :func:`pure_accuracy_reward` metrics."""

    parsed: int = 0
    eligible: int = 0
    direction_available: int = 0
    direction_parsed: int = 0
    direction_correct: int = 0

    def payload(self, total: int, batch_mean: float) -> Dict[str, float]:
        """Return a logging payload summarising the collected metrics.

        :param total: Number of samples scored in the batch.
        :type total: int
        :param batch_mean: Average reward emitted across the batch.
        :type batch_mean: float
        :returns: Dictionary keyed by the wandb metric identifiers.
        :rtype: dict[str, float]
        """

        if total <= 0:
            return {}
        payload: Dict[str, float] = {
            "reward/pure_acc/parsed_rate": self.parsed / total,
            "reward/pure_acc/eligible_rate": self.eligible / total,
            "reward/pure_acc/batch_mean": batch_mean,
        }
        if self.direction_available:
            payload["reward/pure_acc/opinion_available_rate"] = (
                self.direction_available / total
            )
            payload["reward/pure_acc/opinion_parsed_rate"] = (
                self.direction_parsed / self.direction_available
            )
            payload["reward/pure_acc/opinion_accuracy"] = (
                self.direction_correct / self.direction_available
            )
        return payload


def expand_to_batch(value: Any, total: int) -> List[Any]:
    """
    Normalise auxiliary metadata so it matches the batch size.

    :param value: Scalar or sequence metadata.
    :type value: Any
    :param total: Desired batch length.
    :type total: int
    :returns: List of length ``total`` populated with ``value``.
    :rtype: list[Any]
    """

    if total == 0:
        return []
    if isinstance(value, (list, tuple)):
        seq = list(value)
        if len(seq) >= total:
            return seq[:total]
        if not seq:
            return [None] * total
        return seq + [None] * (total - len(seq))
    if value is None:
        return [None] * total
    return [value] * total


def _safe_int(value: Any) -> Optional[int]:
    """Return an int if possible, otherwise ``None``."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_index_from_completion(text: str, allow_bare: bool) -> Optional[int]:
    """
    Extract numeric answers from completion text.

    :param text: Completion text to parse.
    :type text: str
    :param allow_bare: Whether bare numeric answers are permitted.
    :type allow_bare: bool
    :returns: Parsed index or ``None`` if parsing fails.
    :rtype: int | None
    """

    answer_match = _ANS_PAT.search(text)
    if answer_match:
        payload = answer_match.group(1).strip()
    elif allow_bare:
        payload = text.strip()
    else:
        return None

    index_match = _INDEX_ONLY_RE.match(payload)
    if not index_match:
        return None
    return _safe_int(index_match.group(1))


def _normalise_opinion_label(value: Any) -> Optional[str]:
    """Map free-form direction tokens onto canonical labels."""

    if not isinstance(value, str):
        return None
    token = re.sub(r"[^a-z]+", "", value.lower())
    if not token:
        return None
    if token in _OPINION_INCREASE:
        return "increase"
    if token in _OPINION_DECREASE:
        return "decrease"
    if token in _OPINION_NO_CHANGE or token == "nochange":
        return "no_change"
    return None


def _parse_opinion_direction(text: str) -> Optional[str]:
    """Extract the opinion-direction label from completion text."""

    for pattern in _OPINION_PATTERNS:
        match = pattern.search(text)
        if match:
            return _normalise_opinion_label(match.group(1))
    return None


def score_video_prediction(
    predicted_index: Optional[int],
    gold_value: Any,
    options_value: Any,
) -> Tuple[float, bool, bool]:
    """
    Score the next-video prediction for a single sample.

    :param predicted_index: Parsed choice index from the policy completion.
    :type predicted_index: int | None
    :param gold_value: Ground-truth index or identifier.
    :type gold_value: Any
    :param options_value: Number of options presented to the policy.
    :type options_value: Any
    :returns: Tuple of ``(score, video_metadata_available, eligible_for_reward)``.
    :rtype: tuple[float, bool, bool]
    """

    gold_idx = _safe_int(gold_value)
    if gold_idx is None or gold_idx <= 0:
        return 0.0, False, False

    if predicted_index is None:
        return 0.0, True, False

    max_options = _safe_int(options_value)
    if max_options is not None and (max_options <= 0 or not 1 <= predicted_index <= max_options):
        return 0.0, True, True

    return (1.0 if predicted_index == gold_idx else 0.0), True, True


def score_opinion_prediction(text: str, expected_value: Any) -> Tuple[float, int, int, int]:
    """
    Score the opinion-direction prediction for a single sample.

    :param text: Completion text emitted by the policy.
    :type text: str
    :param expected_value: Ground-truth opinion label or alias.
    :type expected_value: Any
    :returns: Tuple containing ``(score, available, parsed, correct)`` flags.
    :rtype: tuple[float, int, int, int]
    """

    expected_dir = _normalise_opinion_label(expected_value)
    if expected_dir is None:
        return 0.0, 0, 0, 0

    predicted_dir = _parse_opinion_direction(text)
    if predicted_dir is None:
        return 0.0, 1, 0, 0

    is_correct = int(predicted_dir == expected_dir)
    return float(is_correct), 1, 1, is_correct


@dataclass
class PureAccuracyContext:
    """
    Bundle configuration and mutable state for :func:`pure_accuracy_reward`.
    """

    allow_bare: bool
    stats: PureAccuracyStats
    completion_to_text: Callable[[Any], str]

    def score(
        self,
        completion: Any,
        gold_value: Any,
        option_value: Any,
        opinion_target: Any,
    ) -> float:
        """
        Evaluate a single completion and update aggregate statistics.

        :param completion: Completion object or text emitted by the policy.
        :type completion: Any
        :param gold_value: Ground-truth next-video index.
        :type gold_value: Any
        :param option_value: Number of recommendation options.
        :type option_value: Any
        :param opinion_target: Opinion-direction label (may be ``None``).
        :type opinion_target: Any
        :returns: Reward value for the completion.
        :rtype: float
        """

        text = self.completion_to_text(completion)
        predicted_index = parse_index_from_completion(text, self.allow_bare)
        if predicted_index is not None:
            self.stats.parsed += 1

        video_score, has_video, eligible = score_video_prediction(
            predicted_index,
            gold_value,
            option_value,
        )
        if eligible:
            self.stats.eligible += 1

        opinion_score, available, parsed, correct = score_opinion_prediction(
            text,
            opinion_target,
        )
        self.stats.direction_available += available
        self.stats.direction_parsed += parsed
        self.stats.direction_correct += correct

        weight = int(has_video) + available
        if weight == 0:
            weight = 1

        return (
            (video_score if has_video else 0.0)
            + (opinion_score if available else 0.0)
        ) / weight


def log_pure_accuracy_metrics(
    total: int,
    outs: List[float],
    stats: PureAccuracyStats,
    logger: Optional[Callable[[Dict[str, float]], None]],
) -> None:
    """
    Emit wandb-compatible metrics for :func:`pure_accuracy_reward`.

    :param total: Number of samples processed.
    :type total: int
    :param outs: Per-sample rewards.
    :type outs: list[float]
    :param stats: Aggregate counters gathered during scoring.
    :type stats: PureAccuracyStats
    :param logger: Logging callback, typically wandb's ``log``.
    :type logger: Callable[[dict[str, float]], None] | None
    """

    if total == 0:
        return

    if not callable(logger):
        return

    batch_mean = sum(outs) / len(outs) if outs else math.nan
    payload = stats.payload(total, batch_mean)
    if not payload:
        return
    try:
        logger(payload)
    except (TypeError, ValueError):
        pass


__all__ = [
    "PureAccuracyContext",
    "PureAccuracyStats",
    "expand_to_batch",
    "log_pure_accuracy_metrics",
    "parse_index_from_completion",
    "score_opinion_prediction",
    "score_video_prediction",
]
