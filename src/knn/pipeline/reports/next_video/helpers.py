"""Shared helper utilities for next-video report submodules."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Mapping, Sequence, Tuple

from ...context import MetricSummary, StudySpec
from ...utils import extract_metric_summary

CANONICAL_FEATURE_SPACES: tuple[str, ...] = ("tfidf", "word2vec", "sentence_transformer")


def _ordered_feature_spaces(
    preferred: Sequence[str],
    available: Iterable[str],
) -> List[str]:
    """
    Return feature spaces prioritised by canonical order then user preference.

    :param preferred: Ordered feature spaces requested by the caller.
    :param available: Feature spaces present in the metrics mapping.
    :returns: Combined ordered list of feature spaces.
    """
    ordered: List[str] = []
    for space in CANONICAL_FEATURE_SPACES:
        if space in preferred or space in available:
            ordered.append(space)
    for space in preferred:
        if space not in ordered:
            ordered.append(space)
    for space in available:
        if space not in ordered:
            ordered.append(space)
    return ordered


def iter_metric_payloads(
    metrics: Mapping[str, Mapping[str, object]],
    studies: Sequence[StudySpec],
) -> Iterator[Tuple[StudySpec, MetricSummary, Mapping[str, object]]]:
    """
    Yield per-study metric summaries alongside their raw payloads.

    :param metrics: Mapping of study key to raw metric payloads.
    :type metrics: Mapping[str, Mapping[str, object]]
    :param studies: Ordered sequence of study specifications.
    :type studies: Sequence[~knn.pipeline.context.StudySpec]
    :returns: Iterator yielding ``(study, summary, payload)`` tuples.
    :rtype: Iterator[Tuple[~knn.pipeline.context.StudySpec, MetricSummary, Mapping[str, object]]]
    """
    for study in studies:
        payload = metrics.get(study.key)
        if not payload:
            continue
        summary = extract_metric_summary(payload)
        yield study, summary, payload


def accuracy_delta(summary: MetricSummary) -> float | None:
    """
    Return the accuracy delta vs. baseline for ``summary`` when available.

    :param summary: Metric summary with ``accuracy`` and ``baseline`` attributes.
    :returns: Float delta or ``None`` if either component is missing.
    """
    accuracy = getattr(summary, "accuracy", None)
    baseline = getattr(summary, "baseline", None)
    if accuracy is None or baseline is None:
        return None
    try:
        return accuracy - baseline
    except TypeError:  # pragma: no cover - defensive for unexpected payloads
        return None
