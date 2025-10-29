"""Leave-one-study-out reporting helpers for the next-video report."""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from ...context import StudySpec
from ..shared import _feature_space_heading
from ...utils import (
    format_count,
    format_delta,
    format_k,
    format_optional_float,
)
from .helpers import (
    CANONICAL_FEATURE_SPACES,
    accuracy_delta,
    iter_metric_payloads,
)

if TYPE_CHECKING:
    from ...context import MetricSummary
else:  # pragma: no cover - type hint fallback
    MetricSummary = object

LOSO_TABLE_HEADER = (
    "| Holdout study | Accuracy ↑ | Δ vs baseline ↑ | Baseline ↑ | Best k | Eligible |"
)
LOSO_TABLE_RULE = "| --- | ---: | ---: | ---: | ---: | ---: |"


def _ordered_loso_feature_spaces(
    loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> List[str]:
    """
    Return feature spaces ordered by canonical preference then presence.

    :param loso_metrics: Nested mapping of leave-one-study-out metrics by feature space.
    :returns: Ordered list of feature-space identifiers.
    """

    ordered = [space for space in CANONICAL_FEATURE_SPACES if space in loso_metrics]
    ordered.extend(space for space in loso_metrics if space not in ordered)
    return ordered


def _collect_loso_summaries(
    metrics: Mapping[str, Mapping[str, object]],
    studies: Sequence[StudySpec],
) -> List[Tuple[str, "MetricSummary", Optional[float]]]:
    """
    Return holdout summaries paired with their accuracy deltas.

    :param metrics: LOSO metrics keyed by study key.
    :param studies: Ordered study specifications used to align extracted summaries.
    :returns: List of tuples ``(label, summary, delta_vs_baseline)``.
    """

    summaries: List[Tuple[str, MetricSummary, Optional[float]]] = []
    for study, summary, _payload in iter_metric_payloads(metrics, studies):
        delta = accuracy_delta(summary)
        summaries.append((study.label, summary, delta))
    return summaries


def _loso_highlight_lines(
    summaries: Sequence[Tuple[str, "MetricSummary", Optional[float]]]
) -> List[str]:
    """
    Return highlight bullets for key holdout statistics.

    :param summaries: Sequence of tuples produced by :func:`_collect_loso_summaries`.
    :returns: Bullet lines highlighting key holdout insights.
    """

    if not summaries:
        return []
    lines: List[str] = []
    accuracy_entries = [
        (label, summary.accuracy, delta)
        for label, summary, delta in summaries
        if summary.accuracy is not None
    ]
    if accuracy_entries:
        best_label, best_acc, best_delta = max(
            accuracy_entries, key=lambda item: item[1] or float("-inf")
        )
        lines.append(
            f"- Highest holdout accuracy: {best_label} "
            f"({format_optional_float(best_acc)}) "
            f"{format_delta(best_delta)} vs. baseline."
        )
        worst_label, worst_acc, worst_delta = min(
            accuracy_entries, key=lambda item: item[1] or float("inf")
        )
        lines.append(
            f"- Lowest holdout accuracy: {worst_label} "
            f"({format_optional_float(worst_acc)}) "
            f"{format_delta(worst_delta)} vs. baseline."
        )
    delta_values = [delta for _label, _summary, delta in summaries if delta is not None]
    if delta_values:
        lines.append(
            f"- Average accuracy delta across holdouts: "
            f"{format_delta(sum(delta_values) / len(delta_values))}."
        )
    return lines


def _loso_table_rows(
    metrics: Mapping[str, Mapping[str, object]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Return Markdown table rows for leave-one-study-out metrics.

    :param metrics: LOSO metrics keyed by study key.
    :param studies: Ordered study specifications used to derive row order.
    :returns: Markdown table rows describing per-holdout results.
    """

    rows: List[str] = []
    for study, summary, _payload in iter_metric_payloads(metrics, studies):
        delta = accuracy_delta(summary)
        rows.append(
            f"| {study.label} | {format_optional_float(summary.accuracy)} | "
            f"{format_delta(delta)} | {format_optional_float(summary.baseline)} | "
            f"{format_k(summary.best_k)} | {format_count(summary.n_eligible)} |"
        )
    return rows


def _next_video_loso_section(
    loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Return a section summarising leave-one-study-out accuracy.

    :param loso_metrics: Leave-one-study-out metrics grouped by feature space and holdout.
    :param studies: Sequence of study specifications targeted by the workflow.
    :returns: Markdown section summarising leave-one-study-out accuracy.
    """

    if not loso_metrics:
        return []
    lines: List[str] = ["## Cross-Study Holdouts", ""]
    for feature_space in _ordered_loso_feature_spaces(loso_metrics):
        metrics = loso_metrics.get(feature_space, {})
        rows = _loso_table_rows(metrics, studies)
        if not rows:
            continue
        lines.append(_feature_space_heading(feature_space))
        lines.append("")
        highlights = _loso_highlight_lines(_collect_loso_summaries(metrics, studies))
        if highlights:
            lines.append("Key holdout takeaways:")
            lines.append("")
            lines.extend(highlights)
            lines.append("")
        lines.append(LOSO_TABLE_HEADER)
        lines.append(LOSO_TABLE_RULE)
        lines.extend(rows)
        lines.append("")
    return lines


__all__ = [
    "LOSO_TABLE_HEADER",
    "LOSO_TABLE_RULE",
    "_collect_loso_summaries",
    "_loso_highlight_lines",
    "_loso_table_rows",
    "_next_video_loso_section",
    "_ordered_loso_feature_spaces",
]
