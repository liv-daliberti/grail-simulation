"""Markdown section builders for the next-video report."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

from ...context import StudySpec
from ...utils import (
    extract_metric_summary,
    format_count,
    format_delta,
    format_k,
    format_optional_float,
    format_uncertainty_details,
    parse_ci,
)
from ..shared import _feature_space_heading
from .helpers import accuracy_delta, _ordered_feature_spaces, iter_metric_payloads
from .inputs import PortfolioAggregate, _ObservationAccumulator, _PortfolioAccumulator

PORTFOLIO_HEADER = (
    "| Feature space | Weighted accuracy ↑ | Δ vs baseline ↑ | Random ↑ | Eligible | "
    "Studies |"
)
PORTFOLIO_RULE = "| --- | ---: | ---: | ---: | ---: | ---: |"
FEATURE_TABLE_HEADER = (
    "| Study | Accuracy ↑ | Accuracy (all rows) ↑ | 95% CI | Δ vs baseline ↑ | "
    "Baseline ↑ | Random ↑ | Best k | Eligible | Total |"
)
FEATURE_TABLE_RULE = (
    "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
)


def _aggregate_portfolio_metrics(
    metrics: Mapping[str, Mapping[str, object]],
) -> Optional[PortfolioAggregate]:
    """
    Compute weighted aggregate metrics across studies for a feature space.

    :param metrics: Study-level metrics for the feature space.
    :returns: Weighted aggregates or None if no eligible studies.
    """
    if not metrics:
        return None

    accumulator = _PortfolioAccumulator()
    for payload in metrics.values():
        accumulator.add(extract_metric_summary(payload))
    return accumulator.result()


def _summarise_feature_observations(
    feature_space: str,
    metrics: Mapping[str, Mapping[str, object]],
    studies: Sequence[StudySpec],
) -> Optional[str]:
    """
    Summarise qualitative observations for a feature space.

    :param feature_space: Feature space identifier such as ``tfidf``.
    :param metrics: Study-level metrics for the feature space.
    :param studies: Ordered study specifications.
    :returns: Markdown bullet or None when there is nothing to report.
    """
    bullet_parts: list[str] = []
    stats = _ObservationAccumulator()
    for study, summary, _payload in iter_metric_payloads(metrics, studies):
        accuracy = summary.accuracy
        if accuracy is None:
            continue
        baseline_val = summary.baseline
        delta_val = accuracy_delta(summary)
        stats.record(summary)
        bullet_parts.append(
            f"{study.label}: {format_optional_float(accuracy)} "
            f"(baseline {format_optional_float(baseline_val)}, "
            f"Δ {format_delta(delta_val)}, "
            f"k={format_k(summary.best_k)}, eligible {format_count(summary.n_eligible)})"
        )
    if not bullet_parts:
        return None
    extras = stats.averages()
    if extras:
        bullet_parts.append("averages: " + ", ".join(extras))
    joined = "; ".join(bullet_parts)
    return f"- {feature_space.upper()}: {joined}."


def _next_video_dataset_info(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Tuple[str, str]:
    """
    Extract a representative dataset name and split from metrics payloads.

    :param metrics_by_feature: Nested mapping of metrics grouped by feature space and study.
    :returns: Dataset name and split identifier.
    """
    for per_feature in metrics_by_feature.values():
        for study_metrics in per_feature.values():
            dataset = study_metrics.get("dataset")
            split = study_metrics.get("split", "validation")
            if dataset:
                return str(dataset), str(split)
    raise RuntimeError("No slate metrics available to build the next-video report.")


def _next_video_uncertainty_info(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Optional[Mapping[str, object]]:
    """
    Return the first uncertainty payload available for reporting.

    :param metrics_by_feature: Nested mapping of metrics grouped by feature space and study.
    :returns: Uncertainty payload if present.
    """
    for per_feature in metrics_by_feature.values():
        for study_metrics in per_feature.values():
            uncertainty = study_metrics.get("uncertainty")
            if isinstance(uncertainty, Mapping):
                return uncertainty
    return None


def _next_video_intro(
    dataset_name: str,
    split: str,
    uncertainty: Optional[Mapping[str, object]] = None,
) -> list[str]:
    """
    Return the introductory Markdown section for the next-video report.

    :param dataset_name: Human-readable label for the dataset being summarised.
    :param split: Dataset split identifier such as ``train`` or ``validation``.
    :param uncertainty: Auxiliary uncertainty information accompanying a metric.
    :returns: Markdown intro lines.
    """
    intro = [
        "# KNN Next-Video Baseline",
        "",
        "This report summarises the slate-ranking KNN model that predicts the next clicked video.",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Split: {split}",
        "- Metric: eligible-only accuracy (gold index present).",
        (
            "- Note: an all-rows accuracy (including ineligible slates) is also recorded in the "
            "per-study metrics as `accuracy_overall_all_rows` to ease comparison with XGB's "
            "overall accuracy."
        ),
        "- Baseline column: accuracy from recommending the most frequent gold index.",
        "- Δ column: improvement over that baseline accuracy.",
        "- Random column: expected accuracy from uniformly sampling one candidate per slate.",
    ]
    if uncertainty:
        method = str(uncertainty.get("method", "unknown"))
        intro.append(f"- Uncertainty: {method}{format_uncertainty_details(uncertainty)}")
    intro.append("")
    return intro


def _next_video_portfolio_summary(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    feature_spaces: Sequence[str],
) -> list[str]:
    """
    Render weighted portfolio statistics across feature spaces.

    :param metrics_by_feature: Nested mapping of metrics grouped by feature space.
    :param feature_spaces: Ordered feature spaces considered by the pipeline.
    :returns: Markdown lines summarising portfolio-level performance.
    """
    ordered_spaces = _ordered_feature_spaces(feature_spaces, metrics_by_feature)
    rows: list[str] = []
    best_space: Optional[str] = None
    best_metrics: Optional[PortfolioAggregate] = None

    for feature_space in ordered_spaces:
        aggregates = _aggregate_portfolio_metrics(metrics_by_feature.get(feature_space, {}))
        if aggregates is None:
            continue
        delta_value: Optional[float] = None
        if aggregates.accuracy is not None and aggregates.baseline is not None:
            delta_value = aggregates.accuracy - aggregates.baseline
        rows.append(
            f"| {feature_space.upper()} | {format_optional_float(aggregates.accuracy)} | "
            f"{format_delta(delta_value)} | {format_optional_float(aggregates.random)} | "
            f"{format_count(aggregates.eligible)} | {format_count(aggregates.studies)} |"
        )
        if aggregates.accuracy is not None:
            if (
                best_metrics is None
                or best_metrics.accuracy is None
                or aggregates.accuracy > best_metrics.accuracy
            ):
                best_space = feature_space
                best_metrics = aggregates

    if not rows:
        return []

    lines: list[str] = ["## Portfolio Summary", ""]
    lines.append(PORTFOLIO_HEADER)
    lines.append(PORTFOLIO_RULE)
    lines.extend(rows)
    lines.append("")
    if best_space and best_metrics:
        lines.append(
            f"Best-performing feature space: **{best_space.upper()}** with weighted "
            f"accuracy {format_optional_float(best_metrics.accuracy)} across "
            f"{format_count(best_metrics.eligible)} eligible slates "
            f"({format_count(best_metrics.studies)} studies)."
        )
        lines.append("")
    return lines


def _format_ci(ci_value: object) -> str:
    """
    Format a 95% confidence interval if present.

    :param ci_value: Confidence-interval payload extracted from a metrics dictionary.
    :returns: Human-readable confidence-interval string formatted for Markdown tables.
    """
    confidence_interval = parse_ci(ci_value)
    if confidence_interval is None:
        return "—"
    low, high = confidence_interval
    return f"[{low:.3f}, {high:.3f}]"


def _next_video_feature_section(
    feature_space: str,
    metrics: Mapping[str, Mapping[str, object]],
    studies: Sequence[StudySpec],
) -> list[str]:
    """
    Render the next-video metrics table for ``feature_space``.

    :param feature_space: Feature space identifier such as ``tfidf`` or ``word2vec``.
    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :param studies: Sequence of study specifications targeted by the workflow.
    :returns: Markdown section summarising next-video metrics for the feature space.
    """
    if not metrics:
        return []
    lines: list[str] = [
        _feature_space_heading(feature_space),
        "",
        FEATURE_TABLE_HEADER,
        FEATURE_TABLE_RULE,
    ]
    for study, summary, _payload in iter_metric_payloads(metrics, studies):
        delta = accuracy_delta(summary)
        lines.append(
            f"| {study.label} | {format_optional_float(summary.accuracy)} | "
            f"{format_optional_float(getattr(summary, 'accuracy_all_rows', None))} | "
            f"{_format_ci(summary.accuracy_ci)} | {format_delta(delta)} | "
            f"{format_optional_float(summary.baseline)} | "
            f"{format_optional_float(summary.random_baseline)} | "
            f"{format_k(summary.best_k)} | {format_count(summary.n_eligible)} | "
            f"{format_count(summary.n_total)} |"
        )
    lines.append("")
    return lines


def _next_video_observations(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> list[str]:
    """
    Summarise per-feature observations for next-video metrics.

    :param metrics_by_feature: Nested mapping of metrics grouped by feature space and study.
    :param studies: Sequence of study specifications targeted by the workflow.
    :returns: Markdown bullet list capturing qualitative next-video observations.
    """
    lines: list[str] = ["## Observations", ""]
    ordered_spaces = _ordered_feature_spaces((), metrics_by_feature)
    for feature_space in ordered_spaces:
        observation = _summarise_feature_observations(
            feature_space,
            metrics_by_feature.get(feature_space, {}),
            studies,
        )
        if observation:
            lines.append(observation)
    lines.append(
        "- Random values approximate the accuracy from uniformly guessing across the slate."
    )
    lines.append("")
    return lines


__all__ = [
    "_aggregate_portfolio_metrics",
    "_format_ci",
    "_next_video_dataset_info",
    "_next_video_feature_section",
    "_next_video_intro",
    "_next_video_observations",
    "_next_video_portfolio_summary",
    "_next_video_uncertainty_info",
    "_summarise_feature_observations",
    "FEATURE_TABLE_HEADER",
    "FEATURE_TABLE_RULE",
    "PORTFOLIO_HEADER",
    "PORTFOLIO_RULE",
]
