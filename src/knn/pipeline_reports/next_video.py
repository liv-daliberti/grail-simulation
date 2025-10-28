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

"""Next-video report builders for the modular KNN pipeline."""
# pylint: disable=too-many-lines

from __future__ import annotations

from dataclasses import dataclass
import json
import csv
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from common.matplotlib_utils import plt
from common.report_utils import extract_curve_sections, extract_numeric_series

from ..pipeline_context import StudySpec
from ..pipeline_utils import (
    extract_metric_summary,
    format_count,
    format_delta,
    format_k,
    format_optional_float,
    format_uncertainty_details,
    parse_ci,
)
from .shared import LOGGER, _feature_space_heading

if TYPE_CHECKING:
    from ..pipeline_context import MetricSummary
else:  # pragma: no cover - type hint fallback
    MetricSummary = Any


@dataclass
class PortfolioAggregate:
    """Weighted portfolio metrics for a feature space."""

    accuracy: Optional[float]
    baseline: Optional[float]
    random: Optional[float]
    eligible: int
    studies: int


@dataclass
class _PortfolioAccumulator:
    """Mutable helper used to compute weighted aggregates."""

    accuracy_total: float = 0.0
    accuracy_weight: int = 0
    baseline_total: float = 0.0
    baseline_weight: int = 0
    random_total: float = 0.0
    random_weight: int = 0
    studies_with_metrics: int = 0

    def add(self, summary: MetricSummary) -> None:
        """Record weighted contributions from ``summary`` when eligible."""

        eligible = summary.n_eligible
        if not eligible:
            return

        recorded = False
        for total_attr, weight_attr, field_name in (
            ("accuracy_total", "accuracy_weight", "accuracy"),
            ("baseline_total", "baseline_weight", "baseline"),
            ("random_total", "random_weight", "random_baseline"),
        ):
            value = getattr(summary, field_name)
            if value is None:
                continue
            setattr(self, total_attr, getattr(self, total_attr) + value * eligible)
            setattr(self, weight_attr, getattr(self, weight_attr) + eligible)
            recorded = True

        if recorded:
            self.studies_with_metrics += 1

    def result(self) -> Optional[PortfolioAggregate]:
        """Return the weighted aggregates accumulated so far."""

        if not self.studies_with_metrics:
            return None

        def _average(total: float, weight: int) -> Optional[float]:
            return total / weight if weight else None

        weighted_accuracy = _average(self.accuracy_total, self.accuracy_weight)
        weighted_baseline = _average(self.baseline_total, self.baseline_weight)
        weighted_random = _average(self.random_total, self.random_weight)
        eligible_total = max(self.accuracy_weight, self.baseline_weight, self.random_weight)

        return PortfolioAggregate(
            accuracy=weighted_accuracy,
            baseline=weighted_baseline,
            random=weighted_random,
            eligible=eligible_total,
            studies=self.studies_with_metrics,
        )


@dataclass
class _ObservationAccumulator:
    """Running statistics for qualitative observations."""

    delta_sum: float = 0.0
    delta_count: int = 0
    random_sum: float = 0.0
    random_count: int = 0

    def record(self, summary: MetricSummary) -> None:
        """Update accumulated statistics with ``summary``."""

        delta = (
            summary.accuracy - summary.baseline
            if summary.accuracy is not None and summary.baseline is not None
            else None
        )
        if delta is not None:
            self.delta_sum += delta
            self.delta_count += 1

        random_value = summary.random_baseline
        if random_value is not None:
            self.random_sum += random_value
            self.random_count += 1

    def averages(self) -> List[str]:
        """Return formatted average statistics when available."""

        extras: List[str] = []
        if self.delta_count:
            extras.append(
                f"mean Δ {format_delta(self.delta_sum / self.delta_count)}"
            )
        if self.random_count:
            extras.append(
                f"mean random {format_optional_float(self.random_sum / self.random_count)}"
            )
        return extras


@dataclass
class NextVideoReportInputs:
    """Input bundle required to render the next-video report."""

    output_dir: Path
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    studies: Sequence[StudySpec]
    feature_spaces: Sequence[str]
    loso_metrics: Optional[Mapping[str, Mapping[str, Mapping[str, object]]]] = None
    allow_incomplete: bool = False
    xgb_next_video_dir: Optional[Path] = None

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
LOSO_TABLE_HEADER = (
    "| Holdout study | Accuracy ↑ | Δ vs baseline ↑ | Baseline ↑ | Best k | Eligible |"
)
LOSO_TABLE_RULE = "| --- | ---: | ---: | ---: | ---: | ---: |"


def _ordered_feature_spaces(
    preferred: Sequence[str],
    available: Iterable[str],
) -> List[str]:
    """
    Return feature spaces prioritised by canonical order then user preference.

    :param preferred: Ordered feature spaces requested by the caller.
    :type preferred: Sequence[str]
    :param available: Feature spaces present in the metrics mapping.
    :type available: Iterable[str]
    :returns: Combined ordered list of feature spaces.
    :rtype: List[str]
    """
    canonical = ["tfidf", "word2vec", "sentence_transformer"]
    ordered: List[str] = []
    for space in canonical:
        if space in preferred or space in available:
            ordered.append(space)
    for space in preferred:
        if space not in ordered:
            ordered.append(space)
    for space in available:
        if space not in ordered:
            ordered.append(space)
    return ordered


def _aggregate_portfolio_metrics(
    metrics: Mapping[str, Mapping[str, object]],
) -> Optional[PortfolioAggregate]:
    """
    Compute weighted aggregate metrics across studies for a feature space.

    :param metrics: Study-level metrics for the feature space.
    :type metrics: Mapping[str, Mapping[str, object]]
    :returns: Weighted aggregates or None if no eligible studies.
    :rtype: Optional[PortfolioAggregate]
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
    :type feature_space: str
    :param metrics: Study-level metrics for the feature space.
    :type metrics: Mapping[str, Mapping[str, object]]
    :param studies: Ordered study specifications.
    :type studies: Sequence[StudySpec]
    :returns: Markdown bullet or None when there is nothing to report.
    :rtype: Optional[str]
    """
    bullet_parts: List[str] = []
    stats = _ObservationAccumulator()
    for study in studies:
        payload = metrics.get(study.key)
        if not payload:
            continue
        summary = extract_metric_summary(payload)
        accuracy = summary.accuracy
        if accuracy is None:
            continue
        baseline_val = summary.baseline
        delta_val = accuracy - baseline_val if baseline_val is not None else None
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
    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    :returns: Dataset name and split identifier.
    :rtype: Tuple[str, str]
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
    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    :returns: Uncertainty payload if present.
    :rtype: Optional[Mapping[str, object]]
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
) -> List[str]:
    """
    Return the introductory Markdown section for the next-video report.

    :param dataset_name: Human-readable label for the dataset being summarised.
    :type dataset_name: str
    :param split: Dataset split identifier such as ``train`` or ``validation``.
    :type split: str
    :param uncertainty: Auxiliary uncertainty information accompanying a metric.
    :type uncertainty: Optional[Mapping[str, object]]
    :returns: Markdown intro lines.
    :rtype: List[str]
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
        intro.append(
            f"- Uncertainty: {method}{format_uncertainty_details(uncertainty)}"
        )
    intro.append("")
    return intro


def _next_video_portfolio_summary(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    feature_spaces: Sequence[str],
) -> List[str]:
    """
    Render weighted portfolio statistics across feature spaces.

    :param metrics_by_feature: Nested mapping of metrics grouped by feature space.
    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param feature_spaces: Ordered feature spaces considered by the pipeline.
    :type feature_spaces: Sequence[str]
    :returns: Markdown lines summarising portfolio-level performance.
    :rtype: List[str]
    """
    ordered_spaces = _ordered_feature_spaces(feature_spaces, metrics_by_feature)
    rows: List[str] = []
    best_space: Optional[str] = None
    best_metrics: Optional[PortfolioAggregate] = None

    for feature_space in ordered_spaces:
        aggregates = _aggregate_portfolio_metrics(
            metrics_by_feature.get(feature_space, {})
        )
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

    lines: List[str] = ["## Portfolio Summary", ""]
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
    :type ci_value: object
    :returns: Human-readable confidence-interval string formatted for Markdown tables.
    :rtype: str
    """
    confidence_interval = parse_ci(ci_value)
    if confidence_interval is None:
        return "—"
    low, high = confidence_interval
    return f"[{low:.3f}, {high:.3f}]"


def _extract_curve_series(curve_block: Mapping[str, object]) -> Tuple[List[int], List[float]]:
    """
    Return sorted k/accuracy pairs extracted from ``curve_block``.

    :param curve_block: Markdown block describing a single KNN performance curve.
    :type curve_block: Mapping[str, object]
    :returns: Sorted k/accuracy pairs extracted from ``curve_block``.
    :rtype: Tuple[List[int], List[float]]
    """
    accuracy_map = curve_block.get("accuracy_by_k")
    if not isinstance(accuracy_map, Mapping):
        return ([], [])
    return extract_numeric_series(accuracy_map)


def _plot_knn_curve_bundle(
    *,
    base_dir: Path,
    feature_space: str,
    study: StudySpec,
    metrics: Mapping[str, object],
) -> Optional[str]:
    """
    Save a train/validation accuracy curve plot for ``study`` when possible.

    :param base_dir: Base directory that contains task-specific output subdirectories.
    :type base_dir: Path
    :param feature_space: Feature space identifier such as ``tfidf`` or ``word2vec``.
    :type feature_space: str
    :param study: Study specification for the item currently being processed.
    :type study: StudySpec
    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :type metrics: Mapping[str, object]
    :returns: Relative path to the generated curve plot.
    :rtype: Optional[str]
    """
    if plt is None:  # pragma: no cover - optional dependency
        return None
    sections = extract_curve_sections(metrics.get("curve_metrics"))
    if sections is None:
        return None
    eval_curve, train_curve = sections
    eval_x, eval_y = _extract_curve_series(eval_curve)
    if not eval_x:
        return None
    train_x, train_y = (
        _extract_curve_series(train_curve) if train_curve is not None else ([], [])
    )

    curves_dir = base_dir / "curves" / feature_space
    curves_dir.mkdir(parents=True, exist_ok=True)
    plot_path = curves_dir / f"{study.study_slug}.png"

    fig, axis = plt.subplots(figsize=(6, 3.5))  # type: ignore[attr-defined]
    axis.plot(eval_x, eval_y, marker="o", label="validation")
    if train_x and train_y:
        axis.plot(train_x, train_y, marker="o", linestyle="--", label="training")
    axis.set_title(f"{study.label} – {feature_space.upper()}")
    axis.set_xlabel("k")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(eval_x)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)  # type: ignore[attr-defined]
    plt.close(fig)  # type: ignore[attr-defined]
    try:
        return plot_path.relative_to(base_dir).as_posix()
    except ValueError:
        return plot_path.as_posix()


def _next_video_feature_section(
    feature_space: str,
    metrics: Mapping[str, Mapping[str, object]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Render the next-video metrics table for ``feature_space``.

    :param feature_space: Feature space identifier such as ``tfidf`` or ``word2vec``.
    :type feature_space: str
    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :type metrics: Mapping[str, Mapping[str, object]]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[StudySpec]
    :returns: Markdown section summarising next-video metrics for the feature space.
    :rtype: List[str]
    """
    if not metrics:
        return []
    lines: List[str] = [
        _feature_space_heading(feature_space),
        "",
        FEATURE_TABLE_HEADER,
        FEATURE_TABLE_RULE,
    ]
    for study in studies:
        data = metrics.get(study.key)
        if not data:
            continue
        summary = extract_metric_summary(data)
        delta = (
            summary.accuracy - summary.baseline
            if summary.accuracy is not None and summary.baseline is not None
            else None
        )
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


def _next_video_curve_sections(
    *,
    output_dir: Path,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Render Markdown sections embedding train/validation accuracy curves.

    :param output_dir: Directory where the rendered report should be written.
    :type output_dir: Path
    :param metrics_by_feature: Nested mapping of metrics grouped by feature space and study.
    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[StudySpec]
    :returns: Markdown sections that document next-video learning curves.
    :rtype: List[str]
    """
    if plt is None:  # pragma: no cover - optional dependency
        LOGGER.debug("Matplotlib not available; skipping KNN curve plots.")
        return []

    sections: List[str] = []
    ordered_spaces = _ordered_feature_spaces((), metrics_by_feature)

    for feature_space in ordered_spaces:
        feature_metrics = metrics_by_feature.get(feature_space, {})
        if not feature_metrics:
            continue
        image_lines: List[str] = []
        for study in studies:
            study_metrics = feature_metrics.get(study.key)
            if not study_metrics:
                continue
            rel_path = _plot_knn_curve_bundle(
                base_dir=output_dir,
                feature_space=feature_space,
                study=study,
                metrics=study_metrics,
            )
            if rel_path:
                image_lines.extend(
                    [
                        f"### {study.label} ({feature_space.upper()})",
                        "",
                        f"![Accuracy curve]({rel_path})",
                        "",
                    ]
                )
        sections.extend(image_lines)
    return sections


def _next_video_observations(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Summarise per-feature observations for next-video metrics.

    :param metrics_by_feature: Nested mapping of metrics grouped by feature space and study.
    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[StudySpec]
    :returns: Markdown bullet list capturing qualitative next-video observations.
    :rtype: List[str]
    """
    lines: List[str] = ["## Observations", ""]
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


def _format_feature_inline(space: str) -> str:
    """Return a compact inline label for a feature space."""
    if space == "sentence_transformer":
        return "ST"
    return space.upper()


def _load_xgb_metrics_for_study(
    base_dir: Path,
    evaluation_slug: str,
) -> Optional[Mapping[str, object]]:
    """Load XGB metrics for a single study when available."""
    metrics_path = base_dir / evaluation_slug / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
        return None


def _best_knn_summary_for_study(
    *,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    study_key: str,
) -> Optional[Tuple[str, "MetricSummary"]]:
    """Select the best KNN feature-space summary for ``study_key`` by eligible-only accuracy."""
    best: Optional[Tuple[str, MetricSummary]] = None
    for space, per_study in metrics_by_feature.items():
        payload = per_study.get(study_key)
        if not payload:
            continue
        summary = extract_metric_summary(payload)
        if summary.accuracy is None:
            continue
        if best is None:
            best = (space, summary)
            continue
        best_accuracy = best[1].accuracy or 0.0
        current_accuracy = summary.accuracy or 0.0
        if best_accuracy < current_accuracy:
            best = (space, summary)
    return best


def _knn_vs_xgb_section(  # pylint: disable=too-many-locals
    *,
    xgb_next_video_dir: Path,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Build a comparison section for matched KNN/XGB studies when XGB metrics are present."""
    if not xgb_next_video_dir.exists():
        return []
    header = (
        "| Study | KNN (feature) eligible-only ↑ | XGB eligible-only ↑ | "
        "KNN all-rows ↑ | XGB overall ↑ |"
    )
    rule = "| --- | ---: | ---: | ---: | ---: |"
    rows: List[str] = []
    for spec in studies:
        xgb_metrics = _load_xgb_metrics_for_study(xgb_next_video_dir, spec.evaluation_slug)
        if not isinstance(xgb_metrics, Mapping):
            continue
        best_knn = _best_knn_summary_for_study(
            metrics_by_feature=metrics_by_feature,
            study_key=spec.key,
        )
        if best_knn is None:
            continue
        space, knn_summary = best_knn
        knn_elig = knn_summary.accuracy
        knn_all = getattr(knn_summary, "accuracy_all_rows", None)
        xgb_elig = xgb_metrics.get("accuracy_eligible")
        xgb_overall = xgb_metrics.get("accuracy")
        rows.append(
            (
                f"| {spec.label} | {format_optional_float(knn_elig)} "
                f"({_format_feature_inline(space)}) | {format_optional_float(xgb_elig)} | "
                f"{format_optional_float(knn_all)} | {format_optional_float(xgb_overall)} |"
            )
        )
    if not rows:
        return []
    lines: List[str] = ["## KNN vs XGB (Matched Studies)", ""]
    lines.append(
        "This section compares the eligible-only accuracy for KNN and XGB, and also shows "
        "an all-rows accuracy for KNN alongside XGB's overall accuracy."
    )
    lines.append("")
    lines.append(header)
    lines.append(rule)
    lines.extend(rows)
    lines.append("")
    return lines


def _ordered_loso_feature_spaces(
    loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
) -> List[str]:
    """Return feature spaces ordered by canonical preference then presence."""

    ordered = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in loso_metrics
    ]
    ordered.extend(space for space in loso_metrics if space not in ordered)
    return ordered


def _collect_loso_summaries(
    metrics: Mapping[str, Mapping[str, object]],
    studies: Sequence[StudySpec],
) -> List[Tuple[str, MetricSummary, Optional[float]]]:
    """Return holdout summaries paired with their accuracy deltas."""

    summaries: List[Tuple[str, MetricSummary, Optional[float]]] = []
    for study in studies:
        payload = metrics.get(study.key)
        if not payload:
            continue
        summary = extract_metric_summary(payload)
        delta = (
            summary.accuracy - summary.baseline
            if summary.accuracy is not None and summary.baseline is not None
            else None
        )
        summaries.append((study.label, summary, delta))
    return summaries


def _loso_highlight_lines(
    summaries: Sequence[Tuple[str, MetricSummary, Optional[float]]]
) -> List[str]:
    """Return highlight bullets for key holdout statistics."""

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
    """Return Markdown table rows for leave-one-study-out metrics."""

    rows: List[str] = []
    for study in studies:
        payload = metrics.get(study.key)
        if not payload:
            continue
        summary = extract_metric_summary(payload)
        delta = (
            summary.accuracy - summary.baseline
            if summary.accuracy is not None and summary.baseline is not None
            else None
        )
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
    :type loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[StudySpec]
    :returns: Markdown section summarising leave-one-study-out accuracy.
    :rtype: List[str]
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


def _build_next_video_report(inputs: NextVideoReportInputs) -> None:
    """
    Compose the next-video evaluation report under ``inputs.output_dir``.

    :param inputs: Structured bundle of report inputs.
    :type inputs: NextVideoReportInputs
    """
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = inputs.output_dir / "README.md"

    try:
        dataset_name, split = _next_video_dataset_info(inputs.metrics_by_feature)
    except RuntimeError:
        if not inputs.allow_incomplete:
            raise
        placeholder = [
            "# KNN Next-Video Baseline",
            "",
            (
                "Next-video slate metrics are not available yet. "
                "Execute the finalize stage to refresh these results."
            ),
            "",
            "This placeholder was generated with `--allow-incomplete` enabled.",
            "",
        ]
        readme_path.write_text("\n".join(placeholder), encoding="utf-8")
        return
    uncertainty = _next_video_uncertainty_info(inputs.metrics_by_feature)

    lines: List[str] = _next_video_intro(dataset_name, split, uncertainty)
    lines.extend(
        _next_video_portfolio_summary(inputs.metrics_by_feature, inputs.feature_spaces)
    )

    for feature_space in inputs.feature_spaces:
        per_feature = inputs.metrics_by_feature.get(feature_space, {})
        lines.extend(_next_video_feature_section(feature_space, per_feature, inputs.studies))

    curve_sections = _next_video_curve_sections(
        output_dir=inputs.output_dir,
        metrics_by_feature=inputs.metrics_by_feature,
        studies=inputs.studies,
    )
    if curve_sections:
        lines.append("## Accuracy Curves")
        lines.append("")
        lines.extend(curve_sections)

    lines.extend(_next_video_observations(inputs.metrics_by_feature, inputs.studies))

    if inputs.xgb_next_video_dir is not None:
        compare = _knn_vs_xgb_section(
            xgb_next_video_dir=inputs.xgb_next_video_dir,
            metrics_by_feature=inputs.metrics_by_feature,
            studies=inputs.studies,
        )
        if compare:
            lines.extend(compare)

    if inputs.loso_metrics:
        lines.extend(_next_video_loso_section(inputs.loso_metrics, inputs.studies))
    elif inputs.allow_incomplete:
        lines.append(
            "Leave-one-study-out metrics were unavailable when this report was generated."
        )
        lines.append("")

    lines.append("")
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    # Emit CSV dumps for downstream analysis
    _write_next_video_metrics_csv(
        inputs.output_dir,
        inputs.metrics_by_feature,
        inputs.studies,
    )
    if inputs.loso_metrics:
        _write_next_video_loso_csv(
            inputs.output_dir,
            inputs.loso_metrics,
            inputs.studies,
        )


__all__ = ["NextVideoReportInputs", "_build_next_video_report"]


def _write_next_video_metrics_csv(
    output_dir: Path,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> None:
    """Write per-study next-video metrics to metrics.csv for all feature spaces."""

    if not metrics_by_feature:
        return
    fieldnames = [
        "feature_space",
        "study",
        "accuracy",
        "accuracy_all_rows",
        "baseline_accuracy",
        "random_baseline_accuracy",
        "best_k",
        "eligible",
        "total",
        "accuracy_ci_low",
        "accuracy_ci_high",
    ]
    with open(output_dir / "metrics.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature_space, per_feature in metrics_by_feature.items():
            for study_key, payload in per_feature.items():
                spec = next((s for s in studies if s.key == study_key), None)
                if spec is None:
                    continue
                summary = extract_metric_summary(payload)
                confidence_interval = summary.accuracy_ci
                writer.writerow(
                    {
                        "feature_space": feature_space,
                        "study": spec.label,
                        "accuracy": summary.accuracy,
                        "accuracy_all_rows": payload.get("accuracy_overall_all_rows"),
                        "baseline_accuracy": summary.baseline,
                        "random_baseline_accuracy": summary.random_baseline,
                        "best_k": summary.best_k,
                        "eligible": summary.n_eligible,
                        "total": summary.n_total,
                        "accuracy_ci_low": (
                            confidence_interval[0] if confidence_interval else None
                        ),
                        "accuracy_ci_high": (
                            confidence_interval[1] if confidence_interval else None
                        ),
                    }
                )


def _write_next_video_loso_csv(
    output_dir: Path,
    loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> None:
    """Write LOSO next-video metrics to loso_metrics.csv for all feature spaces."""

    if not loso_metrics:
        return
    out_path = output_dir / "loso_metrics.csv"
    fieldnames = [
        "feature_space",
        "holdout_study",
        "accuracy",
        "delta_vs_baseline",
        "baseline_accuracy",
        "best_k",
        "eligible",
    ]
    study_by_key = {spec.key: spec for spec in studies}
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature_space, per_feature in loso_metrics.items():
            for study_key, payload in per_feature.items():
                spec = study_by_key.get(study_key)
                if spec is None:
                    continue
                summary = extract_metric_summary(payload)
                delta = (
                    summary.accuracy - summary.baseline
                    if summary.accuracy is not None and summary.baseline is not None
                    else None
                )
                writer.writerow(
                    {
                        "feature_space": feature_space,
                        "holdout_study": spec.label,
                        "accuracy": summary.accuracy,
                        "delta_vs_baseline": delta,
                        "baseline_accuracy": summary.baseline,
                        "best_k": summary.best_k,
                        "eligible": summary.n_eligible,
                    }
                )
