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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from common.report_utils import extract_numeric_series

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt  # type: ignore[assignment]
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]

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


@dataclass
class PortfolioAggregate:
    """Weighted portfolio metrics for a feature space."""

    accuracy: Optional[float]
    baseline: Optional[float]
    random: Optional[float]
    eligible: int
    studies: int


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
    totals = {"accuracy": 0.0, "baseline": 0.0, "random": 0.0}
    weights = {"accuracy": 0, "baseline": 0, "random": 0}
    studies_with_metrics = 0
    for data in metrics.values():
        summary = extract_metric_summary(data)
        eligible = summary.n_eligible
        if not eligible:
            continue
        recorded = False
        for field_name, attr in (
            ("accuracy", "accuracy"),
            ("baseline", "baseline"),
            ("random", "random_baseline"),
        ):
            value = getattr(summary, attr)
            if value is None:
                continue
            totals[field_name] += value * eligible
            weights[field_name] += eligible
            recorded = True
        if recorded:
            studies_with_metrics += 1
    if not studies_with_metrics:
        return None
    weighted = {
        field: (totals[field] / weights[field] if weights[field] else None)
        for field in totals
    }
    eligible_total = max(weights.values()) if weights else 0
    return PortfolioAggregate(
        accuracy=weighted["accuracy"],
        baseline=weighted["baseline"],
        random=weighted["random"],
        eligible=eligible_total,
        studies=studies_with_metrics,
    )


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
        "- Metric: accuracy on eligible slates (gold index present).",
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
    best: Optional[Tuple[str, PortfolioAggregate]] = None

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
            "| {space} | {accuracy} | {delta} | {random} | {eligible} | {studies} |".format(
                space=feature_space.upper(),
                accuracy=format_optional_float(aggregates.accuracy),
                delta=format_delta(delta_value),
                random=format_optional_float(aggregates.random),
                eligible=format_count(aggregates.eligible),
                studies=format_count(aggregates.studies),
            )
        )
        if aggregates.accuracy is not None:
            if best is None or aggregates.accuracy > best[1].accuracy:
                best = (feature_space, aggregates)

    if not rows:
        return []

    lines: List[str] = ["## Portfolio Summary", ""]
    lines.append(
        "| Feature space | Weighted accuracy ↑ | Δ vs baseline ↑ | Random ↑ | Eligible | "
        "Studies |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.extend(rows)
    lines.append("")
    if best:
        best_space, best_metrics = best
        lines.append(
            "Best-performing feature space: **{space}** with weighted accuracy "
            "{accuracy} across {eligible} eligible slates ({studies} studies).".format(
                space=best_space.upper(),
                accuracy=format_optional_float(best_metrics.accuracy),
                eligible=format_count(best_metrics.eligible),
                studies=format_count(best_metrics.studies),
            )
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
    curve_bundle = metrics.get("curve_metrics")
    if not isinstance(curve_bundle, Mapping):
        return None
    eval_curve = curve_bundle.get("eval")
    if not isinstance(eval_curve, Mapping):
        return None
    eval_x, eval_y = _extract_curve_series(eval_curve)
    if not eval_x:
        return None
    train_curve = curve_bundle.get("train")
    train_x: List[int] = []
    train_y: List[float] = []
    if isinstance(train_curve, Mapping):
        train_x, train_y = _extract_curve_series(train_curve)

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
        "| Study | Accuracy ↑ | 95% CI | Δ vs baseline ↑ | Baseline ↑ | Random ↑ | Best k | Eligible | Total |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
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
    ordered_spaces = [
        space for space in ("tfidf", "word2vec", "sentence_transformer") if space in metrics_by_feature
    ]
    for feature_space in metrics_by_feature:
        if feature_space not in ordered_spaces:
            ordered_spaces.append(feature_space)

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
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in metrics_by_feature
    ]
    for space in metrics_by_feature:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
        metrics = metrics_by_feature.get(feature_space, {})
        if not metrics:
            continue
        bullet_bits: List[str] = []
        deltas: List[float] = []
        randoms: List[float] = []
        for study in studies:
            data = metrics.get(study.key)
            if not data:
                continue
            summary = extract_metric_summary(data)
            if summary.accuracy is None:
                continue
            delta_val = (
                summary.accuracy - summary.baseline
                if summary.baseline is not None
                else None
            )
            if delta_val is not None:
                deltas.append(delta_val)
            if summary.random_baseline is not None:
                randoms.append(summary.random_baseline)
            detail = (
                f"{study.label}: {format_optional_float(summary.accuracy)} "
                f"(baseline {format_optional_float(summary.baseline)}, "
                f"Δ {format_delta(delta_val)}, k={format_k(summary.best_k)}, "
                f"eligible {format_count(summary.n_eligible)})"
            )
            bullet_bits.append(detail)
        extras: List[str] = []
        if deltas:
            mean_delta = sum(deltas) / len(deltas)
            extras.append(f"mean Δ {format_delta(mean_delta)}")
        if randoms:
            mean_random = sum(randoms) / len(randoms)
            extras.append(f"mean random {format_optional_float(mean_random)}")
        if extras:
            bullet_bits.append("averages: " + ", ".join(extras))
        if bullet_bits:
            lines.append(f"- {feature_space.upper()}: " + "; ".join(bullet_bits) + ".")
    lines.append("- Random values correspond to the expected accuracy from a uniform guess across the slate options.")
    lines.append("")
    return lines


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
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in loso_metrics
    ]
    for space in loso_metrics:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
        metrics = loso_metrics.get(feature_space, {})
        if not metrics:
            continue
        lines.append(_feature_space_heading(feature_space))
        lines.append("")
        lines.append("| Holdout study | Accuracy ↑ | Δ vs baseline ↑ | Baseline ↑ | Best k | Eligible |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
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
                f"{format_delta(delta)} | {format_optional_float(summary.baseline)} | "
                f"{format_k(summary.best_k)} | {format_count(summary.n_eligible)} |"
            )
        lines.append("")
    return lines


def _build_next_video_report(
    *,
    output_dir: Path,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
    feature_spaces: Sequence[str],
    loso_metrics: Optional[Mapping[str, Mapping[str, Mapping[str, object]]]] = None,
    allow_incomplete: bool = False,
) -> None:
    """
    Compose the next-video evaluation report under ``output_dir``.

    :param output_dir: Directory where the rendered report should be written.
    :type output_dir: Path
    :param metrics_by_feature: Nested mapping of metrics grouped by feature space and study.
    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[StudySpec]
    :param feature_spaces: Ordered feature spaces considered by the pipeline.
    :type feature_spaces: Sequence[str]
    :param loso_metrics: Optional leave-one-study-out evaluation bundle.
    :type loso_metrics: Optional[Mapping[str, Mapping[str, Mapping[str, object]]]]
    :param allow_incomplete: Whether missing metrics should surface placeholders.
    :type allow_incomplete: bool
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / "README.md"

    dataset_name, split = _next_video_dataset_info(metrics_by_feature)
    uncertainty = _next_video_uncertainty_info(metrics_by_feature)

    lines: List[str] = _next_video_intro(dataset_name, split, uncertainty)
    lines.extend(_next_video_portfolio_summary(metrics_by_feature, feature_spaces))

    for feature_space in feature_spaces:
        per_feature = metrics_by_feature.get(feature_space, {})
        lines.extend(_next_video_feature_section(feature_space, per_feature, studies))

    curve_sections = _next_video_curve_sections(
        output_dir=output_dir,
        metrics_by_feature=metrics_by_feature,
        studies=studies,
    )
    if curve_sections:
        lines.append("## Accuracy Curves")
        lines.append("")
        lines.extend(curve_sections)

    lines.extend(_next_video_observations(metrics_by_feature, studies))

    if loso_metrics:
        lines.extend(_next_video_loso_section(loso_metrics, studies))
    elif allow_incomplete:
        lines.append(
            "Leave-one-study-out metrics were unavailable when this report was generated."
        )
        lines.append("")

    lines.append("")
    readme_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["_build_next_video_report"]
