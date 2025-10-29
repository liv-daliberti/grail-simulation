"""Helpers for comparing next-video KNN metrics against XGB baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from ...context import StudySpec
from ..shared import LOGGER
from ...utils import extract_metric_summary, format_optional_float

if TYPE_CHECKING:
    from ...context import MetricSummary
else:  # pragma: no cover - type hint fallback
    MetricSummary = object


def _format_feature_inline(space: str) -> str:
    """
    Return a compact inline label for a feature space.

    :param space: Feature-space identifier such as ``tfidf`` or ``sentence_transformer``.
    :returns: Abbreviated feature-space label.
    """
    if space == "sentence_transformer":
        return "ST"
    return space.upper()


def _load_xgb_metrics_for_study(
    base_dir: Path,
    evaluation_slug: str,
) -> Optional[Mapping[str, object]]:
    """
    Load XGB metrics for a single study when available.

    :param base_dir: Directory containing XGB evaluation exports.
    :param evaluation_slug: Identifier for the study evaluation run.
    :returns: Parsed metrics mapping, or ``None`` when missing or invalid.
    """
    metrics_path = base_dir / evaluation_slug / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
        LOGGER.debug("Failed to load XGB metrics for evaluation slug %s", evaluation_slug)
        return None


def _best_knn_summary_for_study(
    *,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    study_key: str,
) -> Optional[Tuple[str, "MetricSummary"]]:
    """
    Select the best KNN feature-space summary for ``study_key`` by eligible-only accuracy.

    :param metrics_by_feature: Nested mapping of metrics keyed by feature space and study.
    :param study_key: Study identifier used to extract metrics.
    :returns: Tuple of ``(feature_space, summary)`` when available, otherwise ``None``.
    """
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
) -> list[str]:
    """
    Build a comparison section for matched KNN/XGB studies when XGB metrics are present.

    :param xgb_next_video_dir: Directory containing XGB metrics grouped by evaluation slug.
    :param metrics_by_feature: Nested mapping of KNN metrics keyed by feature space and study.
    :param studies: Ordered study specifications to evaluate for matching metrics.
    :returns: Markdown lines describing the comparison table, or an empty list.
    """
    if not xgb_next_video_dir.exists():
        return []
    header = (
        "| Study | KNN (feature) eligible-only ↑ | XGB eligible-only ↑ | "
        "KNN all-rows ↑ | XGB overall ↑ |"
    )
    rule = "| --- | ---: | ---: | ---: | ---: |"
    rows: list[str] = []
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
    lines: list[str] = ["## KNN vs XGB (Matched Studies)", ""]
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


__all__ = [
    "_best_knn_summary_for_study",
    "_knn_vs_xgb_section",
    "_load_xgb_metrics_for_study",
]
