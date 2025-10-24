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

"""Opinion report builders for the modular KNN pipeline."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from ..pipeline_context import OpinionSummary, StudySpec
from ..pipeline_utils import (
    extract_opinion_summary,
    format_count,
    format_delta,
    format_k,
    format_optional_float,
)
from .shared import _feature_space_heading


@dataclass
class _OpinionPortfolioStats:
    """Aggregate opinion-regression metrics across feature spaces and studies."""

    total_weight: float = 0.0
    weighted_mae_sum: float = 0.0
    weighted_baseline_sum: float = 0.0
    mae_entries: List[Tuple[float, str]] = field(default_factory=list)
    delta_entries: List[Tuple[float, str]] = field(default_factory=list)

    def record(self, summary: OpinionSummary, label: str) -> None:
        """Add a study summary to the aggregate."""
        mae_value = summary.mae
        baseline_value = summary.baseline_mae
        delta_value = summary.mae_delta
        participants = float(summary.participants or 0)

        if mae_value is not None:
            self.mae_entries.append((mae_value, label))
            if participants > 0:
                self.total_weight += participants
                self.weighted_mae_sum += mae_value * participants
                if baseline_value is not None:
                    self.weighted_baseline_sum += baseline_value * participants

        if delta_value is not None:
            self.delta_entries.append((delta_value, label))

    def to_lines(self, heading: str = "### Portfolio Summary") -> List[str]:
        """Render aggregated statistics as Markdown."""
        if not self.mae_entries:
            return []

        lines: List[str] = [heading, ""]
        weighted_mae = None
        weighted_baseline = None
        weighted_delta = None
        if self.total_weight > 0:
            weighted_mae = self.weighted_mae_sum / self.total_weight
            if self.weighted_baseline_sum > 0:
                weighted_baseline = self.weighted_baseline_sum / self.total_weight
                weighted_delta = weighted_baseline - weighted_mae

        if weighted_mae is not None:
            lines.append(
                f"- Weighted MAE {format_optional_float(weighted_mae)} "
                f"across {format_count(int(self.total_weight))} participants."
            )
        if weighted_baseline is not None:
            lines.append(
                f"- Weighted baseline MAE {format_optional_float(weighted_baseline)} "
                f"({format_delta(weighted_delta)} vs. final)."
            )
        if self.delta_entries:
            best_delta, best_label = max(self.delta_entries, key=lambda item: item[0])
            lines.append(
                f"- Largest MAE reduction: {best_label} ({format_delta(best_delta)})."
            )
        if len(self.mae_entries) > 1:
            best_mae, best_label = min(self.mae_entries, key=lambda item: item[0])
            worst_mae, worst_label = max(self.mae_entries, key=lambda item: item[0])
            lines.append(
                f"- Lowest MAE: {best_label} ({format_optional_float(best_mae)}); "
                f"Highest MAE: {worst_label} ({format_optional_float(worst_mae)})."
            )

        lines.append("")
        return lines


def _opinion_report_intro(dataset_name: str, split: str) -> List[str]:
    """
    Return the introductory Markdown section for the opinion report.

    :param dataset_name: Human-readable label for the dataset being summarised.
    :type dataset_name: str
    :param split: Dataset split identifier such as ``train`` or ``validation``.
    :type split: str
    :returns: Markdown intro lines.
    :rtype: List[str]
    """
    return [
        "# KNN Opinion Shift Study",
        "",
        (
            "This study evaluates a second KNN baseline that predicts each "
            "participant's post-study opinion index."
        ),
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Split: {split}",
        (
            "- Metrics: MAE / RMSE / R² on the predicted post index, compared "
            "against a no-change baseline."
        ),
        "",
    ]


def _opinion_dataset_info(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Tuple[str, str]:
    """
    Extract dataset metadata from the opinion metrics bundle.

    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :returns: Dataset name and split derived from the metrics payload.
    :rtype: Tuple[str, str]
    """
    for per_feature in metrics.values():
        for study_metrics in per_feature.values():
            summary = extract_opinion_summary(study_metrics)
            return (
                str(summary.dataset or "data/cleaned_grail"),
                str(summary.split or "validation"),
            )
    return ("data/cleaned_grail", "validation")


def _ordered_feature_spaces(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
) -> List[str]:
    """Return feature spaces sorted by preferred order."""
    preferred_order = ["tfidf", "word2vec", "sentence_transformer"]
    ordered = [space for space in preferred_order if space in metrics]
    ordered.extend(space for space in metrics if space not in ordered)
    return ordered


def _opinion_feature_sections(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Render opinion metrics tables grouped by feature space.

    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[StudySpec]
    :returns: Markdown sections summarising opinion metrics per feature space.
    :rtype: List[str]
    """
    lines: List[str] = []
    for feature_space in _ordered_feature_spaces(metrics):
        per_feature = metrics.get(feature_space, {})
        if not per_feature:
            continue
        header = (
            "| Study | Participants | Best k | MAE ↓ | Δ vs baseline ↓ | "
            "RMSE ↓ | R² ↑ | MAE (change) ↓ | Baseline MAE ↓ |"
        )
        divider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        lines.extend(
            [
                _feature_space_heading(feature_space),
                "",
                header,
                divider,
            ]
        )
        for study in studies:
            data = per_feature.get(study.key)
            if not data:
                continue
            lines.append(_format_opinion_row(study, data))
        lines.append("")
    return lines


def _format_opinion_row(study: StudySpec, data: Mapping[str, object]) -> str:
    """
    Return a Markdown table row for opinion metrics.

    :param study: Study specification for the item currently being processed.
    :type study: StudySpec
    :param data: Raw metrics mapping produced by an evaluation stage.
    :type data: Mapping[str, object]
    :returns: Markdown table row for the given study.
    :rtype: str
    """
    summary = extract_opinion_summary(data)
    label = str(data.get("label", study.label))
    participants_text = format_count(summary.participants)
    columns = [
        label,
        participants_text,
        format_k(summary.best_k),
        format_optional_float(summary.mae),
        format_delta(summary.mae_delta),
        format_optional_float(summary.rmse),
        format_optional_float(summary.r2_score),
        format_optional_float(summary.mae_change),
        format_optional_float(summary.baseline_mae),
    ]
    return "| " + " | ".join(columns) + " |"


def _opinion_heatmap_section() -> List[str]:
    """Return the Markdown section referencing opinion heatmaps."""
    return [
        "### Opinion Change Heatmaps",
        "",
        (
            "Plots are refreshed under `reports/knn/opinion/<feature-space>/` "
            "for MAE, R², and change heatmaps."
        ),
        "",
    ]


def _opinion_takeaways(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Generate takeaway bullets comparing opinion performance.

    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[StudySpec]
    :returns: Markdown bullet list capturing opinion takeaways.
    :rtype: List[str]
    """
    lines: List[str] = ["## Takeaways", ""]
    for study in studies:
        collected = _collect_study_metrics(metrics, study)
        if not collected:
            continue
        label = _label_for_study(study, collected)
        details = _describe_study_takeaways(collected)
        if details:
            lines.append(f"- {label}: {details}.")
    lines.append("")
    return lines


def _collect_study_metrics(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    study: StudySpec,
) -> List[Tuple[str, OpinionSummary, Mapping[str, object]]]:
    """Gather opinion summaries per feature space for ``study``."""
    collected: List[Tuple[str, OpinionSummary, Mapping[str, object]]] = []
    for feature_space, per_feature in metrics.items():
        data = per_feature.get(study.key)
        if not data:
            continue
        collected.append((feature_space, extract_opinion_summary(data), data))
    return collected


def _label_for_study(
    study: StudySpec, collected: Sequence[Tuple[str, OpinionSummary, Mapping[str, object]]]
) -> str:
    """Return the display label for the study, preferring recorded labels."""
    for _feature_space, _summary, data in collected:
        label = data.get("label")
        if label:
            return str(label)
    return study.label


def _describe_study_takeaways(
    collected: Sequence[Tuple[str, OpinionSummary, Mapping[str, object]]]
) -> Optional[str]:
    """Return a textual summary of best metrics for the study."""
    r2_candidates = [
        (summary.r2_score, feature_space, summary.best_k)
        for feature_space, summary, _data in collected
        if summary.r2_score is not None
    ]
    delta_candidates = [
        (summary.mae_delta, feature_space)
        for feature_space, summary, _data in collected
        if summary.mae_delta is not None
    ]
    bullet_bits: List[str] = []
    if r2_candidates:
        best_r2_value, best_r2_space, best_r2_k = max(
            r2_candidates, key=lambda item: item[0]
        )
        bullet_bits.append(
            f"best R² {format_optional_float(best_r2_value)} "
            f"with {best_r2_space.upper()} (k={format_k(best_r2_k)})"
        )
    if delta_candidates:
        best_delta_value, best_delta_space = max(
            delta_candidates, key=lambda item: item[0]
        )
        bullet_bits.append(
            f"largest MAE reduction {format_delta(best_delta_value)} "
            f"via {best_delta_space.upper()}"
        )
    if not bullet_bits:
        return None
    return "; ".join(bullet_bits)


def _knn_opinion_cross_study_diagnostics(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Compute cross-study diagnostic statistics for KNN opinion runs."""

    if not metrics:
        return []

    lines: List[str] = ["## Cross-Study Diagnostics", ""]
    any_entries = False
    for feature_space in _ordered_feature_spaces(metrics):
        per_feature = metrics.get(feature_space, {})
        feature_lines = _cross_study_feature_lines(feature_space, per_feature, studies)
        if not feature_lines:
            continue
        any_entries = True
        lines.extend(feature_lines)
    if not any_entries:
        return []
    return lines


def _cross_study_feature_lines(
    feature_space: str,
    per_feature: Mapping[str, Mapping[str, object]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Return diagnostics lines for a single feature space."""
    portfolio = _OpinionPortfolioStats()
    summaries: List[OpinionSummary] = []
    for study in studies:
        payload = per_feature.get(study.key)
        if not payload:
            continue
        summary = extract_opinion_summary(payload)
        summaries.append(summary)
        portfolio.record(summary, f"{study.label} ({feature_space.upper()})")
    if not summaries:
        return []
    lines: List[str] = [_feature_space_heading(feature_space), ""]
    lines.extend(portfolio.to_lines("#### Weighted Summary"))

    maes = [summary.mae for summary in summaries if summary.mae is not None]
    if maes:
        mean_mae = sum(maes) / len(maes)
        stdev_mae = statistics.pstdev(maes) if len(maes) > 1 else 0.0
        lines.append(
            f"- Unweighted MAE {format_optional_float(mean_mae)} "
            f"(σ {format_optional_float(stdev_mae)}, range "
            f"{format_optional_float(min(maes))} – {format_optional_float(max(maes))})."
        )
    deltas = [summary.mae_delta for summary in summaries if summary.mae_delta is not None]
    if deltas:
        mean_delta = sum(deltas) / len(deltas)
        stdev_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        lines.append(
            f"- MAE delta mean {format_optional_float(mean_delta)} "
            f"(σ {format_optional_float(stdev_delta)}, range "
            f"{format_optional_float(min(deltas))} – {format_optional_float(max(deltas))})."
        )
    lines.append("")
    return lines


def _build_opinion_report(
    *,
    output_path: Path,
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
    allow_incomplete: bool = False,
) -> None:
    """
    Compose the opinion regression report at ``output_path``.

    :param output_path: Filesystem path for the generated report or figure.
    :type output_path: Path
    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[StudySpec]
    :param allow_incomplete: Whether processing may continue when some sweep data is missing.
    :type allow_incomplete: bool
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        if not allow_incomplete:
            raise RuntimeError("No opinion metrics available to build the opinion report.")
        placeholder = [
            "# KNN Opinion Shift Study",
            "",
            (
                "Opinion regression metrics are not available yet. "
                "Execute the finalize stage to refresh these results."
            ),
            "",
            "This placeholder was generated with `--allow-incomplete` enabled.",
            "",
        ]
        output_path.write_text("\n".join(placeholder), encoding="utf-8")
        return
    dataset_name, split = _opinion_dataset_info(metrics)
    lines: List[str] = []
    lines.extend(_opinion_report_intro(dataset_name, split))
    lines.extend(_opinion_feature_sections(metrics, studies))
    lines.extend(_opinion_heatmap_section())
    lines.extend(_knn_opinion_cross_study_diagnostics(metrics, studies))
    lines.extend(_opinion_takeaways(metrics, studies))
    output_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["_OpinionPortfolioStats", "_build_opinion_report"]
