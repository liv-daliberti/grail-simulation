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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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

    totals: Dict[str, float] = field(
        default_factory=lambda: {
            "participants": 0.0,
            "mae": 0.0,
            "baseline_mae": 0.0,
            "accuracy": 0.0,
            "accuracy_baseline": 0.0,
        }
    )
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "accuracy": 0.0,
            "accuracy_baseline": 0.0,
        }
    )
    entries: Dict[str, List[Tuple[float, str]]] = field(
        default_factory=lambda: {
            "mae": [],
            "delta": [],
            "accuracy": [],
            "accuracy_delta": [],
        }
    )

    def record(self, summary: OpinionSummary, label: str) -> None:
        """Add a study summary to the aggregate."""

        totals = self.totals
        weights = self.weights
        entries = self.entries

        mae_value = summary.mae
        baseline_value = summary.baseline_mae
        delta_value = summary.mae_delta
        accuracy_value = summary.accuracy
        baseline_accuracy = summary.baseline_accuracy
        accuracy_delta = summary.accuracy_delta
        participants = float(summary.participants or 0)

        if mae_value is not None:
            entries["mae"].append((mae_value, label))
            if participants > 0:
                totals["participants"] += participants
                totals["mae"] += mae_value * participants
                if baseline_value is not None:
                    totals["baseline_mae"] += baseline_value * participants

        if delta_value is not None:
            entries["delta"].append((delta_value, label))

        if accuracy_value is not None:
            entries["accuracy"].append((accuracy_value, label))
        if accuracy_delta is not None:
            entries["accuracy_delta"].append((accuracy_delta, label))
        if participants > 0 and accuracy_value is not None:
            totals["accuracy"] += accuracy_value * participants
            weights["accuracy"] += participants
            if baseline_accuracy is not None:
                totals["accuracy_baseline"] += baseline_accuracy * participants
                weights["accuracy_baseline"] += participants

    def _weighted_summary(self) -> Dict[str, Optional[float]]:
        participants = self.totals["participants"]
        weighted_mae = None
        weighted_baseline = None
        weighted_delta = None
        if participants > 0:
            weighted_mae = self.totals["mae"] / participants
            if self.totals["baseline_mae"] > 0:
                weighted_baseline = self.totals["baseline_mae"] / participants
                weighted_delta = weighted_baseline - weighted_mae

        accuracy_weight = self.weights["accuracy"]
        weighted_accuracy = None
        if accuracy_weight > 0:
            weighted_accuracy = self.totals["accuracy"] / accuracy_weight

        baseline_weight = self.weights["accuracy_baseline"]
        weighted_accuracy_baseline = None
        weighted_accuracy_delta = None
        if baseline_weight > 0:
            weighted_accuracy_baseline = (
                self.totals["accuracy_baseline"] / baseline_weight
            )
        if (
            weighted_accuracy is not None
            and weighted_accuracy_baseline is not None
        ):
            weighted_accuracy_delta = (
                weighted_accuracy - weighted_accuracy_baseline
            )

        return {
            "participants": participants,
            "mae": weighted_mae,
            "baseline_mae": weighted_baseline,
            "mae_delta": weighted_delta,
            "accuracy": weighted_accuracy,
            "accuracy_baseline": weighted_accuracy_baseline,
            "accuracy_delta": weighted_accuracy_delta,
        }

    def _weighted_lines(
        self,
        summary_values: Mapping[str, Optional[float]],
        participant_count: int,
    ) -> List[str]:
        """Return Markdown bullets for weighted portfolio metrics."""

        lines: List[str] = []
        mae_value = summary_values["mae"]
        if mae_value is not None:
            lines.append(
                f"- Weighted MAE {format_optional_float(mae_value)} "
                f"across {format_count(participant_count)} participants."
            )

        baseline_value = summary_values["baseline_mae"]
        if baseline_value is not None:
            lines.append(
                f"- Weighted baseline MAE {format_optional_float(baseline_value)} "
                f"({format_delta(summary_values['mae_delta'])} vs. final)."
            )

        accuracy_value = summary_values["accuracy"]
        if accuracy_value is not None:
            lines.append(
                f"- Weighted directional accuracy {format_optional_float(accuracy_value)} "
                f"across {format_count(participant_count)} participants."
            )

        baseline_accuracy = summary_values["accuracy_baseline"]
        if baseline_accuracy is not None:
            lines.append(
                f"- Weighted baseline accuracy {format_optional_float(baseline_accuracy)} "
                f"({format_delta(summary_values['accuracy_delta'])} vs. final)."
            )

        return lines

    def _mae_lines(self) -> List[str]:
        """Return qualitative bullets derived from MAE statistics."""

        lines: List[str] = []
        delta_entries = self.entries["delta"]
        if delta_entries:
            best_delta, best_label = max(delta_entries, key=lambda item: item[0])
            lines.append(
                f"- Largest MAE reduction: {best_label} "
                f"({format_delta(best_delta)})."
            )

        mae_entries = self.entries["mae"]
        if len(mae_entries) > 1:
            best_mae, best_label = min(mae_entries, key=lambda item: item[0])
            worst_mae, worst_label = max(mae_entries, key=lambda item: item[0])
            lines.append(
                f"- Lowest MAE: {best_label} ({format_optional_float(best_mae)}); "
                f"Highest MAE: {worst_label} ({format_optional_float(worst_mae)})."
            )

        return lines

    def _accuracy_lines(self) -> List[str]:
        """Return qualitative bullets derived from accuracy statistics."""

        lines: List[str] = []
        accuracy_entries = self.entries["accuracy"]
        if accuracy_entries:
            best_acc, best_label = max(accuracy_entries, key=lambda item: item[0])
            lines.append(
                f"- Highest directional accuracy: {best_label} "
                f"({format_optional_float(best_acc)})."
            )
            if len(accuracy_entries) > 1:
                worst_acc, worst_label = min(accuracy_entries, key=lambda item: item[0])
                lines.append(
                    f"- Lowest directional accuracy: {worst_label} "
                    f"({format_optional_float(worst_acc)})."
                )

        delta_entries = self.entries["accuracy_delta"]
        if delta_entries:
            best_delta_acc, best_label = max(delta_entries, key=lambda item: item[0])
            lines.append(
                f"- Largest accuracy gain vs. baseline: {best_label} "
                f"({format_delta(best_delta_acc)})."
            )

        return lines

    def to_lines(self, heading: str = "### Portfolio Summary") -> List[str]:
        """Render aggregated statistics as Markdown."""

        if not self.entries["mae"]:
            return []

        summary_values = self._weighted_summary()
        participant_count = int(summary_values["participants"])
        lines: List[str] = [heading, ""]

        lines.extend(self._weighted_lines(summary_values, participant_count))
        lines.extend(self._mae_lines())
        lines.extend(self._accuracy_lines())
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
            "- Metrics: MAE / RMSE / R² / directional accuracy on the predicted "
            "post index, compared against a no-change baseline."
        ),
        "",
    ]


def _opinion_unweighted_lines(summaries: Sequence[OpinionSummary]) -> List[str]:
    """Return unweighted statistics across opinion summaries."""

    lines: List[str] = []

    def _range_text(values: Sequence[float]) -> str:
        return (
            f"{format_optional_float(min(values))} – "
            f"{format_optional_float(max(values))}"
        )

    mae_values = [summary.mae for summary in summaries if summary.mae is not None]
    if mae_values:
        mean_mae = sum(mae_values) / len(mae_values)
        stdev_mae = statistics.pstdev(mae_values) if len(mae_values) > 1 else 0.0
        lines.append(
            f"- Unweighted MAE {format_optional_float(mean_mae)} "
            f"(σ {format_optional_float(stdev_mae)}, range "
            f"{_range_text(mae_values)})."
        )

    delta_values = [summary.mae_delta for summary in summaries if summary.mae_delta is not None]
    if delta_values:
        mean_delta = sum(delta_values) / len(delta_values)
        stdev_delta = statistics.pstdev(delta_values) if len(delta_values) > 1 else 0.0
        lines.append(
            f"- MAE delta mean {format_optional_float(mean_delta)} "
            f"(σ {format_optional_float(stdev_delta)}, range "
            f"{_range_text(delta_values)})."
        )

    accuracy_values = [
        summary.accuracy for summary in summaries if summary.accuracy is not None
    ]
    if accuracy_values:
        mean_accuracy = sum(accuracy_values) / len(accuracy_values)
        stdev_accuracy = (
            statistics.pstdev(accuracy_values) if len(accuracy_values) > 1 else 0.0
        )
        lines.append(
            f"- Unweighted directional accuracy {format_optional_float(mean_accuracy)} "
            f"(σ {format_optional_float(stdev_accuracy)}, range "
            f"{_range_text(accuracy_values)})."
        )

    accuracy_delta_values = [
        summary.accuracy_delta
        for summary in summaries
        if summary.accuracy_delta is not None
    ]
    if accuracy_delta_values:
        mean_accuracy_delta = sum(accuracy_delta_values) / len(accuracy_delta_values)
        stdev_accuracy_delta = (
            statistics.pstdev(accuracy_delta_values)
            if len(accuracy_delta_values) > 1
            else 0.0
        )
        lines.append(
            f"- Accuracy delta mean {format_optional_float(mean_accuracy_delta)} "
            f"(σ {format_optional_float(stdev_accuracy_delta)}, range "
            f"{_range_text(accuracy_delta_values)})."
        )

    return lines


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
            "| Study | Participants | Best k | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | "
            "MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | Baseline MAE ↓ |"
        )
        divider = (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
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
        lines.append(
            f"*Assets:* [MAE / R² curves and heatmaps](../{feature_space}/opinion/)"
        )
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
        format_optional_float(summary.accuracy),
        format_optional_float(summary.baseline_accuracy),
        format_delta(summary.accuracy_delta),
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
            "Plots are refreshed under `reports/knn/<feature-space>/opinion/` "
            "including MAE vs. k (`mae_<study>.png`), R² vs. k (`r2_<study>.png`), "
            "and change heatmaps (`change_heatmap_<study>.png`)."
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
    lines.extend(_opinion_unweighted_lines(summaries))
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
