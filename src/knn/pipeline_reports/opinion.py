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
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from common.opinion_metrics import summarise_opinion_metrics

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
            "rmse_change": 0.0,
            "baseline_rmse_change": 0.0,
            "calibration_ece": 0.0,
            "baseline_calibration_ece": 0.0,
            "kl_divergence_change": 0.0,
            "baseline_kl_divergence_change": 0.0,
        }
    )
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "baseline_mae": 0.0,
            "accuracy": 0.0,
            "accuracy_baseline": 0.0,
            "rmse_change": 0.0,
            "baseline_rmse_change": 0.0,
            "calibration_ece": 0.0,
            "baseline_calibration_ece": 0.0,
            "kl_divergence_change": 0.0,
            "baseline_kl_divergence_change": 0.0,
        }
    )
    entries: Dict[str, List[Tuple[float, str]]] = field(
        default_factory=lambda: {
            "mae": [],
            "delta": [],
            "accuracy": [],
            "accuracy_delta": [],
            "rmse_change": [],
            "rmse_change_delta": [],
            "calibration_ece": [],
            "calibration_ece_delta": [],
            "kl_divergence_change": [],
            "kl_divergence_delta": [],
        }
    )

    def record(self, summary: OpinionSummary, label: str) -> None:
        """Add a study summary to the aggregate."""

        metrics = summarise_opinion_metrics(summary, prefer_after_fields=False)
        self._record_mae(metrics, label)
        self._record_accuracy(metrics, label)
        self._record_rmse(metrics, label)
        self._record_calibration(metrics, label)
        self._record_kl_divergence(metrics, label)

    def _record_mae(self, metrics, label: str) -> None:
        mae_value = metrics.mae
        if mae_value is None:
            return
        participants = metrics.participants
        self.entries["mae"].append((mae_value, label))
        delta_value = metrics.mae_delta
        if delta_value is not None:
            self.entries["delta"].append((delta_value, label))
        if participants <= 0:
            return
        self.totals["participants"] += participants
        self.totals["mae"] += mae_value * participants
        baseline_value = metrics.baseline_mae
        if baseline_value is not None:
            self.totals["baseline_mae"] += baseline_value * participants
            self.weights["baseline_mae"] += participants

    def _record_accuracy(self, metrics, label: str) -> None:
        accuracy_value = metrics.accuracy
        if accuracy_value is None:
            return
        self.entries["accuracy"].append((accuracy_value, label))
        accuracy_delta = metrics.accuracy_delta
        if accuracy_delta is not None:
            self.entries["accuracy_delta"].append((accuracy_delta, label))
        participants = metrics.participants
        if participants <= 0:
            return
        self.totals["accuracy"] += accuracy_value * participants
        self.weights["accuracy"] += participants
        baseline_accuracy = metrics.baseline_accuracy
        if baseline_accuracy is not None:
            self.totals["accuracy_baseline"] += baseline_accuracy * participants
            self.weights["accuracy_baseline"] += participants

    def _record_rmse(self, metrics, label: str) -> None:
        rmse_change = metrics.rmse_change
        if rmse_change is None:
            return
        self.entries["rmse_change"].append((rmse_change, label))
        baseline_rmse_change = metrics.baseline_rmse_change
        if baseline_rmse_change is not None:
            self.entries["rmse_change_delta"].append(
                (baseline_rmse_change - rmse_change, label)
            )
        participants = metrics.participants
        if participants <= 0:
            return
        self.totals["rmse_change"] += rmse_change * participants
        self.weights["rmse_change"] += participants
        if baseline_rmse_change is not None:
            self.totals["baseline_rmse_change"] += baseline_rmse_change * participants
            self.weights["baseline_rmse_change"] += participants

    def _record_calibration(self, metrics, label: str) -> None:
        calibration_ece = metrics.calibration_ece
        if calibration_ece is None:
            return
        self.entries["calibration_ece"].append((calibration_ece, label))
        baseline_calibration_ece = metrics.baseline_calibration_ece
        if baseline_calibration_ece is not None:
            self.entries["calibration_ece_delta"].append(
                (baseline_calibration_ece - calibration_ece, label)
            )
        participants = metrics.participants
        if participants <= 0:
            return
        self.totals["calibration_ece"] += calibration_ece * participants
        self.weights["calibration_ece"] += participants
        if baseline_calibration_ece is not None:
            self.totals["baseline_calibration_ece"] += (
                baseline_calibration_ece * participants
            )
            self.weights["baseline_calibration_ece"] += participants

    def _record_kl_divergence(self, metrics, label: str) -> None:
        kl_divergence = metrics.kl_divergence_change
        if kl_divergence is None:
            return
        self.entries["kl_divergence_change"].append((kl_divergence, label))
        baseline_kl_divergence = metrics.baseline_kl_divergence_change
        if baseline_kl_divergence is not None:
            self.entries["kl_divergence_delta"].append(
                (baseline_kl_divergence - kl_divergence, label)
            )
        participants = metrics.participants
        if participants <= 0:
            return
        self.totals["kl_divergence_change"] += kl_divergence * participants
        self.weights["kl_divergence_change"] += participants
        if baseline_kl_divergence is not None:
            self.totals["baseline_kl_divergence_change"] += (
                baseline_kl_divergence * participants
            )
            self.weights["baseline_kl_divergence_change"] += participants

    def _weighted_mean(self, total_key: str, weight: float) -> Optional[float]:
        if weight <= 0:
            return None
        return self.totals[total_key] / weight

    @staticmethod
    def _baseline_improvement(
        baseline_value: Optional[float], final_value: Optional[float]
    ) -> Optional[float]:
        if baseline_value is None or final_value is None:
            return None
        return baseline_value - final_value

    @staticmethod
    def _delta(final_value: Optional[float], baseline_value: Optional[float]) -> Optional[float]:
        if final_value is None or baseline_value is None:
            return None
        return final_value - baseline_value

    def _append_extreme_line(
        self,
        lines: List[str],
        entries: Sequence[Tuple[float, str]],
        formatter: Callable[[str, float], str],
        *,
        chooser=max,
    ) -> None:
        if not entries:
            return
        value, label = chooser(entries, key=lambda item: item[0])
        lines.append(formatter(label, value))

    def _append_min_max_line(
        self,
        lines: List[str],
        entries: Sequence[Tuple[float, str]],
        formatter: Callable[[str, float, str, float], str],
    ) -> None:
        if len(entries) < 2:
            return
        best_value, best_label = min(entries, key=lambda item: item[0])
        worst_value, worst_label = max(entries, key=lambda item: item[0])
        lines.append(formatter(best_label, best_value, worst_label, worst_value))

    def _weighted_summary(self) -> Dict[str, Optional[float]]:
        participants = self.totals["participants"]
        mae = self._weighted_mean("mae", participants)
        baseline_mae = self._weighted_mean(
            "baseline_mae", self.weights["baseline_mae"]
        )
        accuracy = self._weighted_mean("accuracy", self.weights["accuracy"])
        baseline_accuracy = self._weighted_mean(
            "accuracy_baseline", self.weights["accuracy_baseline"]
        )
        rmse_change = self._weighted_mean(
            "rmse_change", self.weights["rmse_change"]
        )
        baseline_rmse_change = self._weighted_mean(
            "baseline_rmse_change", self.weights["baseline_rmse_change"]
        )
        calibration_ece = self._weighted_mean(
            "calibration_ece", self.weights["calibration_ece"]
        )
        baseline_calibration_ece = self._weighted_mean(
            "baseline_calibration_ece", self.weights["baseline_calibration_ece"]
        )
        kl_divergence = self._weighted_mean(
            "kl_divergence_change", self.weights["kl_divergence_change"]
        )
        baseline_kl_divergence = self._weighted_mean(
            "baseline_kl_divergence_change",
            self.weights["baseline_kl_divergence_change"],
        )

        return {
            "participants": participants,
            "mae": mae,
            "baseline_mae": baseline_mae,
            "mae_delta": self._baseline_improvement(baseline_mae, mae),
            "accuracy": accuracy,
            "accuracy_baseline": baseline_accuracy,
            "accuracy_delta": self._delta(accuracy, baseline_accuracy),
            "rmse_change": rmse_change,
            "rmse_change_baseline": baseline_rmse_change,
            "rmse_change_delta": self._baseline_improvement(
                baseline_rmse_change, rmse_change
            ),
            "calibration_ece": calibration_ece,
            "calibration_ece_baseline": baseline_calibration_ece,
            "calibration_ece_delta": self._baseline_improvement(
                baseline_calibration_ece, calibration_ece
            ),
            "kl_divergence_change": kl_divergence,
            "kl_divergence_baseline": baseline_kl_divergence,
            "kl_divergence_delta": self._baseline_improvement(
                baseline_kl_divergence, kl_divergence
            ),
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

        rmse_change = summary_values.get("rmse_change")
        if rmse_change is not None:
            lines.append(
                f"- Weighted RMSE (change) {format_optional_float(rmse_change)} "
                f"({format_delta(summary_values.get('rmse_change_delta'))} vs. baseline)."
            )
        baseline_rmse_change = summary_values.get("rmse_change_baseline")
        if baseline_rmse_change is not None and rmse_change is None:
            # Avoid duplicate messaging when the primary value is missing but baseline exists.
            lines.append(
                f"- Weighted baseline RMSE (change) {format_optional_float(baseline_rmse_change)}."
            )

        ece_value = summary_values.get("calibration_ece")
        if ece_value is not None:
            lines.append(
                f"- Weighted calibration ECE {format_optional_float(ece_value)} "
                f"({format_delta(summary_values.get('calibration_ece_delta'))} vs. baseline)."
            )
        kl_value = summary_values.get("kl_divergence_change")
        if kl_value is not None:
            lines.append(
                f"- Weighted KL divergence {format_optional_float(kl_value)} "
                f"({format_delta(summary_values.get('kl_divergence_delta'))} vs. baseline)."
            )

        return lines

    def _mae_lines(self) -> List[str]:
        """Return qualitative bullets derived from MAE statistics."""

        lines: List[str] = []
        self._append_extreme_line(
            lines,
            self.entries["delta"],
            lambda label, value: f"- Largest MAE reduction: {label} ({format_delta(value)}).",
        )
        self._append_extreme_line(
            lines,
            self.entries["rmse_change_delta"],
            lambda label, value: (
                "- Largest RMSE(change) reduction: "
                f"{label} ({format_delta(value)})."
            ),
        )
        self._append_min_max_line(
            lines,
            self.entries["mae"],
            lambda best_label, best_value, worst_label, worst_value: (
                f"- Lowest MAE: {best_label} ({format_optional_float(best_value)}); "
                f"Highest MAE: {worst_label} ({format_optional_float(worst_value)})."
            ),
        )
        self._append_extreme_line(
            lines,
            self.entries["calibration_ece_delta"],
            lambda label, value: (
                "- Largest calibration ECE drop: "
                f"{label} ({format_delta(value)})."
            ),
        )
        self._append_extreme_line(
            lines,
            self.entries["kl_divergence_delta"],
            lambda label, value: (
                "- Biggest KL divergence reduction: "
                f"{label} ({format_delta(value)})."
            ),
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


@dataclass(frozen=True)
class OpinionReportOptions:
    """Bundle of optional parameters for building the opinion report."""

    allow_incomplete: bool = False
    title: str = "# KNN Opinion Shift Study"
    description_lines: Optional[Sequence[str]] = None
    metrics_line: Optional[str] = None


def _opinion_report_intro(
    dataset_name: str,
    split: str,
    *,
    title: str,
    description_lines: Sequence[str],
    metrics_line: str,
) -> List[str]:
    """Return the introductory Markdown section for the opinion report."""

    lines: List[str] = [title, ""]
    if description_lines:
        lines.extend(description_lines)
        if description_lines[-1].strip():
            lines.append("")
    lines.extend(
        [
            f"- Dataset: `{dataset_name}`",
            f"- Split: {split}",
            metrics_line,
            "",
        ]
    )
    return lines

def _unweighted_metric_line(
    summaries: Sequence[OpinionSummary],
    extractor: Callable[[OpinionSummary], Optional[float]],
    prefix: str,
) -> Optional[str]:
    values = [
        value
        for value in (extractor(summary) for summary in summaries)
        if value is not None
    ]
    if not values:
        return None
    mean_value = sum(values) / len(values)
    stdev_value = statistics.pstdev(values) if len(values) > 1 else 0.0
    range_text = (
        f"{format_optional_float(min(values))} – "
        f"{format_optional_float(max(values))}"
    )
    return (
        f"- {prefix} {format_optional_float(mean_value)} "
        f"(σ {format_optional_float(stdev_value)}, range {range_text})."
    )


def _opinion_unweighted_lines(summaries: Sequence[OpinionSummary]) -> List[str]:
    """Return unweighted statistics across opinion summaries."""

    lines: List[str] = []
    specs: Sequence[Tuple[str, Callable[[OpinionSummary], Optional[float]]]] = [
        ("Unweighted MAE", lambda summary: summary.mae),
        ("MAE delta mean", lambda summary: summary.mae_delta),
        (
            "Unweighted directional accuracy",
            lambda summary: summary.accuracy,
        ),
        ("Accuracy delta mean", lambda summary: summary.accuracy_delta),
        ("RMSE (change)", lambda summary: summary.rmse_change),
        (
            "RMSE (change) delta",
            lambda summary: (
                summary.baseline_rmse_change - summary.rmse_change
                if summary.rmse_change is not None
                and summary.baseline_rmse_change is not None
                else None
            ),
        ),
        (
            "Calibration ECE",
            lambda summary: summary.calibration_ece,
        ),
        (
            "Calibration ECE delta",
            lambda summary: (
                summary.baseline_calibration_ece - summary.calibration_ece
                if summary.calibration_ece is not None
                and summary.baseline_calibration_ece is not None
                else None
            ),
        ),
        (
            "KL divergence",
            lambda summary: summary.kl_divergence_change,
        ),
        (
            "KL divergence delta",
            lambda summary: (
                summary.baseline_kl_divergence_change - summary.kl_divergence_change
                if summary.kl_divergence_change is not None
                and summary.baseline_kl_divergence_change is not None
                else None
            ),
        ),
    ]
    for prefix, extractor in specs:
        line = _unweighted_metric_line(summaries, extractor, prefix)
        if line:
            lines.append(line)

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
            "MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | RMSE (change) ↓ | "
            "Δ RMSE (change) ↓ | Calib slope | Calib intercept | ECE ↓ | Δ ECE ↓ | KL div ↓ | "
            "Δ KL ↓ | Baseline MAE ↓ |"
        )
        divider = (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
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
    rmse_change_delta = (
        summary.baseline_rmse_change - summary.rmse_change
        if summary.rmse_change is not None and summary.baseline_rmse_change is not None
        else None
    )
    calibration_ece_delta = (
        summary.baseline_calibration_ece - summary.calibration_ece
        if summary.calibration_ece is not None and summary.baseline_calibration_ece is not None
        else None
    )
    kl_divergence_delta = (
        summary.baseline_kl_divergence_change - summary.kl_divergence_change
        if summary.kl_divergence_change is not None
        and summary.baseline_kl_divergence_change is not None
        else None
    )
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
        format_optional_float(summary.rmse_change),
        format_delta(rmse_change_delta),
        format_optional_float(summary.calibration_slope),
        format_optional_float(summary.calibration_intercept),
        format_optional_float(summary.calibration_ece),
        format_delta(calibration_ece_delta),
        format_optional_float(summary.kl_divergence_change),
        format_delta(kl_divergence_delta),
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
    options: OpinionReportOptions | None = None,
) -> None:
    """
    Compose the opinion regression report at ``output_path``.

    :param output_path: Filesystem path for the generated report or figure.
    :type output_path: Path
    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[StudySpec]
    :param options: Optional bundle controlling report presentation.
    :type options: OpinionReportOptions | None
    """
    options = options or OpinionReportOptions()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        if not options.allow_incomplete:
            raise RuntimeError("No opinion metrics available to build the opinion report.")
        placeholder = [
            options.title,
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
    intro_lines = options.description_lines
    if intro_lines is None:
        intro_lines = [
            (
                "This study evaluates a second KNN baseline that predicts each "
                "participant's post-study opinion index."
            )
        ]
    metrics_text = options.metrics_line or (
        "- Metrics: MAE / RMSE / R² / directional accuracy / MAE (change) / "
        "RMSE (change) / calibration slope & intercept / calibration ECE / "
        "KL divergence, compared against a no-change baseline."
    )
    lines: List[str] = []
    lines.extend(
        _opinion_report_intro(
            dataset_name,
            split,
            title=options.title,
            description_lines=intro_lines,
            metrics_line=metrics_text,
        )
    )
    lines.extend(_opinion_feature_sections(metrics, studies))
    lines.extend(_opinion_heatmap_section())
    lines.extend(_knn_opinion_cross_study_diagnostics(metrics, studies))
    lines.extend(_opinion_takeaways(metrics, studies))
    output_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["_OpinionPortfolioStats", "OpinionReportOptions", "_build_opinion_report"]
