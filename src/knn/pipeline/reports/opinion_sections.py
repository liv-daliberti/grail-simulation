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

"""Markdown section builders for the opinion pipeline report."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from common.reports.utils import append_image_section

from ..context import OpinionSummary, StudySpec
from ..utils import (
    extract_opinion_summary,
    format_count,
    format_delta,
    format_k,
    format_optional_float,
)
from .opinion_portfolio import _ordered_feature_spaces
from .shared import _feature_space_heading

__all__ = [
    "_opinion_report_intro",
    "_opinion_dataset_info",
    "_opinion_feature_sections",
    "_opinion_heatmap_section",
    "_opinion_takeaways",
]


def _opinion_report_intro(
    dataset_name: str,
    split: str,
    *,
    title: str,
    description_lines: Sequence[str],
    metrics_line: str,
) -> List[str]:
    """
    Build the introductory Markdown section for the opinion report.

    :param dataset_name: Display name for the dataset under evaluation.
    :type dataset_name: str
    :param split: Dataset split identifier (e.g., ``validation``).
    :type split: str
    :param title: Heading used at the start of the report.
    :type title: str
    :param description_lines: Additional Markdown lines describing the study.
    :type description_lines: Sequence[str]
    :param metrics_line: Summary bullet describing the reported metrics.
    :type metrics_line: str
    :returns: Markdown lines forming the introductory section.
    :rtype: List[str]
    """

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


def _opinion_feature_sections(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Render opinion metrics tables grouped by feature space.

    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[~knn.pipeline.context.StudySpec]
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
    :type study: ~knn.pipeline.context.StudySpec
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


def _opinion_heatmap_section(
    output_path: Path,
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> List[str]:
    """
    Build the heatmap section for the opinion report.

    :param output_path: Final Markdown path used to resolve relative asset links.
    :type output_path: Path
    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :returns: Markdown lines referencing available heatmap images.
    :rtype: List[str]
    """

    base_dir = output_path.parent.parent
    sections: List[str] = ["### Opinion Change Heatmaps", ""]
    found = False
    for feature_space in sorted(metrics.keys()):
        feature_dir = base_dir / feature_space / "opinion"
        if not feature_dir.exists():
            continue
        images = sorted(feature_dir.glob("*.png"))
        if not images:
            continue
        found = True
        sections.append(f"#### {feature_space.upper()}")
        sections.append("")
        for image in images:
            append_image_section(
                sections,
                image=image,
                relative_root=output_path.parent,
            )
    if not found:
        sections.extend(
            [
                (
                    "Plots are refreshed under `reports/knn/<feature-space>/opinion/` "
                    "including MAE vs. k (`mae_<study>.png`), R² vs. k (`r2_<study>.png`), "
                    "and change heatmaps (`change_heatmap_<study>.png`)."
                ),
                "",
            ]
        )
    return sections


def _opinion_takeaways(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Generate takeaway bullets comparing opinion performance.

    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[~knn.pipeline.context.StudySpec]
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
    """
    Gather opinion summaries per feature space for ``study``.

    :param metrics: Metrics dictionary captured from a previous pipeline stage.
    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param study: Study specification to retrieve metrics for.
    :type study: ~knn.pipeline.context.StudySpec
    :returns: List of tuples containing feature space, summary, and raw payload.
    :rtype: List[Tuple[str, OpinionSummary, Mapping[str, object]]]
    """
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
    """
    Select the display label for the study.

    :param study: Default study specification.
    :type study: ~knn.pipeline.context.StudySpec
    :param collected: Feature-space tuples gathered via :func:`_collect_study_metrics`.
    :type collected: Sequence[Tuple[str, OpinionSummary, Mapping[str, object]]]
    :returns: Study label sourced from metrics payloads when available.
    :rtype: str
    """
    for _feature_space, _summary, data in collected:
        label = data.get("label")
        if label:
            return str(label)
    return study.label


def _describe_study_takeaways(
    collected: Sequence[Tuple[str, OpinionSummary, Mapping[str, object]]]
) -> Optional[str]:
    """
    Summarise headline metrics for a study across feature spaces.

    :param collected: Feature-space tuples gathered via :func:`_collect_study_metrics`.
    :type collected: Sequence[Tuple[str, OpinionSummary, Mapping[str, object]]]
    :returns: Summary text describing the best-performing feature spaces, or ``None``
        if no metrics exist.
    :rtype: Optional[str]
    """
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
