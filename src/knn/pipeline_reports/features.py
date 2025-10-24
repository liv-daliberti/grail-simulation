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

"""Report builders summarising the extra text fields used by KNN pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from common.prompt_docs import DEFAULT_EXTRA_TEXT_FIELDS
from common.pipeline_io import write_markdown_lines
from common.report_utils import start_markdown_report

from ..pipeline_context import OpinionSweepOutcome, ReportBundle, SweepOutcome

_DEFAULT_FIELD_SET = tuple(DEFAULT_EXTRA_TEXT_FIELDS)


def _normalise_fields(values: Iterable[object] | None) -> tuple[str, ...]:
    """Canonicalise the incoming sequence of text fields."""
    if values is None:
        return _DEFAULT_FIELD_SET
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw or "").strip()
        if not token or token in seen:
            continue
        ordered.append(token)
        seen.add(token)
    return tuple(ordered) if ordered else _DEFAULT_FIELD_SET


def _extract_fields_from_metrics(metrics: Mapping[str, object]) -> tuple[str, ...]:
    """Extract text-field metadata from the stored metrics payload."""
    candidates: list[Iterable[object] | None] = []
    payload = metrics.get("knn_hparams")
    if isinstance(payload, Mapping):
        candidates.append(payload.get("text_fields"))
    candidates.append(metrics.get("extra_fields"))
    for candidate in candidates:
        fields = _normalise_fields(candidate) if candidate is not None else ()
        if fields:
            return fields
    return _DEFAULT_FIELD_SET


def _format_field_list(fields: Sequence[str]) -> str:
    """Render a comma-separated field list with inline code formatting."""
    if not fields:
        return "—"
    return ", ".join(f"`{field}`" for field in fields)


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a GitHub-flavoured Markdown table."""
    if not rows:
        return ["No entries recorded.", ""]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _collect_next_video_sweeps(outcomes: Sequence[SweepOutcome]) -> tuple[list[Sequence[str]], set[str]]:
    """Return table rows and the unique field set for KNN next-video sweeps."""
    rows: list[Sequence[str]] = []
    unique: set[str] = set()
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for outcome in outcomes:
        fields = _normalise_fields(outcome.config.text_fields)
        unique.update(fields)
        key = (outcome.feature_space, outcome.study.label, outcome.config.label(), fields)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            (
                outcome.feature_space,
                outcome.study.label,
                outcome.config.label(),
                _format_field_list(fields),
            )
        )
    return rows, unique


def _collect_next_video_final(bundle: ReportBundle) -> tuple[list[Sequence[str]], set[str]]:
    """Return table rows and unique fields for final KNN evaluations."""
    rows: list[Sequence[str]] = []
    unique: set[str] = set()
    for feature_space, metrics_by_study in bundle.metrics_by_feature.items():
        for study_key, payload in metrics_by_study.items():
            fields = _extract_fields_from_metrics(payload)
            unique.update(fields)
            study_label = payload.get("study_label")
            if not study_label:
                for study in bundle.studies:
                    if study.key == study_key:
                        study_label = study.label
                        break
            rows.append(
                (
                    feature_space,
                    str(study_label or study_key),
                    _format_field_list(fields),
                )
            )
    return rows, unique


def _collect_opinion_sweeps(outcomes: Sequence[OpinionSweepOutcome]) -> tuple[list[Sequence[str]], set[str]]:
    """Return table rows and unique fields for opinion sweeps."""
    rows: list[Sequence[str]] = []
    unique: set[str] = set()
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for outcome in outcomes:
        fields = _normalise_fields(outcome.config.text_fields)
        unique.update(fields)
        key = (outcome.feature_space, outcome.study.label, outcome.config.label(), fields)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            (
                outcome.feature_space,
                outcome.study.label,
                outcome.config.label(),
                _format_field_list(fields),
            )
        )
    return rows, unique


def _collect_opinion_final(bundle: ReportBundle) -> tuple[list[Sequence[str]], set[str]]:
    """Return table rows and unique fields for opinion final metrics."""
    rows: list[Sequence[str]] = []
    unique: set[str] = set()
    for feature_space, metrics_by_study in bundle.opinion_metrics.items():
        for study_key, payload in metrics_by_study.items():
            fields = _extract_fields_from_metrics(payload)
            unique.update(fields)
            study_label = payload.get("study_label")
            if not study_label:
                for study in bundle.studies:
                    if study.key == study_key:
                        study_label = study.label
                        break
            rows.append(
                (
                    feature_space,
                    str(study_label or study_key),
                    _format_field_list(fields),
                )
            )
    return rows, unique


def build_feature_report(repo_root: Path, bundle: ReportBundle) -> None:
    """Write the KNN additional-feature markdown report."""
    output_dir = repo_root / "reports" / "knn" / "additional_features"
    path, lines = start_markdown_report(output_dir, title="Additional Text Features")
    lines.append("")
    lines.append(
        "Overview of the supplementary text columns appended to the viewer prompt "
        "alongside the prompt builder output."
    )
    lines.append("")

    observed: set[str] = set(_DEFAULT_FIELD_SET)

    if bundle.include_next_video:
        sweep_rows, sweep_fields = _collect_next_video_sweeps(bundle.sweep_outcomes)
        final_rows, final_fields = _collect_next_video_final(bundle)
        lines.append("## Next-Video Pipeline")
        lines.append("")
        if sweep_rows:
            lines.append("### Sweep Configurations")
            lines.append("")
            lines.extend(_render_table(("Feature space", "Study", "Configuration", "Extra text fields"), sweep_rows))
        else:
            lines.append("### Sweep Configurations")
            lines.append("")
            lines.append("No sweep metrics were supplied.")
            lines.append("")
        if final_rows:
            lines.append("### Final Evaluations")
            lines.append("")
            lines.extend(_render_table(("Feature space", "Study", "Extra text fields"), final_rows))
        else:
            lines.append("### Final Evaluations")
            lines.append("")
            lines.append("No final evaluation metrics were supplied.")
            lines.append("")
        observed.update(sweep_fields)
        observed.update(final_fields)
    else:
        lines.append("## Next-Video Pipeline")
        lines.append("")
        lines.append("Next-video stages were disabled for this run.")
        lines.append("")

    if bundle.include_opinion:
        sweep_rows, sweep_fields = _collect_opinion_sweeps(bundle.opinion_sweep_outcomes)
        final_rows, final_fields = _collect_opinion_final(bundle)
        lines.append("## Opinion Regression")
        lines.append("")
        if sweep_rows:
            lines.append("### Sweep Configurations")
            lines.append("")
            lines.extend(_render_table(("Feature space", "Study", "Configuration", "Extra text fields"), sweep_rows))
        else:
            lines.append("### Sweep Configurations")
            lines.append("")
            lines.append("No opinion sweep metrics were supplied.")
            lines.append("")
        if final_rows:
            lines.append("### Final Evaluations")
            lines.append("")
            lines.extend(_render_table(("Feature space", "Study", "Extra text fields"), final_rows))
        else:
            lines.append("### Final Evaluations")
            lines.append("")
            lines.append("No opinion metrics were recorded.")
            lines.append("")
        observed.update(sweep_fields)
        observed.update(final_fields)
    else:
        lines.append("## Opinion Regression")
        lines.append("")
        lines.append("Opinion regression stages were disabled for this run.")
        lines.append("")

    additional = sorted(field for field in observed if field not in _DEFAULT_FIELD_SET)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Default extra text fields: {_format_field_list(_DEFAULT_FIELD_SET)}")
    if additional:
        lines.append(f"- Additional fields observed: {_format_field_list(additional)}")
    else:
        lines.append("- Additional fields observed: none (defaults only).")
    lines.append("")

    write_markdown_lines(path, lines)


__all__ = ["build_feature_report"]
