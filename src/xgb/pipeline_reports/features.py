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

"""Summaries of the extra text features used by the XGBoost pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence, TYPE_CHECKING

from common.prompt_docs import DEFAULT_EXTRA_TEXT_FIELDS
from common.pipeline_io import write_markdown_lines
from common.report_utils import start_markdown_report

from ..pipeline_context import OpinionStudySelection, OpinionSweepOutcome, StudySelection, SweepOutcome

if TYPE_CHECKING:
    from .runner import OpinionReportData, SweepReportData

_DEFAULT_FIELD_SET = tuple(DEFAULT_EXTRA_TEXT_FIELDS)


def _normalise_fields(values: Iterable[object] | None) -> tuple[str, ...]:
    """Return a canonical tuple of extra text fields."""
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
    """Pull extra-field metadata from stored metrics."""
    candidates: list[Iterable[object] | None] = [
        metrics.get("extra_fields"),
        metrics.get("config", {}).get("extra_fields") if isinstance(metrics.get("config"), Mapping) else None,
        metrics.get("xgboost_params", {}).get("extra_fields")
        if isinstance(metrics.get("xgboost_params"), Mapping)
        else None,
    ]
    for payload in candidates:
        fields = _normalise_fields(payload) if payload is not None else ()
        if fields:
            return fields
    return _DEFAULT_FIELD_SET


def _format_field_list(fields: Sequence[str]) -> str:
    """Render a sequence of fields as inline code."""
    if not fields:
        return "—"
    return ", ".join(f"`{field}`" for field in fields)


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render rows as a GitHub-flavoured Markdown table."""
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
    """Return table rows and the unique field set for next-video sweeps."""
    rows: list[Sequence[str]] = []
    unique_fields: set[str] = set()
    seen_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    for outcome in outcomes:
        fields = _extract_fields_from_metrics(outcome.metrics)
        unique_fields.update(fields)
        key = (outcome.study.label, outcome.config.label(), fields)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(
            (
                outcome.study.label,
                outcome.config.label(),
                _format_field_list(fields),
            )
        )
    return rows, unique_fields


def _collect_next_video_final(
    selections: Mapping[str, StudySelection],
    metrics: Mapping[str, Mapping[str, object]],
) -> tuple[list[Sequence[str]], set[str]]:
    """Return table rows and the unique field set for final next-video runs."""
    rows: list[Sequence[str]] = []
    unique_fields: set[str] = set()
    for study_key, payload in metrics.items():
        fields = _extract_fields_from_metrics(payload)
        unique_fields.update(fields)
        study = selections.get(study_key)
        label = payload.get("study_label") or (study.study.label if study else study_key)
        issue = payload.get("issue_label") or payload.get("issue") or (study.study.issue if study else "—")
        rows.append(
            (
                str(label),
                str(issue),
                _format_field_list(fields),
            )
        )
    if not rows and selections:
        for selection in selections.values():
            fields = _extract_fields_from_metrics(selection.outcome.metrics)
            unique_fields.update(fields)
            rows.append(
                (
                    selection.study.label,
                    selection.study.issue,
                    _format_field_list(fields),
                )
            )
    return rows, unique_fields


def _collect_opinion_sweeps(outcomes: Sequence[OpinionSweepOutcome]) -> tuple[list[Sequence[str]], set[str]]:
    """Return table rows and unique field set for opinion sweeps."""
    rows: list[Sequence[str]] = []
    unique_fields: set[str] = set()
    seen_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    for outcome in outcomes:
        fields = _extract_fields_from_metrics(outcome.metrics)
        unique_fields.update(fields)
        key = (outcome.study.label, outcome.config.label(), fields)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(
            (
                outcome.study.label,
                outcome.config.label(),
                _format_field_list(fields),
            )
        )
    return rows, unique_fields


def _collect_opinion_final(
    selections: Mapping[str, OpinionStudySelection],
    metrics: Mapping[str, Mapping[str, object]],
) -> tuple[list[Sequence[str]], set[str]]:
    """Return table rows and unique field set for opinion final evaluations."""
    rows: list[Sequence[str]] = []
    unique_fields: set[str] = set()
    for study_key, payload in metrics.items():
        fields = _extract_fields_from_metrics(payload)
        unique_fields.update(fields)
        selection = selections.get(study_key)
        label = payload.get("study_label") or (selection.study.label if selection else study_key)
        rows.append(
            (
                str(label),
                _format_field_list(fields),
            )
        )
    if not rows and selections:
        for selection in selections.values():
            fields = _extract_fields_from_metrics(selection.outcome.metrics)
            unique_fields.update(fields)
            rows.append(
                (
                    selection.study.label,
                    _format_field_list(fields),
                )
            )
    return rows, unique_fields


def _write_feature_report(
    output_dir: Path,
    sweeps: "SweepReportData",
    *,
    include_next_video: bool,
    opinion: "OpinionReportData" | None,
) -> None:
    """Emit Markdown summarising the extra feature fields used across the run."""
    path, lines = start_markdown_report(output_dir, title="Additional Text Features")
    lines.append("")
    lines.append(
        "This report tracks the supplementary text columns appended to the prompt "
        "builder output during training and evaluation."
    )
    lines.append("")

    stage_fields: set[str] = set(_DEFAULT_FIELD_SET)

    if include_next_video:
        sweep_rows, sweep_fields = _collect_next_video_sweeps(sweeps.outcomes)
        final_rows, final_fields = _collect_next_video_final(sweeps.selections, sweeps.final_metrics)
        lines.append("## Next-Video Pipeline")
        lines.append("")
        if sweep_rows:
            lines.append("### Sweep Configurations")
            lines.append("")
            lines.extend(_render_table(("Study", "Configuration", "Extra text fields"), sweep_rows))
        else:
            lines.append("### Sweep Configurations")
            lines.append("")
            lines.append("No sweep metrics found for the selected run.")
            lines.append("")
        if final_rows:
            lines.append("### Final Evaluations")
            lines.append("")
            lines.extend(_render_table(("Study", "Issue", "Extra text fields"), final_rows))
        else:
            lines.append("### Final Evaluations")
            lines.append("")
            lines.append("No final evaluation metrics were supplied.")
            lines.append("")
        stage_fields.update(sweep_fields)
        stage_fields.update(final_fields)
    else:
        lines.append("## Next-Video Pipeline")
        lines.append("")
        lines.append("Next-video tasks were disabled for this run.")
        lines.append("")

    if opinion is not None:
        sweep_rows, sweep_fields = _collect_opinion_sweeps(opinion.outcomes)
        final_rows, final_fields = _collect_opinion_final(opinion.selections, opinion.metrics)
        lines.append("## Opinion Regression")
        lines.append("")
        if sweep_rows:
            lines.append("### Sweep Configurations")
            lines.append("")
            lines.extend(_render_table(("Study", "Configuration", "Extra text fields"), sweep_rows))
        else:
            lines.append("### Sweep Configurations")
            lines.append("")
            lines.append("No opinion sweep metrics were provided.")
            lines.append("")
        if final_rows:
            lines.append("### Final Evaluations")
            lines.append("")
            lines.extend(_render_table(("Study", "Extra text fields"), final_rows))
        else:
            lines.append("### Final Evaluations")
            lines.append("")
            lines.append("Opinion final metrics were not generated.")
            lines.append("")
        stage_fields.update(sweep_fields)
        stage_fields.update(final_fields)
    else:
        lines.append("## Opinion Regression")
        lines.append("")
        lines.append("Opinion regression tasks were disabled for this run.")
        lines.append("")

    additional_fields = sorted(field for field in stage_fields if field not in _DEFAULT_FIELD_SET)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Default extra text fields: {_format_field_list(_DEFAULT_FIELD_SET)}")
    if additional_fields:
        lines.append(f"- Additional fields observed: {_format_field_list(additional_fields)}")
    else:
        lines.append("- Additional fields observed: none (defaults only).")
    lines.append("")

    write_markdown_lines(path, lines)


__all__ = ["_write_feature_report"]
