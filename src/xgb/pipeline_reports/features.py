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

from ..pipeline_context import (
    OpinionStudySelection,
    OpinionSweepOutcome,
    StudySelection,
    SweepOutcome,
)

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
    config = metrics.get("config")
    xgboost_params = metrics.get("xgboost_params")
    candidates: list[Iterable[object] | None] = [
        metrics.get("extra_fields"),
        config.get("extra_fields") if isinstance(config, Mapping) else None,
        xgboost_params.get("extra_fields") if isinstance(xgboost_params, Mapping) else None,
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


def _append_section(
    lines: list[str],
    heading: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    empty_message: str,
) -> None:
    """Append a subsection summarising sweep or final metrics."""
    lines.append(f"### {heading}")
    lines.append("")
    if rows:
        lines.extend(_render_table(headers, rows))
    else:
        lines.append(empty_message)
        lines.append("")


def _append_next_video_section(
    lines: list[str],
    sweeps: "SweepReportData",
    include_next_video: bool,
) -> set[str]:
    """Add the next-video section and return the observed fields."""
    lines.append("## Next-Video Pipeline")
    lines.append("")
    if not include_next_video:
        lines.append("Next-video tasks were disabled for this run.")
        lines.append("")
        return set()

    sweep_rows, sweep_fields = _collect_next_video_sweeps(sweeps.outcomes)
    final_rows, final_fields = _collect_next_video_final(
        sweeps.selections,
        sweeps.final_metrics,
    )

    _append_section(
        lines,
        "Sweep Configurations",
        ("Study", "Configuration", "Extra text fields"),
        sweep_rows,
        "No sweep metrics found for the selected run.",
    )
    _append_section(
        lines,
        "Final Evaluations",
        ("Study", "Issue", "Extra text fields"),
        final_rows,
        "No final evaluation metrics were supplied.",
    )

    return sweep_fields | final_fields


def _append_opinion_section(
    lines: list[str],
    opinion: "OpinionReportData" | None,
) -> set[str]:
    """Add the opinion regression section and return the observed fields."""
    lines.append("## Opinion Regression")
    lines.append("")
    if opinion is None:
        lines.append("Opinion regression tasks were disabled for this run.")
        lines.append("")
        return set()

    sweep_rows, sweep_fields = _collect_opinion_sweeps(opinion.outcomes)
    final_rows, final_fields = _collect_opinion_final(
        opinion.selections,
        opinion.metrics,
    )

    _append_section(
        lines,
        "Sweep Configurations",
        ("Study", "Configuration", "Extra text fields"),
        sweep_rows,
        "No opinion sweep metrics were provided.",
    )
    _append_section(
        lines,
        "Final Evaluations",
        ("Study", "Extra text fields"),
        final_rows,
        "Opinion final metrics were not generated.",
    )

    return sweep_fields | final_fields


def _collect_next_video_sweeps(
    outcomes: Sequence[SweepOutcome],
) -> tuple[list[Sequence[str]], set[str]]:
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
        label = payload.get("study_label") or (
            study.study.label if study else study_key
        )
        issue = (
            payload.get("issue_label")
            or payload.get("issue")
            or (study.study.issue if study else "—")
        )
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


def _collect_opinion_sweeps(
    outcomes: Sequence[OpinionSweepOutcome],
) -> tuple[list[Sequence[str]], set[str]]:
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

    stage_fields.update(
        _append_next_video_section(lines, sweeps, include_next_video)
    )
    stage_fields.update(_append_opinion_section(lines, opinion))

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
