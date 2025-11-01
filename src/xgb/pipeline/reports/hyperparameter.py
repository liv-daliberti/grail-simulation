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

"""Hyper-parameter sweep reporting for the XGBoost pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from common.pipeline.formatters import (
    format_count as _format_count,
    format_delta as _format_delta,
    format_float as _format_float,
    format_optional_float as _format_optional_float,
    format_ratio as _format_ratio,
)
from common.pipeline.io import write_markdown_lines
from common.reports.utils import start_markdown_report
from common.reports.fields import (
    NEXT_VIDEO_COVERAGE_FIELDS as _COVERAGE_FNS,
    next_video_coverage_mapping as _coverage_map,
)

from ..context import (
    OpinionStudySelection,
    OpinionSummary,
    OpinionSweepOutcome,
    StudySelection,
    SweepConfig,
    SweepOutcome,
)
from .next_video import _extract_next_video_summary
from .opinion import _OpinionPortfolioAccumulator, _extract_opinion_summary
from .shared import _xgb_next_video_command, _xgb_opinion_command

if TYPE_CHECKING:
    from .runner import OpinionReportData, SweepReportData

HYPERPARAM_TABLE_TOP_N = 10
HYPERPARAM_LEADERBOARD_TOP_N = 5


@dataclass
class _NextVideoStudyView:
    """Precomputed data required to render a next-video sweep table."""

    key: str
    label: str
    issue_label: str
    selection: Optional[StudySelection]
    ordered: List[SweepOutcome]
    displayed: List[SweepOutcome]
    display_limit: int

    def has_more_results(self) -> bool:
        """Return ``True`` when additional configurations were truncated."""

        return len(self.ordered) > self.display_limit


def _group_outcomes_by_study(
    outcomes: Sequence[SweepOutcome],
) -> Dict[str, List[SweepOutcome]]:
    """Bucket sweep outcomes by study key."""

    grouped: Dict[str, List[SweepOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.study.key, []).append(outcome)
    return grouped


def _next_video_sort_key(
    study_key: str,
    selections: Mapping[str, StudySelection],
    grouped: Mapping[str, Sequence[SweepOutcome]],
) -> str:
    """Return a case-insensitive sort key for a study."""

    selection = selections.get(study_key)
    if selection is not None:
        return selection.study.label.lower()
    outcomes = grouped.get(study_key)
    if outcomes:
        return outcomes[0].study.label.lower()
    return study_key.lower()


def _build_next_video_view(
    study_key: str,
    outcomes: Sequence[SweepOutcome],
    selection: Optional[StudySelection],
    *,
    display_limit: int,
) -> _NextVideoStudyView:
    """Assemble ordering and display metadata for a study."""

    reference_study = selection.study if selection is not None else outcomes[0].study
    ordered = sorted(
        outcomes,
        key=lambda item: (item.accuracy, item.coverage, item.evaluated),
        reverse=True,
    )
    displayed = list(ordered[:display_limit])
    if selection is not None:
        selected = next(
            (candidate for candidate in ordered if candidate.config == selection.config),
            None,
        )
        if selected is not None and selected not in displayed:
            displayed.append(selected)
            displayed.sort(
                key=lambda item: (item.accuracy, item.coverage, item.evaluated),
                reverse=True,
            )
            del displayed[display_limit:]
    issue_label = reference_study.issue.replace("_", " ").title()
    return _NextVideoStudyView(
        key=study_key,
        label=reference_study.label,
        issue_label=issue_label,
        selection=selection,
        ordered=list(ordered),
        displayed=displayed,
        display_limit=display_limit,
    )


def _render_next_video_table(view: _NextVideoStudyView) -> List[str]:
    """Render the Markdown table for a single study view."""

    lines: List[str] = [f"### {view.label}", "", f"*Issue:* {view.issue_label}", ""]
    lines.append(
        "| Config | Accuracy ↑ | Acc (eligible) ↑ | Coverage ↑ | Known hits / total | "
        "Known availability ↑ | Avg prob ↑ | Evaluated |"
    )
    lines.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for outcome in view.displayed:
        label = outcome.config.label()
        formatted = (
            f"**{label}**"
            if view.selection and outcome.config == view.selection.config
            else label
        )
        summary = _extract_next_video_summary(outcome.metrics)
        lines.append(
            f"| {formatted} | {_format_optional_float(summary.accuracy)} | "
            f"{_format_optional_float(summary.accuracy_eligible)} | "
            f"{_format_optional_float(summary.coverage)} | "
            f"{_format_ratio(summary.known_hits, summary.known_total)} | "
            f"{_format_optional_float(summary.known_availability)} | "
            f"{_format_optional_float(summary.avg_probability)} | "
            f"{_format_count(summary.evaluated)} |"
        )
    if view.has_more_results():
        total = len(view.ordered)
        lines.append(f"*Showing top {view.display_limit} of {total} configurations.*")
    lines.append("")
    return lines


def _next_video_empty_section(allow_incomplete: bool) -> List[str]:
    """Return messaging for missing next-video sweeps."""

    lines = ["## Next-Video Sweeps", ""]
    lines.append(
        "No next-video sweep runs were available when this report was generated."
    )
    if allow_incomplete:
        lines.append(
            "Run the XGBoost pipeline with `--stage sweeps` or `--stage full` once "
            "artifacts are ready."
        )
    lines.append("")
    return lines


def _render_next_video_sections(
    outcomes: Sequence[SweepOutcome],
    selections: Mapping[str, StudySelection],
    *,
    allow_incomplete: bool,
) -> Tuple[List[str], Dict[str, List[SweepOutcome]]]:
    """
    Render the next-video sweep portion of the hyper-parameter report.

    Returns both the Markdown lines and the ordered outcomes per study so that
    leaderboard sections can reuse the ranking.
    """

    grouped = _group_outcomes_by_study(outcomes)
    if not grouped:
        return (_next_video_empty_section(allow_incomplete), {})

    lines: List[str] = ["## Next-Video Sweeps", ""]
    sorted_outcomes: Dict[str, List[SweepOutcome]] = {}
    display_limit = max(1, HYPERPARAM_TABLE_TOP_N)
    for study_key in sorted(
        grouped.keys(), key=lambda key: _next_video_sort_key(key, selections, grouped)
    ):
        selection = selections.get(study_key)
        view = _build_next_video_view(
            study_key,
            grouped[study_key],
            selection,
            display_limit=display_limit,
        )
        lines.extend(_render_next_video_table(view))
        sorted_outcomes[study_key] = view.ordered
    return (lines, sorted_outcomes)


def _write_hyperparameter_report(
    directory: Path,
    sweeps: "SweepReportData",
    *,
    allow_incomplete: bool,
    include_next_video: bool,
    opinion: "OpinionReportData" | None = None,
) -> None:
    """
    Create the hyper-parameter sweep summary document.

    :param directory: Directory where the report and assets are written.
    :type directory: Path
    :param sweeps: Sweep outcomes, selections, and evaluation metrics.
    :type sweeps: SweepReportData
    :param allow_incomplete: Flag controlling placeholder messaging when artefacts are missing.
    :type allow_incomplete: bool
    """

    path, lines = start_markdown_report(directory, title="Hyper-parameter Tuning")
    lines.append(
        "This summary lists the top-performing configurations uncovered during the "
        "hyper-parameter sweeps."
    )
    if include_next_video:
        lines.append(
            f"- Next-video tables highlight up to {HYPERPARAM_TABLE_TOP_N} "
            "configurations per study ranked by validation accuracy."
        )
        lines.append("- Eligible-only accuracy is shown for comparison next to overall accuracy.")
    include_opinion = opinion is not None
    if include_opinion:
        lines.append(
            f"- Opinion regression tables highlight up to {HYPERPARAM_TABLE_TOP_N} "
            "configurations per study ranked by MAE relative to the baseline."
        )
    lines.append("- Rows in bold mark the configuration promoted to the final evaluation.")
    lines.append("")

    sorted_study_outcomes: Dict[str, List[SweepOutcome]] = {}
    if include_next_video:
        next_video_lines, sorted_study_outcomes = _render_next_video_sections(
            sweeps.outcomes,
            sweeps.selections,
            allow_incomplete=allow_incomplete,
        )
        lines.extend(next_video_lines)
        # CSV dump for next-video sweeps
        _write_next_video_sweeps_csv(directory, sweeps.outcomes)
        if sorted_study_outcomes:
            lines.extend(
                _xgb_leaderboard_section(
                    per_study_sorted=sorted_study_outcomes,
                    selections=sweeps.selections,
                    top_n=HYPERPARAM_LEADERBOARD_TOP_N,
                )
            )
            if sweeps.selections:
                lines.extend(
                    _xgb_selection_summary_section(sorted_study_outcomes, sweeps.selections)
                )
                lines.extend(_xgb_parameter_frequency_section(sweeps.selections))

    if include_opinion:
        lines.extend(
            _opinion_hyperparameter_section(
                outcomes=opinion.outcomes,
                selections=opinion.selections,
                allow_incomplete=allow_incomplete,
            )
        )

    write_markdown_lines(path, lines)


def _write_next_video_sweeps_csv(directory: Path, outcomes: Sequence[SweepOutcome]) -> None:
    """Write a CSV with next-video sweep outcomes for downstream analysis."""

    if not outcomes:
        return
    out_path = directory / "next_video_sweeps.csv"
    fieldnames = [
        "study_key",
        "study_label",
        "issue",
        "config_label",
        "accuracy",
        "accuracy_eligible",
        *_COVERAGE_FNS,
        "evaluated",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for outcome in outcomes:
            summary = _extract_next_video_summary(outcome.metrics)
            writer.writerow(
                {
                    "study_key": outcome.study.key,
                    "study_label": outcome.study.label,
                    "issue": outcome.study.issue,
                    "config_label": outcome.config.label(),
                    "accuracy": summary.accuracy,
                    "accuracy_eligible": summary.accuracy_eligible,
                    **_coverage_map(summary),
                    "evaluated": summary.evaluated,
                }
            )


@dataclass
class _OpinionStudyView:
    """Aggregated information required to render an opinion sweep table."""

    key: str
    label: str
    issue_label: str
    selection: Optional["xgb.pipeline.context.OpinionStudySelection"]
    ordered: List[OpinionSweepOutcome]
    displayed: List[OpinionSweepOutcome]
    display_limit: int

    def has_more_results(self) -> bool:
        """Return ``True`` when additional configurations were truncated."""

        return len(self.ordered) > self.display_limit

    def selected_config(self) -> Optional["xgb.pipeline.context.SweepConfig"]:
        """Return the SweepConfig associated with the promoted outcome."""

        if self.selection is None:
            return None
        return self.selection.outcome.config


def _group_opinion_outcomes(
    outcomes: Sequence[OpinionSweepOutcome],
) -> Dict[str, List[OpinionSweepOutcome]]:
    """Bucket opinion sweep outcomes by study key."""

    grouped: Dict[str, List[OpinionSweepOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.study.key, []).append(outcome)
    return grouped


def _opinion_sort_key(
    study_key: str,
    selections: Mapping[str, "xgb.pipeline.context.OpinionStudySelection"],
    grouped: Mapping[str, Sequence[OpinionSweepOutcome]],
) -> str:
    """Return a case-insensitive sort key for the opinion study tables."""

    selection = selections.get(study_key)
    if selection is not None:
        return selection.study.label.lower()
    candidates = grouped.get(study_key)
    if candidates:
        return candidates[0].study.label.lower()
    return study_key.lower()


def _build_opinion_view(
    study_key: str,
    outcomes: Sequence[OpinionSweepOutcome],
    selection: Optional["xgb.pipeline.context.OpinionStudySelection"],
    *,
    display_limit: int,
) -> _OpinionStudyView:
    """Prepare ordered/displayed opinion outcomes for rendering."""

    reference = selection.study if selection is not None else outcomes[0].study
    ordered = sorted(
        outcomes,
        key=lambda item: (item.mae, item.rmse, -item.r_squared, item.order_index),
    )
    displayed = list(ordered[:display_limit])
    if selection is not None and selection.outcome not in displayed:
        displayed.append(selection.outcome)
        displayed.sort(
            key=lambda item: (item.mae, item.rmse, -item.r_squared, item.order_index),
        )
        del displayed[display_limit:]
    issue_label = reference.issue.replace("_", " ").title()
    return _OpinionStudyView(
        key=study_key,
        label=reference.label,
        issue_label=issue_label,
        selection=selection,
        ordered=list(ordered),
        displayed=displayed,
        display_limit=display_limit,
    )


def _render_opinion_table(view: _OpinionStudyView) -> List[str]:
    """Render Markdown lines for a single opinion sweep table."""

    lines: List[str] = [f"### {view.label}", "", f"*Issue:* {view.issue_label}", ""]
    lines.append(
        "| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | "
        "MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    selected_config = view.selected_config()
    for outcome in view.displayed:
        summary = _extract_opinion_summary(outcome.metrics)
        config_label = outcome.config.label()
        config_summary = _summarise_xgb_config(outcome.config)
        config_cell = f"{config_label}<br>{config_summary}"
        if selected_config is not None and outcome.config == selected_config:
            config_cell = f"**{config_label}**<br>{config_summary}"
        accuracy_text = _format_optional_float(summary.accuracy_after)
        baseline_text = _format_optional_float(summary.baseline_accuracy)
        accuracy_delta = _format_delta(summary.accuracy_delta)
        eligible_value = summary.eligible if summary.eligible is not None else summary.participants
        eligible_text = _format_count(eligible_value)
        lines.append(
            f"| {config_cell} | {accuracy_text} | {baseline_text} | {accuracy_delta} | "
            f"— | {eligible_text} | {_format_optional_float(summary.mae_after)} | "
            f"{_format_delta(summary.mae_delta)} | "
            f"{_format_optional_float(summary.rmse_after)} | "
            f"{_format_optional_float(summary.r2_after)} |"
        )
    if view.has_more_results():
        total = len(view.ordered)
        lines.append(f"*Showing top {view.display_limit} of {total} configurations.*")
    reproduction = _xgb_opinion_command(view.selection)
    if reproduction:
        lines.append(f"  Command: `{reproduction}`")
    lines.append("")
    return lines


def _opinion_empty_section(allow_incomplete: bool) -> List[str]:
    """Return messaging for missing opinion sweeps."""

    lines = [
        "No opinion sweep runs were available when this report was generated."
    ]
    if allow_incomplete:
        lines.append(
            "Run the XGBoost pipeline with `--stage sweeps` or `--stage full` once "
            "artifacts are ready."
        )
    lines.append("")
    return lines


def _selected_opinion_summary(view: _OpinionStudyView) -> Optional[OpinionSummary]:
    """Return the summary associated with the promoted configuration."""

    if view.selection is not None:
        return _extract_opinion_summary(view.selection.outcome.metrics)
    if view.displayed:
        return _extract_opinion_summary(view.displayed[0].metrics)
    return None


def _opinion_hyperparameter_section(
    *,
    outcomes: Sequence[OpinionSweepOutcome],
    selections: Mapping[str, "xgb.pipeline.context.OpinionStudySelection"],
    allow_incomplete: bool,
) -> List[str]:
    """
    Render the opinion hyper-parameter sweep summary.

    :param outcomes: Opinion sweep outcomes considered during selection.
    :type outcomes: Sequence[~xgb.pipeline.context.OpinionSweepOutcome]
    :param selections: Mapping from study key to selected opinion sweep outcome.
    :type selections: Mapping[str, ~xgb.pipeline.context.OpinionStudySelection]
    :param allow_incomplete: Flag controlling placeholder messaging when artefacts are missing.
    :type allow_incomplete: bool
    :returns: Markdown lines describing opinion sweeps.
    :rtype: List[str]
    """

    lines: List[str] = ["## Opinion Regression Sweeps", ""]
    grouped = _group_opinion_outcomes(outcomes)
    portfolio = _OpinionPortfolioAccumulator()

    if not grouped:
        lines.extend(_opinion_empty_section(allow_incomplete))
        return lines

    display_limit = max(1, HYPERPARAM_TABLE_TOP_N)
    for study_key in sorted(
        grouped.keys(), key=lambda key: _opinion_sort_key(key, selections, grouped)
    ):
        selection = selections.get(study_key)
        view = _build_opinion_view(
            study_key,
            grouped[study_key],
            selection,
            display_limit=display_limit,
        )
        lines.extend(_render_opinion_table(view))
        summary = _selected_opinion_summary(view)
        if summary is not None:
            portfolio.record(summary, view.label)

    lines.extend(portfolio.to_lines(heading="### Portfolio Summary"))
    return lines


def _leaderboard_label(
    study_key: str,
    selections: Mapping[str, StudySelection],
    per_study_sorted: Mapping[str, Sequence[SweepOutcome]],
) -> str:
    """Return the display label used in leaderboard sections."""

    selection = selections.get(study_key)
    if selection is not None:
        return selection.study.label
    outcomes = per_study_sorted.get(study_key, ())
    if outcomes:
        return outcomes[0].study.label
    return study_key


def _format_leaderboard_row(
    rank: int,
    outcome: SweepOutcome,
    selection: Optional[StudySelection],
    *,
    best_accuracy: Optional[float],
    best_coverage: Optional[float],
) -> str:
    """Format a single leaderboard row."""

    label = outcome.config.label()
    if selection and outcome.config == selection.config:
        label_text = f"**{label}**"
    else:
        label_text = label

    delta_accuracy = None
    if best_accuracy is not None and outcome.accuracy is not None:
        delta_accuracy = max(0.0, best_accuracy - outcome.accuracy)
    delta_coverage = None
    if best_coverage is not None and outcome.coverage is not None:
        delta_coverage = max(0.0, best_coverage - outcome.coverage)

    return (
        f"| {rank} | {label_text} | {_format_optional_float(outcome.accuracy)} | "
        f"{_format_optional_float(delta_accuracy)} | "
        f"{_format_optional_float(outcome.coverage)} | "
        f"{_format_optional_float(delta_coverage)} | "
        f"{_format_count(outcome.evaluated)} |"
    )


def _render_leaderboard_for_study(
    label: str,
    outcomes: Sequence[SweepOutcome],
    selection: Optional[StudySelection],
    *,
    limit: int,
) -> List[str]:
    """Render leaderboard rows for a single study."""

    if not outcomes:
        return []
    best = outcomes[0]
    lines: List[str] = [f"#### {label}", ""]
    lines.append(
        "| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |"
    )
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for rank, outcome in enumerate(outcomes[:limit], start=1):
        lines.append(
            _format_leaderboard_row(
                rank,
                outcome,
                selection,
                best_accuracy=best.accuracy,
                best_coverage=best.coverage,
            )
        )
    if len(outcomes) > limit:
        lines.append(f"*Showing top {limit} of {len(outcomes)} configurations.*")
    lines.append("")
    return lines


def _xgb_leaderboard_section(
    *,
    per_study_sorted: Mapping[str, Sequence[SweepOutcome]],
    selections: Mapping[str, StudySelection],
    top_n: int,
) -> List[str]:
    """
    Render ranked leaderboards mirroring the KNN report format.

    :param per_study_sorted: Mapping from study key to ordered sweep outcomes.
    :type per_study_sorted: Mapping[str, Sequence[~xgb.pipeline.context.SweepOutcome]]
    :param selections: Mapping from study key to selected sweep outcome.
    :type selections: Mapping[str, StudySelection]
    :param top_n: Maximum number of leaderboard entries per study.
    :type top_n: int
    :returns: Markdown lines representing the leaderboard section.
    :rtype: List[str]
    """

    if not per_study_sorted:
        return []

    lines: List[str] = ["### Configuration Leaderboards", ""]
    limit = max(1, top_n)
    sorted_keys = sorted(
        per_study_sorted.keys(),
        key=lambda key: _leaderboard_label(key, selections, per_study_sorted).lower(),
    )
    for study_key in sorted_keys:
        label = _leaderboard_label(study_key, selections, per_study_sorted)
        selection = selections.get(study_key)
        lines.extend(
            _render_leaderboard_for_study(
                label,
                per_study_sorted.get(study_key, ()),
                selection,
                limit=limit,
            )
        )
    return lines


@dataclass
class _SelectionSummaryView:
    """View model capturing data for selection summary bullets."""

    descriptor: str
    selection: Optional[StudySelection]
    ordered: Sequence[SweepOutcome]


def _build_selection_view(
    study_key: str,
    per_study_sorted: Mapping[str, Sequence[SweepOutcome]],
    selections: Mapping[str, StudySelection],
) -> Optional[_SelectionSummaryView]:
    """Construct the summary view for a study."""

    ordered = per_study_sorted.get(study_key, ())
    if not ordered:
        return None
    selection = selections.get(study_key)
    descriptor = selection.study.label if selection is not None else ordered[0].study.label
    issue_label = ordered[0].study.issue.replace("_", " ").title()
    descriptor_full = f"{descriptor} (issue {issue_label})"
    return _SelectionSummaryView(descriptor_full, selection, ordered)


def _selection_summary_lines(view: _SelectionSummaryView) -> List[str]:
    """Return the bullet lines describing the chosen configuration."""

    lines: List[str] = []
    selection = view.selection
    if selection is None:
        top = view.ordered[0]
        lines.append(
            f"- **{view.descriptor}**: accuracy {_format_float(top.accuracy)} "
            f"with {_summarise_xgb_config(top.config)}."
        )
        return lines

    best = selection.outcome
    summary = _summarise_xgb_config(selection.config)
    runner_up = view.ordered[1] if len(view.ordered) > 1 else None
    if runner_up is not None:
        delta_acc = best.accuracy - runner_up.accuracy
        delta_cov = best.coverage - runner_up.coverage
        lines.append(
            f"- **{view.descriptor}**: accuracy {_format_float(best.accuracy)} "
            f"(coverage {_format_float(best.coverage)}) using {summary}. "
            f"Δ accuracy vs. runner-up {_format_delta(delta_acc)}; "
            f"Δ coverage {_format_delta(delta_cov)}."
        )
    else:
        lines.append(
            f"- **{view.descriptor}**: accuracy {_format_float(best.accuracy)} "
            f"(coverage {_format_float(best.coverage)}) using {summary}."
        )
    reproduction = _xgb_next_video_command(selection)
    if reproduction:
        lines.append(f"  Command: `{reproduction}`")
    return lines


def _xgb_selection_summary_section(
    per_study_sorted: Mapping[str, Sequence[SweepOutcome]],
    selections: Mapping[str, StudySelection],
) -> List[str]:
    """
    Render a bullet summary comparing winning configurations to runner-ups.

    :param per_study_sorted: Mapping from study key to ordered sweep outcomes.
    :type per_study_sorted: Mapping[str, Sequence[~xgb.pipeline.context.SweepOutcome]]
    :param selections: Mapping from study key to selected sweep outcome.
    :type selections: Mapping[str, StudySelection]
    :returns: Markdown bullet list highlighting improvements over runner-ups.
    :rtype: List[str]
    """

    if not per_study_sorted:
        return []

    lines: List[str] = ["### Selection Summary", ""]
    sorted_keys = sorted(
        per_study_sorted.keys(),
        key=lambda key: _leaderboard_label(key, selections, per_study_sorted).lower(),
    )
    for study_key in sorted_keys:
        view = _build_selection_view(study_key, per_study_sorted, selections)
        if view is None:
            continue
        lines.extend(_selection_summary_lines(view))
    lines.append("")
    return lines


def _xgb_parameter_frequency_section(
    selections: Mapping[str, StudySelection],
) -> List[str]:
    """
    Summarise the hyper-parameter values chosen across all studies.

    :param selections: Mapping from study key to selected sweep outcome.
    :type selections: Mapping[str, StudySelection]
    :returns: Markdown lines describing parameter frequency tables.
    :rtype: List[str]
    """

    if not selections:
        return []

    param_counters = {
        "text_vectorizer": Counter(),
        "learning_rate": Counter(),
        "max_depth": Counter(),
        "n_estimators": Counter(),
        "subsample": Counter(),
        "colsample_bytree": Counter(),
        "reg_lambda": Counter(),
        "reg_alpha": Counter(),
    }

    for selection in selections.values():
        config = selection.config
        param_counters["text_vectorizer"][config.text_vectorizer] += 1
        param_counters["learning_rate"][config.learning_rate] += 1
        param_counters["max_depth"][config.max_depth] += 1
        param_counters["n_estimators"][config.n_estimators] += 1
        param_counters["subsample"][config.subsample] += 1
        param_counters["colsample_bytree"][config.colsample_bytree] += 1
        param_counters["reg_lambda"][config.reg_lambda] += 1
        param_counters["reg_alpha"][config.reg_alpha] += 1

    display_names = {
        "text_vectorizer": "Vectorizer",
        "learning_rate": "Learning rate",
        "max_depth": "Max depth",
        "n_estimators": "Estimators",
        "subsample": "Subsample",
        "colsample_bytree": "Column subsample",
        "reg_lambda": "L2 regularisation",
        "reg_alpha": "L1 regularisation",
    }

    lines: List[str] = ["### Parameter Frequency Across Selected Configurations", ""]
    lines.append("| Parameter | Preferred values (count) |")
    lines.append("| --- | --- |")
    for key, label in display_names.items():
        counter = param_counters.get(key, Counter())
        lines.append(f"| {label} | {_format_param_counter(counter)} |")
    lines.append("")
    return lines


def _summarise_xgb_config(config: SweepConfig) -> str:
    """
    Return a human-readable description of a sweep configuration.

    :param config: Sweep configuration to summarise.
    :type config: ~xgb.pipeline.context.SweepConfig
    :returns: Comma-separated summary highlighting key hyper-parameters.
    :rtype: str
    """

    vectorizer = config.text_vectorizer
    if config.vectorizer_tag and config.vectorizer_tag != config.text_vectorizer:
        vectorizer = f"{vectorizer} ({config.vectorizer_tag})"
    return (
        f"vectorizer={vectorizer}, lr={config.learning_rate:g}, depth={config.max_depth}, "
        f"estimators={config.n_estimators}, subsample={config.subsample:g}, "
        f"colsample={config.colsample_bytree:g}, λ={config.reg_lambda:g}, α={config.reg_alpha:g}"
    )


def _format_param_counter(counter: Counter) -> str:
    """
    Format a parameter usage counter for Markdown output.

    :param counter: Counter mapping parameter values to counts.
    :type counter: Counter
    :returns: Concise string showing values with usage counts.
    :rtype: str
    """

    if not counter:
        return "—"
    parts = []
    for value, count in counter.most_common():
        if isinstance(value, float):
            value_repr = f"{value:g}"
        else:
            value_repr = str(value)
        parts.append(f"{value_repr} ×{count}")
    return ", ".join(parts)


__all__ = [
    "_write_hyperparameter_report",
]
