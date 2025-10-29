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

"""Builders for the KNN hyper-parameter Markdown reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..context import (
    MetricSummary,
    OpinionStudySelection,
    OpinionSweepOutcome,
    OpinionSummary,
    StudySelection,
    StudySpec,
    SweepOutcome,
)
from ..utils import (
    extract_metric_summary,
    extract_opinion_summary,
    format_count,
    format_delta,
    format_k,
    format_optional_float,
)
from .opinion import _OpinionPortfolioStats
from .shared import _feature_space_heading, _format_shell_command


@dataclass(frozen=True)
class HyperparameterCommonContext:
    """Shared context used across hyper-parameter report sections."""

    studies: Sequence[StudySpec]
    feature_spaces: Sequence[str]
    k_sweep: Sequence[int]
    sentence_model: Optional[str]


@dataclass(frozen=True)
class NextVideoSectionConfig:
    """Configuration for the next-video portion of the hyper-parameter report."""

    selections: Mapping[str, Mapping[str, StudySelection]]
    sweep_outcomes: Sequence[SweepOutcome]


@dataclass(frozen=True)
class OpinionSectionConfig:
    """Configuration for the opinion portion of the hyper-parameter report."""

    selections: Mapping[str, Mapping[str, OpinionStudySelection]]
    sweep_outcomes: Sequence[OpinionSweepOutcome]


@dataclass(frozen=True)
class HyperparameterReportConfig:
    """Configuration bundle for the hyper-parameter report builder."""

    output_dir: Path
    common: HyperparameterCommonContext
    allow_incomplete: bool
    next_video: Optional[NextVideoSectionConfig] = None
    opinion: Optional[OpinionSectionConfig] = None


def _metric_summary_to_mapping(summary: MetricSummary) -> Dict[str, object]:
    """Return a dictionary view used by legacy formatting helpers."""

    accuracy = summary.accuracy
    baseline = summary.baseline
    delta = None
    if accuracy is not None and baseline is not None:
        delta = accuracy - baseline

    return {
        "accuracy": accuracy,
        "baseline": baseline,
        "delta": delta,
        "eligible": summary.n_eligible,
        "best_k": summary.best_k,
    }


def _opinion_summary_to_mapping(summary: OpinionSummary) -> Dict[str, object]:
    """Return a dictionary view of opinion summaries for report rendering."""

    return {
        "accuracy": summary.accuracy,
        "baseline": summary.baseline_accuracy,
        "delta": summary.accuracy_delta,
        "eligible": summary.eligible,
        "participants": summary.participants,
        "best_k": summary.best_k,
    }


def _knn_next_video_command(
    feature_space: str,
    selection: Optional[StudySelection],
) -> Optional[str]:
    """Build a reproduction command for a next-video sweep selection."""
    if selection is None:
        return None
    metrics = selection.outcome.metrics or {}
    dataset = metrics.get("dataset") or "data/cleaned_grail"
    config = selection.config
    text_fields_value = ",".join(config.text_fields) if config.text_fields else ""

    command: List[str] = [
        "python",
        "-m",
        "knn.cli",
        "--task",
        "slate",
        "--dataset",
        str(dataset),
        "--feature-space",
        feature_space,
        "--issues",
        selection.study.issue,
        "--participant-studies",
        selection.study.key,
        "--knn-metric",
        config.metric,
        "--knn-k",
        str(selection.best_k),
        "--knn-k-sweep",
        str(selection.best_k),
        "--out-dir",
        "<run_dir>",
    ]
    command.extend(["--knn-text-fields", text_fields_value])
    if config.word2vec_size is not None:
        command.extend(["--word2vec-size", str(config.word2vec_size)])
    if config.word2vec_window is not None:
        command.extend(["--word2vec-window", str(config.word2vec_window)])
    if config.word2vec_min_count is not None:
        command.extend(["--word2vec-min-count", str(config.word2vec_min_count)])
    if config.word2vec_epochs is not None:
        command.extend(["--word2vec-epochs", str(config.word2vec_epochs)])
    if config.word2vec_workers is not None:
        command.extend(["--word2vec-workers", str(config.word2vec_workers)])
    if config.sentence_transformer_model:
        command.extend(
            ["--sentence-transformer-model", config.sentence_transformer_model]
        )
    if config.sentence_transformer_device:
        command.extend(
            ["--sentence-transformer-device", config.sentence_transformer_device]
        )
    if config.sentence_transformer_batch_size is not None:
        command.extend(
            [
                "--sentence-transformer-batch-size",
                str(config.sentence_transformer_batch_size),
            ]
        )
    if config.sentence_transformer_normalize is not None:
        command.append(
            "--sentence-transformer-normalize"
            if config.sentence_transformer_normalize
            else "--sentence-transformer-no-normalize"
        )
    return _format_shell_command(command)


def _knn_opinion_command(
    feature_space: str,
    selection: Optional[OpinionStudySelection],
) -> Optional[str]:
    """Build a reproduction command for an opinion sweep selection."""
    if selection is None:
        return None
    outcome = selection.outcome
    metrics = outcome.metrics or {}
    dataset = metrics.get("dataset") or "data/cleaned_grail"
    config = outcome.config
    text_fields_value = ",".join(config.text_fields) if config.text_fields else ""

    command: List[str] = [
        "python",
        "-m",
        "knn.cli",
        "--task",
        "opinion",
        "--dataset",
        str(dataset),
        "--feature-space",
        feature_space,
        "--issues",
        selection.study.issue,
        "--participant-studies",
        selection.study.key,
        "--knn-metric",
        config.metric,
        "--knn-k",
        str(outcome.best_k),
        "--knn-k-sweep",
        str(outcome.best_k),
        "--out-dir",
        "<run_dir>",
    ]
    command.extend(["--knn-text-fields", text_fields_value])
    if config.word2vec_size is not None:
        command.extend(["--word2vec-size", str(config.word2vec_size)])
    if config.word2vec_window is not None:
        command.extend(["--word2vec-window", str(config.word2vec_window)])
    if config.word2vec_min_count is not None:
        command.extend(["--word2vec-min-count", str(config.word2vec_min_count)])
    if config.word2vec_epochs is not None:
        command.extend(["--word2vec-epochs", str(config.word2vec_epochs)])
    if config.word2vec_workers is not None:
        command.extend(["--word2vec-workers", str(config.word2vec_workers)])
    if config.sentence_transformer_model:
        command.extend(
            ["--sentence-transformer-model", config.sentence_transformer_model]
        )
    if config.sentence_transformer_device:
        command.extend(
            ["--sentence-transformer-device", config.sentence_transformer_device]
        )
    if config.sentence_transformer_batch_size is not None:
        command.extend(
            [
                "--sentence-transformer-batch-size",
                str(config.sentence_transformer_batch_size),
            ]
        )
    if config.sentence_transformer_normalize is not None:
        command.append(
            "--sentence-transformer-normalize"
            if config.sentence_transformer_normalize
            else "--sentence-transformer-no-normalize"
        )
    return _format_shell_command(command)


def _hyperparameter_report_intro(
    *,
    studies: Sequence[StudySpec],
    k_sweep: Sequence[int],
    feature_spaces: Sequence[str],
    sentence_model: Optional[str],
) -> List[str]:
    """
    Return the hyper-parameter README introduction.

    :param studies: Studies that participated in the sweep.
    :type studies: Sequence[StudySpec]
    :param k_sweep: Iterable of ``k`` values evaluated during the sweep stage.
    :type k_sweep: Sequence[int]
    :param feature_spaces: Ordered set of feature spaces included in the sweeps.
    :type feature_spaces: Sequence[str]
    :param sentence_model: Optional sentence-transformer identifier examined.
    :type sentence_model: Optional[str]
    :returns: Markdown introduction lines.
    :rtype: List[str]
    """

    lines: List[str] = [
        "# Hyper-Parameter Sweep Results",
        "",
        (
            "This catalog aggregates the grid-search results used to select the "
            "production KNN configurations. Each table lists the top "
            "configurations per study, ranked by validation accuracy (for the "
            "slate-ranking task) or validation MAE (for the opinion task)."
        ),
        "",
        "Key settings:",
    ]
    lines.append(
        "- Studies: "
        + ", ".join(study.label for study in studies)
        + " ("
        + ", ".join(study.key for study in studies)
        + ")"
    )
    if k_sweep:
        formatted = ", ".join(format_k(value) for value in k_sweep)
        lines.append(f"- k sweep: {formatted}")
    if feature_spaces:
        lines.append(
            "- Feature spaces: " + ", ".join(space.upper() for space in feature_spaces)
        )
    if sentence_model:
        lines.append(f"- Sentence-transformer baseline: `{sentence_model}`")
    lines.append("")
    lines.append(
        "Tables bold the configurations promoted to the finalize stage. "
        "Commands beneath each table reproduce the selected configuration."
    )
    lines.append(
        "Accuracy values reflect eligible-only accuracy on the validation split "
        "at the selected best k (per the configured k-selection method)."
    )
    lines.append("")
    return lines


def _hyperparameter_feature_rows(
    feature_space: str,
    per_study: Mapping[str, StudySelection],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Return table rows covering ``feature_space`` selections.

    :param feature_space: Feature space identifier such as ``tfidf`` or ``word2vec``.
    :type feature_space: str
    :param per_study: Mapping of study keys to their associated selections or metrics.
    :type per_study: Mapping[str, StudySelection]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[StudySpec]
    :returns: Markdown table rows describing selections.
    :rtype: List[str]
    """

    rows: List[str] = []
    for study in studies:
        selection = per_study.get(study.key)
        if selection is None:
            continue
        summary = _metric_summary_to_mapping(
            extract_metric_summary(selection.outcome.metrics or {})
        )
        text_info = _describe_text_fields(selection.config.text_fields)
        command = _knn_next_video_command(feature_space, selection)
        rows.append(
            _format_hyperparameter_row(
                study=study,
                summary=summary,
                selection=selection,
                text_info=text_info,
                command=command,
            )
        )
    return rows


def _format_hyperparameter_row(
    *,
    study: StudySpec,
    summary: Mapping[str, object],
    selection: StudySelection,
    text_info: str,
    command: Optional[str],
) -> str:
    """
    Format a Markdown table row summarising a sweep selection.

    :param study: Study metadata associated with the selection.
    :type study: StudySpec
    :param summary: Derived metrics extracted from the selection outcome.
    :type summary: Mapping[str, object]
    :param selection: Winning selection entry.
    :type selection: StudySelection
    :param text_info: Human-readable text field description.
    :type text_info: str
    :param command: Optional reproduction command string.
    :type command: Optional[str]
    :returns: Markdown row representing the selection.
    :rtype: str
    """

    accuracy_value = summary.get("accuracy")
    baseline_value = summary.get("baseline")
    delta_value = summary.get("delta")
    eligible = summary.get("eligible")
    metric = selection.config.metric
    row = (
        f"| **{study.label}** | {metric} | {text_info or '—'} | "
        f"{format_optional_float(accuracy_value)} | {format_optional_float(baseline_value)} | "
        f"{format_delta(delta_value)} | {format_k(selection.best_k)} | "
        f"{format_count(eligible)} |"
    )
    if command:
        row += f" `{command}` |"
    else:
        row += " — |"
    return row


def _hyperparameter_table_section(
    *,
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
    feature_spaces: Sequence[str],
) -> List[str]:
    """
    Produce the per-feature-space table summarising selections.

    :param selections: Nested mapping of feature spaces to study selections.
    :type selections: Mapping[str, Mapping[str, StudySelection]]
    :param studies: Sequence of studies targeted by the pipeline.
    :type studies: Sequence[StudySpec]
    :param feature_spaces: Ordered feature spaces considered for the sweep.
    :type feature_spaces: Sequence[str]
    :returns: Markdown section with tables summarising hyperparameter selections.
    :rtype: List[str]
    """

    lines: List[str] = [
        "",
        "## Slate-Ranking Sweep Leaders",
        "",
        "### Configuration Leaderboards",
        "",
    ]
    for feature_space in feature_spaces:
        per_study = selections.get(feature_space)
        if not per_study:
            continue
        lines.append(_feature_space_heading(feature_space))
        lines.append("")
        header = (
            "| Study | Metric | Text fields | Acc (best k) ↑ | Baseline ↑ | "
            "Δ vs baseline ↑ | Best k | Eligible | Command |"
        )
        divider = "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"
        lines.append(header)
        lines.append(divider)
        lines.extend(
            _hyperparameter_feature_rows(
                feature_space=feature_space,
                per_study=per_study,
                studies=studies,
            )
        )
        lines.append("")
    return lines


def _hyperparameter_leaderboard_section(
    sweep_outcomes: Sequence[SweepOutcome],
) -> List[str]:
    """
    Build a Markdown leaderboard summarising sweep outcomes.

    :param sweep_outcomes: Chronological list of sweep outcomes.
    :type sweep_outcomes: Sequence[SweepOutcome]
    :returns: Markdown lines covering the leaderboard.
    :rtype: List[str]
    """

    lines: List[str] = ["", "### Configuration Leaderboards", ""]
    if not sweep_outcomes:
        lines.append("No sweep outcomes were recorded for this run.")
        lines.append("")
        return lines

    header = (
        "| Order | Study | Feature space | Metric | Text fields | Acc (best k) ↑ | "
        "Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible |"
    )
    divider = (
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"
    )
    lines.append(header)
    lines.append(divider)
    ordered = sorted(
        sweep_outcomes,
        key=lambda outcome: outcome.order_index,
    )
    for outcome in ordered:
        summary = _metric_summary_to_mapping(
            extract_metric_summary(outcome.metrics or {})
        )
        text_info = _describe_text_fields(outcome.config.text_fields)
        row = (
            f"| {outcome.order_index} | {outcome.study.label} | "
            f"{outcome.feature_space.upper()} | {outcome.config.metric} | {text_info or '—'} | "
            f"{format_optional_float(summary.get('accuracy'))} | "
            f"{format_optional_float(summary.get('baseline'))} | "
            f"{format_delta(summary.get('delta'))} | "
            f"{format_k(outcome.best_k)} | {format_count(summary.get('eligible'))} |"
        )
        lines.append(row)
    lines.append("")
    return lines


def _describe_text_fields(fields: Sequence[str]) -> str:
    """Return a human-readable description of concatenated text fields."""
    if not fields:
        return ""
    if len(fields) == 1:
        return fields[0]
    return ", ".join(fields[:-1]) + f", {fields[-1]}"


def _hyperparameter_observations_section(
    selections: Mapping[str, Mapping[str, StudySelection]],
) -> List[str]:
    """
    Generate qualitative observations for the slate-ranking sweeps.

    :param selections: Nested mapping of feature spaces to study selections.
    :type selections: Mapping[str, Mapping[str, StudySelection]]
    :returns: Bullet-point observations comparing selections across feature spaces.
    :rtype: List[str]
    """

    lines: List[str] = ["", "### Observations", ""]
    for feature_space, feature_selections in selections.items():
        if not feature_selections:
            continue
        details: List[str] = []
        reproductions: List[Tuple[str, str]] = []
        for selection in feature_selections.values():
            detail, reproduction = _build_observation_entry(feature_space, selection)
            if detail:
                details.append(detail)
            if reproduction:
                reproductions.append((selection.study.label, reproduction))
        if details:
            lines.append(f"- {feature_space.upper()}: " + "; ".join(details) + ".")
            for label, reproduction in reproductions:
                lines.append(f"  Command ({label}): `{reproduction}`")
    lines.append("")
    return lines


def _build_observation_entry(
    feature_space: str, selection: StudySelection
) -> Tuple[str, Optional[str]]:
    """Return observation detail text and optional reproduction command."""
    summary = _metric_summary_to_mapping(
        extract_metric_summary(selection.outcome.metrics or {})
    )
    accuracy_value = summary.get("accuracy")
    baseline_value = summary.get("baseline")
    delta_value = summary.get("delta")
    config = selection.config
    text_info = _describe_text_fields(config.text_fields)
    text_suffix = f" with {text_info}" if text_info else ""
    if feature_space == "word2vec":
        config_bits = (
            f"word2vec ({config.word2vec_size}d, window {config.word2vec_window}, "
            f"min_count {config.word2vec_min_count}){text_suffix}"
        )
    elif feature_space == "sentence_transformer":
        config_bits = (
            f"sentence-transformer `{config.sentence_transformer_model}`{text_suffix}"
        )
    else:
        config_bits = f"{config.metric} distance{text_suffix}"
    detail = (
        f"{selection.study.label}: accuracy {format_optional_float(accuracy_value)} "
        f"(baseline {format_optional_float(baseline_value)}, "
        f"Δ {format_delta(delta_value)}, "
        f"k={format_k(summary.get('best_k') or selection.best_k)}) "
        f"using {config_bits}"
    )
    return detail, _knn_next_video_command(feature_space, selection)


def _hyperparameter_opinion_section(
    *,
    opinion_selections: Mapping[str, Mapping[str, OpinionStudySelection]],
    opinion_sweep_outcomes: Sequence[OpinionSweepOutcome],
    studies: Sequence[StudySpec],
    feature_spaces: Sequence[str],
    allow_incomplete: bool,
) -> List[str]:
    """
    Render the opinion-regression sweep summary.

    :param opinion_selections: Mapping of feature spaces to their chosen opinion selections.
    :type opinion_selections: Mapping[str, Mapping[str, OpinionStudySelection]]
    :param opinion_sweep_outcomes: Chronological list of opinion sweep outcomes.
    :type opinion_sweep_outcomes: Sequence[OpinionSweepOutcome]
    :param studies: Sequence of study specifications targeted by the workflow.
    :type studies: Sequence[StudySpec]
    :param feature_spaces: Ordered collection of feature space names under consideration.
    :type feature_spaces: Sequence[str]
    :param allow_incomplete: Whether missing sweeps should surface placeholders.
    :type allow_incomplete: bool
    :returns: Markdown section covering opinion hyper-parameter sweeps.
    :rtype: List[str]
    """

    lines: List[str] = ["", "## Post-Study Opinion Regression", ""]
    if not opinion_sweep_outcomes:
        lines.append("No opinion sweeps were available when this report was generated.")
        if allow_incomplete:
            lines.append(
                "Run the KNN pipeline with `--stage sweeps` or `--stage full` once "
                "artifacts are ready."
            )
        lines.append("")
        return lines

    lines.append(
        "Configurations are ranked by validation MAE (lower is better). "
        "Bold rows indicate the selections promoted to the finalize stage."
    )
    lines.append("")

    grouped_outcomes = _group_opinion_outcomes(opinion_sweep_outcomes)
    portfolio = _OpinionPortfolioStats()
    top_n = 5
    for feature_space in _ordered_opinion_spaces(feature_spaces, grouped_outcomes):
        context = OpinionFeatureContext(
            selection_map=opinion_selections.get(feature_space, {}),
            studies=studies,
            top_n=top_n,
            portfolio=portfolio,
        )
        section_lines = _opinion_feature_section_lines(
            feature_space=feature_space,
            study_outcomes=grouped_outcomes.get(feature_space, {}),
            context=context,
        )
        if section_lines:
            lines.extend(section_lines)

    lines.extend(portfolio.to_lines())
    lines.append("### Opinion Reproduction Commands")
    lines.append("")
    lines.extend(_opinion_reproduction_lines(feature_spaces, opinion_selections))
    return lines


def _group_opinion_outcomes(
    opinion_sweep_outcomes: Sequence[OpinionSweepOutcome],
) -> Dict[str, Dict[str, List[OpinionSweepOutcome]]]:
    """Group opinion sweep outcomes by feature space and study."""
    grouped: Dict[str, Dict[str, List[OpinionSweepOutcome]]] = {}
    for outcome in opinion_sweep_outcomes:
        feature_bucket = grouped.setdefault(outcome.feature_space, {})
        feature_bucket.setdefault(outcome.study.key, []).append(outcome)
    return grouped


def _ordered_opinion_spaces(
    feature_spaces: Sequence[str],
    grouped_outcomes: Mapping[str, Mapping[str, Sequence[OpinionSweepOutcome]]],
) -> List[str]:
    """Return feature spaces sorted according to pipeline preference."""
    ordered = [space for space in feature_spaces if space in grouped_outcomes]
    ordered.extend(space for space in grouped_outcomes if space not in ordered)
    return ordered


@dataclass(frozen=True)
class OpinionFeatureContext:
    """Reusable bundle of context needed for opinion feature sections."""

    selection_map: Mapping[str, OpinionStudySelection]
    studies: Sequence[StudySpec]
    top_n: int
    portfolio: _OpinionPortfolioStats


def _opinion_feature_section_lines(
    *,
    feature_space: str,
    study_outcomes: Mapping[str, Sequence[OpinionSweepOutcome]],
    context: OpinionFeatureContext,
) -> List[str]:
    """Render table lines for a single feature space."""
    if not study_outcomes:
        return []
    lines: List[str] = [_feature_space_heading(feature_space), ""]
    header = (
        "| Study | Metric | Text fields | Model | Vec size | Window | Min count | "
        "Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | "
        "MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |"
    )
    divider = (
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    lines.append(header)
    lines.append(divider)
    for study in context.studies:
        outcomes = study_outcomes.get(study.key, [])
        if not outcomes:
            continue
        rows = _opinion_rows_for_study(
            feature_space=feature_space,
            study=study,
            outcomes=outcomes,
            context=context,
        )
        lines.extend(rows)
    lines.append("")
    return lines


def _opinion_rows_for_study(
    *,
    feature_space: str,
    study: StudySpec,
    outcomes: Sequence[OpinionSweepOutcome],
    context: OpinionFeatureContext,
) -> List[str]:
    """Return formatted rows for a single study within a feature space."""
    selection = context.selection_map.get(study.key)
    limit = max(1, context.top_n)
    displayed = _select_displayed_outcomes(outcomes, selection, limit)
    rows: List[str] = []
    for outcome in displayed:
        summary_obj = extract_opinion_summary(outcome.metrics or {})
        context.portfolio.record(
            summary_obj, f"{feature_space.upper()} – {study.label}"
        )
        summary = _opinion_summary_to_mapping(summary_obj)
        rows.append(
            _format_opinion_sweep_row(
                study=study,
                outcome=outcome,
                summary=summary,
                selected=selection is not None and outcome is selection.outcome,
            )
        )
    return rows


def _select_displayed_outcomes(
    outcomes: Sequence[OpinionSweepOutcome],
    selection: Optional[OpinionStudySelection],
    limit: int,
) -> List[OpinionSweepOutcome]:
    """Return the subset of outcomes to display, ensuring the selection is present."""
    ordered = sorted(outcomes, key=_opinion_outcome_sort_key)
    displayed = ordered[:limit]
    if selection is not None and selection.outcome not in displayed:
        candidates = [*displayed, selection.outcome]
        displayed = sorted(candidates, key=_opinion_outcome_sort_key)[:limit]
    return displayed


def _opinion_outcome_sort_key(
    outcome: OpinionSweepOutcome,
) -> Tuple[float, float, float, int]:
    """Return the ranking key used for opinion sweep tables."""
    mae_value = outcome.mae if outcome.mae is not None else float("inf")
    rmse_value = outcome.rmse if outcome.rmse is not None else float("inf")
    r2_value = outcome.r2_score if outcome.r2_score is not None else float("-inf")
    return (mae_value, rmse_value, -r2_value, outcome.best_k)


def _string_or_dash(value: Optional[int]) -> str:
    """Format optional integer configuration values."""
    return str(value) if value is not None else "—"


def _format_opinion_sweep_row(
    *,
    study: StudySpec,
    outcome: OpinionSweepOutcome,
    summary: Mapping[str, object],
    selected: bool,
) -> str:
    """Format a single opinion sweep outcome table row."""
    text_info = _describe_text_fields(outcome.config.text_fields)
    study_label = f"**{study.label}**" if selected else study.label
    columns = [
        study_label,
        outcome.config.metric,
        text_info or "—",
        outcome.config.sentence_transformer_model or "—",
        _string_or_dash(outcome.config.word2vec_size),
        _string_or_dash(outcome.config.word2vec_window),
        _string_or_dash(outcome.config.word2vec_min_count),
        format_optional_float(summary.get("accuracy")),
        format_optional_float(summary.get("baseline")),
        format_delta(summary.get("delta")),
        format_k(outcome.best_k),
        format_count(summary.get("eligible")),
        format_optional_float(outcome.mae),
        format_delta(outcome.mae_delta),
        format_optional_float(outcome.rmse),
        format_optional_float(outcome.r2_score),
        format_count(summary.get("participants")),
    ]
    return "| " + " | ".join(columns) + " |"


def _opinion_reproduction_lines(
    feature_spaces: Sequence[str],
    opinion_selections: Mapping[str, Mapping[str, OpinionStudySelection]],
) -> List[str]:
    """Return bullet list reproductions for opinion selections."""
    lines: List[str] = []
    for feature_space in feature_spaces:
        selections_for_space = opinion_selections.get(feature_space, {})
        if not selections_for_space:
            continue
        lines.append(f"- {feature_space.upper()}:")
        for selection in selections_for_space.values():
            command = _knn_opinion_command(feature_space, selection)
            if command:
                lines.append(f"  - {selection.study.label}: `{command}`")
        lines.append("")
    return lines


def _build_hyperparameter_report(config: HyperparameterReportConfig) -> None:
    """
    Write the combined hyper-parameter README.

    :param config: Aggregated hyper-parameter selections and report options.
    :type config: HyperparameterReportConfig
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = config.output_dir / "README.md"

    common = config.common
    lines: List[str] = _hyperparameter_report_intro(
        studies=common.studies,
        k_sweep=common.k_sweep,
        feature_spaces=common.feature_spaces,
        sentence_model=common.sentence_model,
    )
    if config.next_video:
        lines.extend(
            _hyperparameter_table_section(
                selections=config.next_video.selections,
                studies=common.studies,
                feature_spaces=common.feature_spaces,
            )
        )
        lines.extend(
            _hyperparameter_observations_section(config.next_video.selections)
        )
        lines.extend(
            _hyperparameter_leaderboard_section(config.next_video.sweep_outcomes)
        )
    if config.opinion:
        lines.extend(
            _hyperparameter_opinion_section(
                opinion_selections=config.opinion.selections,
                opinion_sweep_outcomes=config.opinion.sweep_outcomes,
                studies=common.studies,
                feature_spaces=common.feature_spaces,
                allow_incomplete=config.allow_incomplete,
            )
        )
    lines.append("")
    readme_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["_build_hyperparameter_report"]
