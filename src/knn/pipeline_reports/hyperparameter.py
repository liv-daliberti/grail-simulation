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

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..pipeline_context import (
    MetricSummary,
    OpinionStudySelection,
    OpinionSweepOutcome,
    OpinionSummary,
    StudySelection,
    StudySpec,
    SweepOutcome,
)
from ..pipeline_utils import (
    extract_metric_summary,
    extract_opinion_summary,
    format_count,
    format_delta,
    format_k,
    format_optional_float,
)
from .opinion import _OpinionPortfolioStats
from .shared import _feature_space_heading, _format_shell_command


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
        "This catalog aggregates the grid-search results used to select the production "
        "KNN configurations. Each table lists the top configurations per study, ranked by "
        "validation accuracy (for the slate-ranking task) or validation MAE (for the opinion task).",
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
        lines.append(
            "| Study | Metric | Text fields | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | Command |"
        )
        lines.append(
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"
        )
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

    lines.append(
        "| Order | Study | Feature space | Metric | Text fields | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible |"
    )
    lines.append(
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"
    )
    ordered = sorted(
        sweep_outcomes,
        key=lambda outcome: outcome.order_index,
    )
    for outcome in ordered:
        summary = _metric_summary_to_mapping(
            extract_metric_summary(outcome.metrics or {})
        )
        selection = (
            outcome.study.key,
            outcome.feature_space,
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
    bullet_bits: List[str] = []
    command_refs: List[Tuple[str, str]] = []

    for feature_space, feature_selections in selections.items():
        if not feature_selections:
            continue
        bullet_bits.clear()
        command_refs.clear()
        for study_key, selection in feature_selections.items():
            study = selection.study
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
                f"{study.label}: accuracy {format_optional_float(accuracy_value)} "
                f"(baseline {format_optional_float(baseline_value)}, Δ {format_delta(delta_value)}, "
                f"k={format_k(summary.get('best_k') or selection.best_k)}) using {config_bits}"
            )
            bullet_bits.append(detail)
            reproduction = _knn_next_video_command(feature_space, selection)
            if reproduction:
                command_refs.append((study.label, reproduction))
        if bullet_bits:
            lines.append(f"- {feature_space.upper()}: " + "; ".join(bullet_bits) + ".")
            for label, reproduction in command_refs:
                lines.append(f"  Command ({label}): `{reproduction}`")
    lines.append("")
    return lines


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
                "Run the KNN pipeline with `--stage sweeps` or `--stage full` once artifacts are ready."
            )
        lines.append("")
        return lines

    lines.append(
        "Configurations are ranked by validation MAE (lower is better). "
        "Bold rows indicate the selections promoted to the finalize stage."
    )
    lines.append("")

    per_feature: Dict[str, Dict[str, List[OpinionSweepOutcome]]] = {}
    portfolio = _OpinionPortfolioStats()
    for outcome in opinion_sweep_outcomes:
        feature_bucket = per_feature.setdefault(outcome.feature_space, {})
        feature_bucket.setdefault(outcome.study.key, []).append(outcome)

    ordered_spaces: List[str] = [
        space for space in feature_spaces if space in per_feature
    ]
    for space in per_feature:
        if space not in ordered_spaces:
            ordered_spaces.append(space)

    top_n = 5
    for feature_space in ordered_spaces:
        study_outcomes = per_feature.get(feature_space, {})
        if not study_outcomes:
            continue
        lines.append(_feature_space_heading(feature_space))
        lines.append("")
        lines.append(
            "| Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |"
        )
        lines.append(
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for study in studies:
            outcomes = study_outcomes.get(study.key, [])
            if not outcomes:
                continue
            selection = opinion_selections.get(feature_space, {}).get(study.key)
            ordered = sorted(
                outcomes,
                key=lambda item: (item.mae, item.rmse, -item.r2_score, item.best_k),
            )
            display_limit = max(1, top_n)
            displayed = ordered[:display_limit]
            if selection is not None and selection.outcome not in displayed:
                displayed.append(selection.outcome)
                displayed = sorted(
                    displayed,
                    key=lambda item: (item.mae, item.rmse, -item.r2_score, item.best_k),
                )[:display_limit]

            for outcome in displayed:
                summary_obj = extract_opinion_summary(outcome.metrics or {})
                summary = _opinion_summary_to_mapping(summary_obj)
                study_cell = study.label
                metric_cell = outcome.config.metric
                text_info = _describe_text_fields(outcome.config.text_fields)
                model_cell = outcome.config.sentence_transformer_model or "—"
                vector_size = (
                    str(outcome.config.word2vec_size)
                    if outcome.config.word2vec_size is not None
                    else "—"
                )
                window = (
                    str(outcome.config.word2vec_window)
                    if outcome.config.word2vec_window is not None
                    else "—"
                )
                min_count = (
                    str(outcome.config.word2vec_min_count)
                    if outcome.config.word2vec_min_count is not None
                    else "—"
                )
                accuracy_value = summary.get("accuracy")
                baseline_value = summary.get("baseline")
                delta_value = summary.get("delta")
                participants = summary.get("participants")
                mae_value = outcome.mae
                rmse_value = outcome.rmse
                r2_value = outcome.r2_score
                mae_delta = outcome.mae_delta
                label = f"{feature_space.upper()} – {study.label}"
                portfolio.record(summary_obj, label)
                row = (
                    f"| {'**' + study_cell + '**' if selection and outcome is selection.outcome else study_cell} | "
                    f"{metric_cell} | {text_info or '—'} | {model_cell} | "
                    f"{vector_size} | {window} | {min_count} | "
                    f"{format_optional_float(accuracy_value)} | "
                    f"{format_optional_float(baseline_value)} | "
                    f"{format_delta(delta_value)} | "
                    f"{format_k(outcome.best_k)} | "
                    f"{format_count(summary.get('eligible'))} | "
                    f"{format_optional_float(mae_value)} | "
                    f"{format_delta(mae_delta)} | "
                    f"{format_optional_float(rmse_value)} | "
                    f"{format_optional_float(r2_value)} | "
                    f"{format_count(participants)} |"
                )
                lines.append(row)
        lines.append("")

    lines.extend(portfolio.to_lines())
    lines.append("### Opinion Reproduction Commands")
    lines.append("")
    for feature_space in feature_spaces:
        selections_for_space = opinion_selections.get(feature_space, {})
        if not selections_for_space:
            continue
        lines.append(f"- {feature_space.upper()}:")
        for study_key, selection in selections_for_space.items():
            command = _knn_opinion_command(feature_space, selection)
            if command:
                lines.append(f"  - {selection.study.label}: `{command}`")
        lines.append("")
    return lines


def _build_hyperparameter_report(
    *,
    output_dir: Path,
    selections: Mapping[str, Mapping[str, StudySelection]],
    sweep_outcomes: Sequence[SweepOutcome],
    studies: Sequence[StudySpec],
    k_sweep: Sequence[int],
    feature_spaces: Sequence[str],
    sentence_model: Optional[str],
    opinion_selections: Mapping[str, Mapping[str, OpinionStudySelection]],
    opinion_sweep_outcomes: Sequence[OpinionSweepOutcome],
    allow_incomplete: bool,
    include_next_video: bool,
    include_opinion: bool,
) -> None:
    """
    Write the combined hyper-parameter README.

    :param output_dir: Destination directory for ``README.md``.
    :type output_dir: Path
    :param selections: Nested mapping of next-video study selections.
    :type selections: Mapping[str, Mapping[str, StudySelection]]
    :param sweep_outcomes: Chronological collection of next-video sweep outcomes.
    :type sweep_outcomes: Sequence[SweepOutcome]
    :param studies: Studies targeted by the pipeline run.
    :type studies: Sequence[StudySpec]
    :param k_sweep: Iterable of ``k`` values evaluated during sweeps.
    :type k_sweep: Sequence[int]
    :param feature_spaces: Ordered list of feature spaces covered by the sweep.
    :type feature_spaces: Sequence[str]
    :param sentence_model: Baseline sentence-transformer identifier (if any).
    :type sentence_model: Optional[str]
    :param opinion_selections: Nested mapping of opinion study selections.
    :type opinion_selections: Mapping[str, Mapping[str, OpinionStudySelection]]
    :param opinion_sweep_outcomes: Chronological list of opinion sweep outcomes.
    :type opinion_sweep_outcomes: Sequence[OpinionSweepOutcome]
    :param allow_incomplete: Whether missing data should surface placeholders.
    :type allow_incomplete: bool
    :param include_next_video: Whether to render next-video sections.
    :type include_next_video: bool
    :param include_opinion: Whether to render opinion sections.
    :type include_opinion: bool
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / "README.md"

    lines: List[str] = _hyperparameter_report_intro(
        studies=studies,
        k_sweep=k_sweep,
        feature_spaces=feature_spaces,
        sentence_model=sentence_model,
    )
    if include_next_video:
        lines.extend(
            _hyperparameter_table_section(
                selections=selections,
                studies=studies,
                feature_spaces=feature_spaces,
            )
        )
        lines.extend(_hyperparameter_observations_section(selections))
        lines.extend(_hyperparameter_leaderboard_section(sweep_outcomes))
    if include_opinion:
        lines.extend(
            _hyperparameter_opinion_section(
                opinion_selections=opinion_selections,
                opinion_sweep_outcomes=opinion_sweep_outcomes,
                studies=studies,
                feature_spaces=feature_spaces,
                allow_incomplete=allow_incomplete,
            )
        )
    lines.append("")
    readme_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["_build_hyperparameter_report"]
