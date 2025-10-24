# pylint: disable=line-too-long,too-many-arguments,too-many-branches,too-many-lines,too-many-locals,too-many-statements
"""Report generation helpers for the modular KNN pipeline."""
from __future__ import annotations

import logging
import shlex
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from common.report_utils import extract_numeric_series

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt  # type: ignore[assignment]
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]

from .pipeline_context import (
    OpinionSummary,
    OpinionStudySelection,
    OpinionSweepOutcome,
    ReportBundle,
    StudySelection,
    StudySpec,
    SweepOutcome,
)
from .pipeline_utils import (
    extract_metric_summary,
    extract_opinion_summary,
    format_count,
    format_delta,
    format_float,
    format_k,
    format_optional_float,
    format_uncertainty_details,
    parse_ci,
    snake_to_title,
)

LOGGER = logging.getLogger("knn.pipeline.reports")


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


def _format_shell_command(bits: Sequence[str]) -> str:
    """Join CLI arguments into a shell-friendly command."""
    return " ".join(shlex.quote(str(bit)) for bit in bits if str(bit))


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
    k_sweep: str,
    feature_spaces: Sequence[str],
    sentence_model: Optional[str],
    *,
    include_next_video: bool,
    include_opinion: bool,
) -> List[str]:
    """
    Return the Markdown header introducing the hyperparameter report.

    :param k_sweep: Comma-separated list of ``k`` values evaluated during a sweep.

    :type k_sweep: str

    :param feature_spaces: Ordered collection of feature space names under consideration.

    :type feature_spaces: Sequence[str]

    :param sentence_model: SentenceTransformer model identifier referenced in the report.

    :type sentence_model: Optional[str]

    :returns: the Markdown header introducing the hyperparameter report

    :rtype: List[str]

    """
    feature_label = ", ".join(space.replace("_", "-").upper() for space in feature_spaces)
    lines = [
        "# KNN Hyperparameter Tuning Notes",
        "",
        "This document consolidates the selected grid searches for the KNN baselines.",
        "",
    ]
    if include_next_video:
        lines.extend(
            [
                "## Next-Video Prediction",
                "",
                f"The latest sweeps cover the {feature_label} feature spaces with:",
                f"- `k ∈ {{{k_sweep}}}`",
                "- Distance metrics: cosine and L2",
                "- Text-field augmentations: none, `viewer_profile,state_text`",
            ]
        )
        if "word2vec" in feature_spaces:
            lines.append(
                "- Word2Vec variants: vector size ∈ {128, 256}, window ∈ {5, 10}, min_count ∈ {1}"
            )
        if "sentence_transformer" in feature_spaces and sentence_model:
            lines.append(f"- Sentence-transformer model: `{sentence_model}`")
        lines.extend(
            [
                "",
                "| Feature space | Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    if include_opinion and not include_next_video:
        lines.append("Opinion-regression sweeps are summarised below.")
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

    :returns: table rows covering ``feature_space`` selections

    :rtype: List[str]

    """
    rows: List[str] = []
    for study in studies:
        selection = per_study.get(study.key)
        if not selection:
            continue
        rows.append(_format_hyperparameter_row(feature_space, study, selection))
    return rows

def _format_hyperparameter_row(
    feature_space: str,
    study: StudySpec,
    selection: StudySelection,
) -> str:
    """
    Format a Markdown table row summarising a sweep selection.

    :param feature_space: Feature space identifier such as ``tfidf`` or ``word2vec``.

    :type feature_space: str

    :param study: Study specification for the item currently being processed.

    :type study: StudySpec

    :param selection: Winning sweep selection for the current study.

    :type selection: StudySelection

    :returns: Markdown table row describing a single study/feature selection.

    :rtype: str

    """
    config = selection.config
    text_label = ",".join(config.text_fields) if config.text_fields else "none"
    size = str(config.word2vec_size) if config.word2vec_size is not None else "—"
    window = str(config.word2vec_window) if config.word2vec_window is not None else "—"
    min_count = str(config.word2vec_min_count) if config.word2vec_min_count is not None else "—"
    if feature_space == "sentence_transformer":
        model = config.sentence_transformer_model or "sentence-transformer"
    elif feature_space == "word2vec":
        model = "word2vec"
    else:
        model = "tfidf"
    summary = extract_metric_summary(selection.outcome.metrics or {})
    delta = (
        summary.accuracy - summary.baseline
        if summary.accuracy is not None and summary.baseline is not None
        else None
    )
    eligible = summary.n_eligible if summary.n_eligible is not None else selection.outcome.eligible
    return (
        f"| {feature_space.upper()} | {study.label} | {config.metric} | {text_label} | {model} | "
        f"{size} | {window} | {min_count} | {format_optional_float(summary.accuracy)} | "
        f"{format_optional_float(summary.baseline)} | {format_delta(delta)} | "
        f"{format_k(summary.best_k or selection.best_k)} | {format_count(eligible)} |"
    )

def _hyperparameter_table_section(
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Render the hyperparameter summary table for each feature space.

    :param selections: Mapping of feature spaces to their chosen study selections.

    :type selections: Mapping[str, Mapping[str, StudySelection]]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :returns: Markdown section with tables summarising hyperparameter selections.

    :rtype: List[str]

    """
    lines: List[str] = []
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in selections
    ]
    for space in selections:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
        per_study = selections.get(feature_space, {})
        lines.extend(_hyperparameter_feature_rows(feature_space, per_study, studies))
    lines.append("")
    return lines

def _hyperparameter_leaderboard_section(
    *,
    sweep_outcomes: Sequence[SweepOutcome],
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
    top_n: int,
) -> List[str]:
    """
    Return detailed leaderboards for the top-performing sweep configurations.

    :param sweep_outcomes: Iterable of sweep outcomes used to select final configurations.

    :type sweep_outcomes: Sequence[SweepOutcome]

    :param selections: Mapping of feature spaces to their chosen study selections.

    :type selections: Mapping[str, Mapping[str, StudySelection]]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :param top_n: Number of top items to retain when truncating the index.

    :type top_n: int

    :returns: detailed leaderboards for the top-performing sweep configurations

    :rtype: List[str]

    """
    if not sweep_outcomes:
        return []

    lines: List[str] = ["### Configuration Leaderboards", ""]
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in selections
    ]
    seen_spaces: set[str] = set(ordered_spaces)
    for outcome in sweep_outcomes:
        if outcome.feature_space not in seen_spaces:
            ordered_spaces.append(outcome.feature_space)
            seen_spaces.add(outcome.feature_space)

    per_feature: Dict[str, Dict[str, List[SweepOutcome]]] = {}
    for outcome in sweep_outcomes:
        per_feature.setdefault(outcome.feature_space, {}).setdefault(outcome.study.key, []).append(outcome)

    for feature_space in ordered_spaces:
        feature_outcomes = per_feature.get(feature_space)
        if not feature_outcomes:
            continue
        lines.append(_feature_space_heading(feature_space))
        lines.append("")
        for study in studies:
            study_outcomes = feature_outcomes.get(study.key, [])
            if not study_outcomes:
                continue
            ranked = sorted(
                study_outcomes,
                key=lambda item: (item.accuracy, item.eligible, -item.best_k),
                reverse=True,
            )
            top_results = ranked[: max(1, top_n)]
            selected = selections.get(feature_space, {}).get(study.key)
            selected_label = selected.config.label if selected else None
            best_accuracy = top_results[0].accuracy if top_results else 0.0
            lines.append(f"#### {study.label}")
            lines.append("")
            lines.append("| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |")
            lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
            for idx, outcome in enumerate(top_results, start=1):
                config_label = outcome.config.label()
                label_display = f"**{config_label}**" if config_label == selected_label else config_label
                delta = max(0.0, best_accuracy - outcome.accuracy)
                lines.append(
                    f"| {idx} | "
                    f"{label_display} | "
                    f"{format_float(outcome.accuracy)} | "
                    f"{format_float(delta)} | "
                    f"{outcome.best_k} | "
                    f"{outcome.eligible} |"
                )
            lines.append("")
        lines.append("")
    return lines

def _describe_text_fields(fields: Sequence[str]) -> str:
    """
    Return a readable description of text-field augmentations.

    :param fields: Iterable of dataset field names incorporated into the prompt.

    :type fields: Sequence[str]

    :returns: a readable description of text-field augmentations

    :rtype: str

    """
    if not fields:
        return "base prompt only"
    return ", ".join(snake_to_title(field) for field in fields)

def _hyperparameter_observations_section(
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Summarise key takeaways from the hyperparameter sweeps.

    :param selections: Mapping of feature spaces to their chosen study selections.

    :type selections: Mapping[str, Mapping[str, StudySelection]]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :returns: Markdown section containing narrative observations for the hyperparameter report.

    :rtype: List[str]

    """
    lines: List[str] = ["### Observations", ""]
    for feature_space in ("tfidf", "word2vec", "sentence_transformer"):
        per_feature = selections.get(feature_space)
        if not per_feature:
            continue
        bullet_bits: List[str] = []
        command_refs: List[Tuple[str, str]] = []
        for study in studies:
            selection = per_feature.get(study.key)
            if selection is None:
                continue
            summary = extract_metric_summary(selection.outcome.metrics or {})
            accuracy_value = summary.accuracy
            baseline_value = summary.baseline
            delta_value = (
                accuracy_value - baseline_value
                if accuracy_value is not None and baseline_value is not None
                else None
            )
            config = selection.config
            text_info = _describe_text_fields(config.text_fields)
            if feature_space == "word2vec":
                config_bits = (
                    f"word2vec ({config.word2vec_size}d, window {config.word2vec_window}, "
                    f"min_count {config.word2vec_min_count}) with {text_info}"
                )
            elif feature_space == "sentence_transformer":
                config_bits = f"sentence-transformer `{config.sentence_transformer_model}` with {text_info}"
            else:
                config_bits = f"{config.metric} distance with {text_info}"
            detail = (
                f"{study.label}: accuracy {format_optional_float(accuracy_value)} "
                f"(baseline {format_optional_float(baseline_value)}, Δ {format_delta(delta_value)}, "
                f"k={format_k(summary.best_k or selection.best_k)}) using {config_bits}"
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
                "Run the KNN pipeline with `--stage sweeps` or `--stage full` once artefacts are ready."
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
            "| Study | Metric | Text fields | Model | Vec size | Window | Min count | Best k | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |"
        )
        lines.append(
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for study in studies:
            outcomes = study_outcomes.get(study.key, [])
            if not outcomes:
                continue
            selection = opinion_selections.get(feature_space, {}).get(study.key)
            ordered = sorted(
                outcomes,
                key=lambda item: (item.mae, item.rmse, -item.r2, item.best_k),
            )
            display_limit = max(1, top_n)
            displayed = ordered[:display_limit]
            if selection is not None and selection.outcome not in displayed:
                displayed.append(selection.outcome)
                displayed = sorted(
                    displayed,
                    key=lambda item: (item.mae, item.rmse, -item.r2, item.best_k),
                )[:display_limit]

            for outcome in displayed:
                summary = extract_opinion_summary(outcome.metrics or {})
                study_cell = study.label
                if selection and outcome.config == selection.outcome.config:
                    study_cell = f"**{study_cell}**"
                config = outcome.config
                text_label = ",".join(config.text_fields) if config.text_fields else "none"
                if feature_space == "sentence_transformer":
                    model = config.sentence_transformer_model or "sentence-transformer"
                elif feature_space == "word2vec":
                    model = "word2vec"
                else:
                    model = "tfidf"
                size = (
                    str(config.word2vec_size)
                    if config.word2vec_size is not None
                    else "—"
                )
                window = (
                    str(config.word2vec_window)
                    if config.word2vec_window is not None
                    else "—"
                )
                min_count = (
                    str(config.word2vec_min_count)
                    if config.word2vec_min_count is not None
                    else "—"
                )
                best_k = summary.best_k if summary.best_k is not None else outcome.best_k
                participants = (
                    summary.participants
                    if summary.participants is not None
                    else outcome.participants
                )
                lines.append(
                    f"| {study_cell} | {config.metric} | {text_label} | {model} | "
                    f"{size} | {window} | {min_count} | {format_k(best_k)} | "
                    f"{format_optional_float(summary.mae)} | "
                    f"{format_delta(summary.mae_delta)} | "
                    f"{format_optional_float(summary.rmse)} | "
                    f"{format_optional_float(summary.r2)} | "
                    f"{format_count(participants)} |"
                )
            if len(ordered) > display_limit:
                lines.append(
                    f"*Showing top {display_limit} of {len(ordered)} configurations for {study.label}.*"
                )
        lines.append("")

        selection_map = opinion_selections.get(feature_space, {})
        if selection_map:
            bullet_bits: List[str] = []
            for study in studies:
                selection = selection_map.get(study.key)
                if selection is None:
                    continue
                summary = extract_opinion_summary(selection.outcome.metrics or {})
                portfolio.record(
                    summary, f"{study.label} ({feature_space.upper()})"
                )
                bullet_bits.append(
                    f"{study.label}: MAE {format_optional_float(summary.mae)} "
                    f"(Δ {format_delta(summary.mae_delta)}, k={format_k(summary.best_k or selection.outcome.best_k)})"
                )
            if bullet_bits:
                lines.append(
                    f"- **{feature_space.upper()} selections**: " + "; ".join(bullet_bits) + "."
                )
                lines.append("")
                for study in studies:
                    selection = selection_map.get(study.key)
                    if selection is None:
                        continue
                    reproduction = _knn_opinion_command(feature_space, selection)
                    if reproduction:
                        lines.append(f"  Command ({study.label}): `{reproduction}`")
                lines.append("")
        else:
            # Fall back to the top-ranked configuration for aggregation.
            for study in studies:
                outcomes = study_outcomes.get(study.key, [])
                if not outcomes:
                    continue
                summary = extract_opinion_summary(outcomes[0].metrics or {})
                portfolio.record(
                    summary, f"{study.label} ({feature_space.upper()})"
                )
    lines.extend(portfolio.to_lines())
    return lines

def _feature_space_heading(feature_space: str) -> str:
    """
    Return the Markdown heading for ``feature_space``.

    :param feature_space: Feature space identifier such as ``tfidf`` or ``word2vec``.

    :type feature_space: str

    :returns: the Markdown heading for ``feature_space``

    :rtype: str

    """
    if feature_space == "tfidf":
        return "## TF-IDF Feature Space"
    if feature_space == "word2vec":
        return "## Word2Vec Feature Space"
    if feature_space == "sentence_transformer":
        return "## Sentence-Transformer Feature Space"
    return f"## {feature_space.replace('_', ' ').title()} Feature Space"

def _next_video_dataset_info(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Tuple[str, str]:
    """
    Extract a representative dataset name and split from metrics payloads.

    :param metrics_by_feature: Nested mapping of metrics grouped by feature space and study.

    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]

    :returns: Collection of Markdown lines describing the next-video dataset and splits.

    :rtype: Tuple[str, str]

    """
    for per_feature in metrics_by_feature.values():
        for study_metrics in per_feature.values():
            dataset = study_metrics.get("dataset")
            split = study_metrics.get("split", "validation")
            if dataset:
                return str(dataset), str(split)
    raise RuntimeError("No slate metrics available to build the next-video report.")

def _next_video_uncertainty_info(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Optional[Mapping[str, object]]:
    """
    Return the first uncertainty payload available for reporting.

    :param metrics_by_feature: Nested mapping of metrics grouped by feature space and study.

    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]

    :returns: the first uncertainty payload available for reporting

    :rtype: Optional[Mapping[str, object]]

    """
    for per_feature in metrics_by_feature.values():
        for study_metrics in per_feature.values():
            uncertainty = study_metrics.get("uncertainty")
            if isinstance(uncertainty, Mapping):
                return uncertainty
    return None

def _next_video_intro(
    dataset_name: str,
    split: str,
    uncertainty: Optional[Mapping[str, object]] = None,
) -> List[str]:
    """
    Return the introductory Markdown section for the next-video report.

    :param dataset_name: Human-readable label for the dataset being summarised.

    :type dataset_name: str

    :param split: Dataset split identifier such as ``train`` or ``validation``.

    :type split: str

    :param uncertainty: Auxiliary uncertainty information accompanying a metric.

    :type uncertainty: Optional[Mapping[str, object]]

    :returns: the introductory Markdown section for the next-video report

    :rtype: List[str]

    """
    intro = [
        "# KNN Next-Video Baseline",
        "",
        "This report summarises the slate-ranking KNN model that predicts the next video a viewer will click.",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Split: {split}",
        "- Metric: accuracy on eligible slates (gold index present)",
        "- Baseline column: accuracy from always recommending the most-frequent gold index for the study.",
        "- Δ column: improvement over that baseline accuracy.",
        "- Random column: expected accuracy from uniformly sampling one candidate per slate.",
    ]
    if uncertainty:
        intro.append(
            "- Uncertainty: "
            + str(uncertainty.get("method", "unknown"))
            + format_uncertainty_details(uncertainty)
        )
    intro.append("")
    return intro


def _next_video_portfolio_summary(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    feature_spaces: Sequence[str],
) -> List[str]:
    """
    Render weighted portfolio statistics across feature spaces.

    :param metrics_by_feature: Nested mapping of metrics grouped by feature space.
    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param feature_spaces: Ordered feature spaces considered by the pipeline.
    :type feature_spaces: Sequence[str]
    :returns: Markdown lines summarising portfolio-level performance.
    :rtype: List[str]
    """
    ordered_spaces = [
        space for space in ("tfidf", "word2vec", "sentence_transformer") if space in feature_spaces
    ]
    for space in feature_spaces:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for space in metrics_by_feature:
        if space not in ordered_spaces:
            ordered_spaces.append(space)

    rows: List[str] = []
    best_feature: Optional[str] = None
    best_accuracy: Optional[float] = None
    best_eligible: int = 0
    best_study_count: int = 0

    for feature_space in ordered_spaces:
        per_feature = metrics_by_feature.get(feature_space, {})
        if not per_feature:
            continue
        accuracy_total = 0.0
        accuracy_weight = 0
        baseline_total = 0.0
        baseline_weight = 0
        random_total = 0.0
        random_weight = 0
        studies_with_metrics = 0
        for data in per_feature.values():
            summary = extract_metric_summary(data)
            if summary.n_eligible is None or summary.n_eligible <= 0:
                continue
            eligible = summary.n_eligible
            recorded = False
            if summary.accuracy is not None:
                accuracy_total += summary.accuracy * eligible
                accuracy_weight += eligible
                recorded = True
            if summary.baseline is not None:
                baseline_total += summary.baseline * eligible
                baseline_weight += eligible
                recorded = True
            if summary.random_baseline is not None:
                random_total += summary.random_baseline * eligible
                random_weight += eligible
                recorded = True
            if recorded:
                studies_with_metrics += 1
        if not studies_with_metrics:
            continue
        weighted_accuracy = accuracy_total / accuracy_weight if accuracy_weight else None
        weighted_baseline = baseline_total / baseline_weight if baseline_weight else None
        weighted_random = random_total / random_weight if random_weight else None
        eligible_total = max(accuracy_weight, baseline_weight, random_weight)
        delta_value: Optional[float] = None
        if weighted_accuracy is not None and weighted_baseline is not None:
            delta_value = weighted_accuracy - weighted_baseline
        rows.append(
            f"| {feature_space.upper()} | {format_optional_float(weighted_accuracy)} | "
            f"{format_delta(delta_value)} | {format_optional_float(weighted_random)} | "
            f"{format_count(eligible_total)} | {format_count(studies_with_metrics)} |"
        )
        if weighted_accuracy is not None:
            if best_accuracy is None or weighted_accuracy > best_accuracy:
                best_accuracy = weighted_accuracy
                best_feature = feature_space
                best_eligible = eligible_total
                best_study_count = studies_with_metrics

    if not rows:
        return []

    lines: List[str] = ["## Portfolio Summary", ""]
    lines.append(
        "| Feature space | Weighted accuracy ↑ | Δ vs baseline ↑ | Random ↑ | Eligible | Studies |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.extend(rows)
    lines.append("")
    if best_feature is not None and best_accuracy is not None:
        lines.append(
            f"Best-performing feature space: **{best_feature.upper()}** with weighted accuracy "
            f"{format_optional_float(best_accuracy)} across {format_count(best_eligible)} eligible slates "
            f"({format_count(best_study_count)} studies)."
        )
        lines.append("")
    return lines

def _format_ci(ci_value: object) -> str:
    """
    Format a 95% confidence interval if present.

    :param ci_value: Confidence-interval payload extracted from a metrics dictionary.

    :type ci_value: object

    :returns: Human-readable confidence-interval string formatted for Markdown tables.

    :rtype: str

    """
    confidence_interval = parse_ci(ci_value)
    if confidence_interval is None:
        return "—"
    low, high = confidence_interval
    return f"[{low:.3f}, {high:.3f}]"

def _extract_curve_series(curve_block: Mapping[str, object]) -> Tuple[List[int], List[float]]:
    """
    Return sorted k/accuracy pairs extracted from ``curve_block``.

    :param curve_block: Markdown block describing a single KNN performance curve.

    :type curve_block: Mapping[str, object]

    :returns: sorted k/accuracy pairs extracted from ``curve_block``

    :rtype: Tuple[List[int], List[float]]

    """
    accuracy_map = curve_block.get("accuracy_by_k")
    if not isinstance(accuracy_map, Mapping):
        return ([], [])
    return extract_numeric_series(accuracy_map)

def _plot_knn_curve_bundle(
    *,
    base_dir: Path,
    feature_space: str,
    study: StudySpec,
    metrics: Mapping[str, object],
) -> Optional[str]:
    """
    Save a train/validation accuracy curve plot for ``study`` when possible.

    :param base_dir: Base directory that contains task-specific output subdirectories.

    :type base_dir: Path

    :param feature_space: Feature space identifier such as ``tfidf`` or ``word2vec``.

    :type feature_space: str

    :param study: Study specification for the item currently being processed.

    :type study: StudySpec

    :param metrics: Metrics dictionary captured from a previous pipeline stage.

    :type metrics: Mapping[str, object]

    :returns: Matplotlib figure handle for the rendered KNN performance curves.

    :rtype: Optional[str]

    """
    if plt is None:  # pragma: no cover - optional dependency
        return None
    curve_bundle = metrics.get("curve_metrics")
    if not isinstance(curve_bundle, Mapping):
        return None
    eval_curve = curve_bundle.get("eval")
    if not isinstance(eval_curve, Mapping):
        return None
    eval_x, eval_y = _extract_curve_series(eval_curve)
    if not eval_x:
        return None
    train_curve = curve_bundle.get("train")
    train_x: List[int] = []
    train_y: List[float] = []
    if isinstance(train_curve, Mapping):
        train_x, train_y = _extract_curve_series(train_curve)

    curves_dir = base_dir / "curves" / feature_space
    curves_dir.mkdir(parents=True, exist_ok=True)
    plot_path = curves_dir / f"{study.study_slug}.png"

    fig, axis = plt.subplots(figsize=(6, 3.5))  # type: ignore[attr-defined]
    axis.plot(eval_x, eval_y, marker="o", label="validation")
    if train_x and train_y:
        axis.plot(train_x, train_y, marker="o", linestyle="--", label="training")
    axis.set_title(f"{study.label} – {feature_space.upper()}")
    axis.set_xlabel("k")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(eval_x)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)  # type: ignore[attr-defined]
    plt.close(fig)  # type: ignore[attr-defined]
    try:
        return plot_path.relative_to(base_dir).as_posix()
    except ValueError:
        return plot_path.as_posix()

def _next_video_feature_section(
    feature_space: str,
    metrics: Mapping[str, Mapping[str, object]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Render the next-video metrics table for ``feature_space``.

    :param feature_space: Feature space identifier such as ``tfidf`` or ``word2vec``.

    :type feature_space: str

    :param metrics: Metrics dictionary captured from a previous pipeline stage.

    :type metrics: Mapping[str, Mapping[str, object]]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :returns: Markdown section summarising feature-space results for next-video evaluation.

    :rtype: List[str]

    """
    if not metrics:
        return []
    lines: List[str] = [
        _feature_space_heading(feature_space),
        "",
        "| Study | Accuracy ↑ | 95% CI | Δ vs baseline ↑ | Baseline ↑ | Random ↑ | Best k | Eligible | Total |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for study in studies:
        data = metrics.get(study.key)
        if not data:
            continue
        summary = extract_metric_summary(data)
        delta = (
            summary.accuracy - summary.baseline
            if summary.accuracy is not None and summary.baseline is not None
            else None
        )
        lines.append(
            f"| {study.label} | {format_optional_float(summary.accuracy)} | "
            f"{_format_ci(summary.accuracy_ci)} | {format_delta(delta)} | "
            f"{format_optional_float(summary.baseline)} | "
            f"{format_optional_float(summary.random_baseline)} | "
            f"{format_k(summary.best_k)} | {format_count(summary.n_eligible)} | "
            f"{format_count(summary.n_total)} |"
        )
    lines.append("")
    return lines

def _next_video_curve_sections(
    *,
    output_dir: Path,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Render Markdown sections embedding train/validation accuracy curves.

    :param output_dir: Directory where the rendered report should be written.

    :type output_dir: Path

    :param metrics_by_feature: Nested mapping of metrics grouped by feature space and study.

    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :returns: Markdown sections that document next-video learning curves.

    :rtype: List[str]

    """
    if plt is None:  # pragma: no cover - optional dependency
        LOGGER.debug("Matplotlib not available; skipping KNN curve plots.")
        return []

    sections: List[str] = []
    ordered_spaces = [
        space for space in ("tfidf", "word2vec", "sentence_transformer") if space in metrics_by_feature
    ]
    for feature_space in metrics_by_feature:
        if feature_space not in ordered_spaces:
            ordered_spaces.append(feature_space)

    for feature_space in ordered_spaces:
        feature_metrics = metrics_by_feature.get(feature_space, {})
        if not feature_metrics:
            continue
        image_lines: List[str] = []
        for study in studies:
            study_metrics = feature_metrics.get(study.key)
            if not study_metrics:
                continue
            rel_path = _plot_knn_curve_bundle(
                base_dir=output_dir,
                feature_space=feature_space,
                study=study,
                metrics=study_metrics,
            )
            if rel_path:
                image_lines.extend(
                    [
                        f"### {study.label} ({feature_space.upper()})",
                        "",
                        f"![Accuracy curve]({rel_path})",
                        "",
                    ]
                )
        sections.extend(image_lines)
    return sections

def _next_video_observations(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Summarise per-feature observations for next-video metrics.

    :param metrics_by_feature: Nested mapping of metrics grouped by feature space and study.

    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :returns: Markdown bullet list capturing qualitative next-video observations.

    :rtype: List[str]

    """
    lines: List[str] = ["## Observations", ""]
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in metrics_by_feature
    ]
    for space in metrics_by_feature:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
        metrics = metrics_by_feature.get(feature_space, {})
        if not metrics:
            continue
        bullet_bits: List[str] = []
        deltas: List[float] = []
        randoms: List[float] = []
        for study in studies:
            data = metrics.get(study.key)
            if not data:
                continue
            summary = extract_metric_summary(data)
            if summary.accuracy is None:
                continue
            delta_val = (
                summary.accuracy - summary.baseline
                if summary.baseline is not None
                else None
            )
            if delta_val is not None:
                deltas.append(delta_val)
            if summary.random_baseline is not None:
                randoms.append(summary.random_baseline)
            detail = (
                f"{study.label}: {format_optional_float(summary.accuracy)} "
                f"(baseline {format_optional_float(summary.baseline)}, "
                f"Δ {format_delta(delta_val)}, k={format_k(summary.best_k)}, "
                f"eligible {format_count(summary.n_eligible)})"
            )
            bullet_bits.append(detail)
        extras: List[str] = []
        if deltas:
            mean_delta = sum(deltas) / len(deltas)
            extras.append(f"mean Δ {format_delta(mean_delta)}")
        if randoms:
            mean_random = sum(randoms) / len(randoms)
            extras.append(f"mean random {format_optional_float(mean_random)}")
        if extras:
            bullet_bits.append("averages: " + ", ".join(extras))
        if bullet_bits:
            lines.append(f"- {feature_space.upper()}: " + "; ".join(bullet_bits) + ".")
    lines.append("- Random values correspond to the expected accuracy from a uniform guess across the slate options.")
    lines.append("")
    return lines

def _next_video_loso_section(
    loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """
    Return a section summarising leave-one-study-out accuracy.

    :param loso_metrics: Leave-one-study-out metrics grouped by feature space and holdout.

    :type loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :returns: a section summarising leave-one-study-out accuracy

    :rtype: List[str]

    """
    if not loso_metrics:
        return []
    lines: List[str] = ["## Cross-Study Holdouts", ""]
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in loso_metrics
    ]
    for space in loso_metrics:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
        metrics = loso_metrics.get(feature_space, {})
        if not metrics:
            continue
        lines.append(_feature_space_heading(feature_space))
        lines.append("")
        lines.append("| Holdout study | Accuracy ↑ | Δ vs baseline ↑ | Baseline ↑ | Best k | Eligible |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for study in studies:
            data = metrics.get(study.key)
            if not data:
                continue
            summary = extract_metric_summary(data)
            delta = (
                summary.accuracy - summary.baseline
                if summary.accuracy is not None and summary.baseline is not None
                else None
            )
            lines.append(
                f"| {study.label} | {format_optional_float(summary.accuracy)} | "
                f"{format_delta(delta)} | {format_optional_float(summary.baseline)} | "
                f"{format_k(summary.best_k)} | {format_count(summary.n_eligible)} |"
            )
        lines.append("")
    return lines

def _build_hyperparameter_report(
    *,
    output_dir: Path,
    selections: Mapping[str, Mapping[str, StudySelection]],
    sweep_outcomes: Sequence[SweepOutcome],
    studies: Sequence[StudySpec],
    k_sweep: str,
    feature_spaces: Sequence[str],
    sentence_model: Optional[str],
    opinion_selections: Mapping[str, Mapping[str, OpinionStudySelection]] | None = None,
    opinion_sweep_outcomes: Sequence[OpinionSweepOutcome] = (),
    allow_incomplete: bool = False,
    include_next_video: bool = True,
    include_opinion: bool = True,
) -> None:
    """
    Write the hyperparameter tuning summary under ``output_dir``.

    :param output_dir: Directory where the rendered report should be written.

    :type output_dir: Path

    :param selections: Mapping of feature spaces to their chosen study selections.

    :type selections: Mapping[str, Mapping[str, StudySelection]]

    :param sweep_outcomes: Iterable of sweep outcomes used to select final configurations.

    :type sweep_outcomes: Sequence[SweepOutcome]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :param k_sweep: Comma-separated list of ``k`` values evaluated during a sweep.

    :type k_sweep: str

    :param feature_spaces: Ordered collection of feature space names under consideration.

    :type feature_spaces: Sequence[str]

    :param sentence_model: SentenceTransformer model identifier referenced in the report.

    :type sentence_model: Optional[str]

    :returns: None.

    :rtype: None

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "README.md"
    lines: List[str] = []
    lines.extend(
        _hyperparameter_report_intro(
            k_sweep,
            feature_spaces,
            sentence_model,
            include_next_video=include_next_video,
            include_opinion=include_opinion,
        )
    )
    if include_next_video:
        if selections:
            lines.extend(_hyperparameter_table_section(selections, studies))
        if sweep_outcomes:
            lines.extend(
                _hyperparameter_leaderboard_section(
                    sweep_outcomes=sweep_outcomes,
                    selections=selections,
                    studies=studies,
                    top_n=3,
                )
            )
        if selections:
            lines.extend(_hyperparameter_observations_section(selections, studies))
        if not selections and allow_incomplete:
            lines.append(
                "Next-video sweeps were not available when this report was generated."
            )
            lines.append("")
    if include_opinion:
        lines.extend(
            _hyperparameter_opinion_section(
                opinion_selections=opinion_selections or {},
                opinion_sweep_outcomes=opinion_sweep_outcomes,
                studies=studies,
                feature_spaces=feature_spaces,
                allow_incomplete=allow_incomplete,
            )
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")

def _build_next_video_report(
    *,
    output_dir: Path,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
    feature_spaces: Sequence[str],
    loso_metrics: Optional[Mapping[str, Mapping[str, Mapping[str, object]]]] = None,
    allow_incomplete: bool = False,
) -> None:
    """
    Compose the next-video evaluation report under ``output_dir``.

    :param output_dir: Directory where the rendered report should be written.

    :type output_dir: Path

    :param metrics_by_feature: Nested mapping of metrics grouped by feature space and study.

    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :param feature_spaces: Ordered collection of feature space names under consideration.

    :type feature_spaces: Sequence[str]

    :param loso_metrics: Leave-one-study-out metrics grouped by feature space and holdout.

    :type loso_metrics: Optional[Mapping[str, Mapping[str, Mapping[str, object]]]]

    :param allow_incomplete: Whether processing may continue when some sweep data is missing.

    :type allow_incomplete: bool

    :returns: None.

    :rtype: None

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "README.md"
    if not metrics_by_feature:
        if not allow_incomplete:
            raise RuntimeError("No slate metrics available to build the next-video report.")
        placeholder = [
            "# KNN Next-Video Baseline",
            "",
            "Final slate metrics are not available yet. Rerun the pipeline with `--stage=finalize` once sweeps finish.",
            "",
            "This placeholder was generated with `--allow-incomplete` enabled.",
            "",
        ]
        output_path.write_text("\n".join(placeholder), encoding="utf-8")
        return

    dataset_name, split = _next_video_dataset_info(metrics_by_feature)
    uncertainty = _next_video_uncertainty_info(metrics_by_feature)
    lines: List[str] = []
    lines.extend(_next_video_intro(dataset_name, split, uncertainty))
    lines.extend(
        _next_video_portfolio_summary(
            metrics_by_feature=metrics_by_feature,
            feature_spaces=feature_spaces,
        )
    )
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in feature_spaces
    ]
    for space in feature_spaces:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
        metrics = metrics_by_feature.get(feature_space, {})
        lines.extend(_next_video_feature_section(feature_space, metrics, studies))
    curve_sections = _next_video_curve_sections(
        output_dir=output_dir,
        metrics_by_feature=metrics_by_feature,
        studies=studies,
    )
    if curve_sections:
        lines.extend(curve_sections)
    lines.extend(_next_video_observations(metrics_by_feature, studies))
    if loso_metrics:
        lines.extend(_next_video_loso_section(loso_metrics, studies))
    output_path.write_text("\n".join(lines), encoding="utf-8")

def _opinion_report_intro(dataset_name: str, split: str) -> List[str]:
    """
    Return the introductory Markdown section for the opinion report.

    :param dataset_name: Human-readable label for the dataset being summarised.

    :type dataset_name: str

    :param split: Dataset split identifier such as ``train`` or ``validation``.

    :type split: str

    :returns: the introductory Markdown section for the opinion report

    :rtype: List[str]

    """
    return [
        "# KNN Opinion Shift Study",
        "",
        "This study evaluates a second KNN baseline that predicts each participant's post-study opinion index.",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Split: {split}",
        "- Metrics: MAE / RMSE / R² on the predicted post index, compared against a no-change baseline.",
        "",
    ]

def _opinion_dataset_info(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Tuple[str, str]:
    """
    Extract dataset metadata from the opinion metrics bundle.

    :param metrics: Metrics dictionary captured from a previous pipeline stage.

    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]

    :returns: Markdown fragment describing the opinion dataset and preprocessing decisions.

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

    :type studies: Sequence[StudySpec]

    :returns: Markdown sections summarising opinion metrics per feature space.

    :rtype: List[str]

    """
    lines: List[str] = []
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in metrics
    ]
    for space in metrics:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
        per_feature = metrics.get(feature_space, {})
        if not per_feature:
            continue
        lines.extend(
            [
                _feature_space_heading(feature_space),
                "",
                "| Study | Participants | Best k | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | Baseline MAE ↓ |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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

    :returns: a Markdown table row for opinion metrics

    :rtype: str

    """
    summary = extract_opinion_summary(data)
    label = str(data.get("label", study.label))
    participants_text = format_count(summary.participants)
    return (
        f"| {label} | {participants_text} | {format_k(summary.best_k)} | "
        f"{format_optional_float(summary.mae)} | {format_delta(summary.mae_delta)} | "
        f"{format_optional_float(summary.rmse)} | {format_optional_float(summary.r2)} | "
        f"{format_optional_float(summary.mae_change)} | "
        f"{format_optional_float(summary.baseline_mae)} |"
    )

def _opinion_heatmap_section() -> List[str]:
    """
    Return the Markdown section referencing opinion heatmaps.

    :returns: the Markdown section referencing opinion heatmaps

    :rtype: List[str]

    """
    return [
        "### Opinion Change Heatmaps",
        "",
        "Plots are refreshed under `reports/knn/opinion/<feature-space>/` for MAE, R², and change heatmaps.",
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

    :returns: Markdown bullet list capturing the key takeaways from opinion evaluation.

    :rtype: List[str]

    """
    lines: List[str] = ["## Takeaways", ""]
    for study in studies:
        per_study: Dict[str, Tuple[OpinionSummary, Mapping[str, object]]] = {}
        for feature_space, per_feature in metrics.items():
            data = per_feature.get(study.key)
            if not data:
                continue
            per_study[feature_space] = (extract_opinion_summary(data), data)
        if not per_study:
            continue

        label = next(
            (data.get("label") for _summary, data in per_study.values() if data.get("label")),
            study.label,
        )
        best_r2_value: Optional[float] = None
        best_r2_space: Optional[str] = None
        best_r2_k: Optional[int] = None
        best_delta_value: Optional[float] = None
        best_delta_space: Optional[str] = None
        for feature_space, (summary, _data) in per_study.items():
            if summary.r2 is not None:
                if best_r2_value is None or summary.r2 > best_r2_value:
                    best_r2_value = summary.r2
                    best_r2_space = feature_space
                    best_r2_k = summary.best_k
            if summary.mae_delta is not None:
                if best_delta_value is None or summary.mae_delta > best_delta_value:
                    best_delta_value = summary.mae_delta
                    best_delta_space = feature_space

        bullet_bits: List[str] = []
        if best_r2_value is not None and best_r2_space is not None:
            bullet_bits.append(
                f"best R² {format_optional_float(best_r2_value)} with {best_r2_space.upper()} "
                f"(k={format_k(best_r2_k)})"
            )
        if best_delta_value is not None and best_delta_space is not None:
            bullet_bits.append(
                f"largest MAE reduction {format_delta(best_delta_value)} via {best_delta_space.upper()}"
            )
        if bullet_bits:
            lines.append(f"- {label}: " + "; ".join(bullet_bits) + ".")
    lines.append("")
    return lines


def _knn_opinion_cross_study_diagnostics(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Compute cross-study diagnostic statistics for KNN opinion runs."""

    if not metrics:
        return []

    lines: List[str] = ["## Cross-Study Diagnostics", ""]
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in metrics
    ]
    for space in metrics:
        if space not in ordered_spaces:
            ordered_spaces.append(space)

    any_entries = False
    for feature_space in ordered_spaces:
        per_feature = metrics.get(feature_space, {})
        if not per_feature:
            continue
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
            continue
        any_entries = True
        lines.append(_feature_space_heading(feature_space))
        lines.append("")
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

    if not any_entries:
        return []
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

    :returns: None.

    :rtype: None

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        if not allow_incomplete:
            raise RuntimeError("No opinion metrics available to build the opinion report.")
        placeholder = [
            "# KNN Opinion Shift Study",
            "",
            "Opinion regression metrics are not available yet. Execute the finalize stage to refresh these results.",
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

def _build_catalog_report(
    reports_root: Path,
    include_next_video: bool,
    include_opinion: bool,
) -> None:
    """
    Create the catalog README summarising generated artefacts.

    :param reports_root: Filesystem directory that will receive the Markdown reports.

    :type reports_root: Path

    :param include_next_video: Flag signalling whether next-video report sections should be rendered.

    :type include_next_video: bool

    :param include_opinion: Flag signalling whether opinion report sections should be rendered.

    :type include_opinion: bool

    :returns: None.

    :rtype: None

    """
    reports_root.mkdir(parents=True, exist_ok=True)
    path = reports_root / "README.md"
    lines: List[str] = []
    lines.append("# KNN Report Catalog")
    lines.append("")
    lines.append(
        "Baseline: k-nearest neighbours slate-ranking and opinion-regression models built on "
        "TF-IDF, Word2Vec, and Sentence-Transformer feature spaces."
    )
    lines.append("")
    lines.append(
        "These summaries are regenerated by `python -m knn.pipeline` (or the training wrappers) "
        "and provide a quick overview of the latest KNN baseline performance."
    )
    lines.append("")
    if include_next_video:
        lines.append(
            "- `hyperparameter_tuning/README.md` — sweep leaderboards and the per-study configuration"
            " that won each feature space."
        )
        lines.append(
            "- `next_video/README.md` — validation accuracy, confidence intervals, baseline deltas,"
            " and training vs. validation accuracy curves for the production slate task."
        )
    if include_opinion:
        lines.append(
            "- `opinion/README.md` — post-study opinion regression metrics alongside heatmaps"
            " generated under `reports/knn/opinion/`."
        )
    lines.append("")
    lines.append(
        "All artefacts referenced in the Markdown live under `models/knn/…`. Running the pipeline"
        " again will overwrite these reports with fresh numbers."
    )
    lines.append("")
    lines.append("## Refreshing Reports")
    lines.append("")
    lines.append("```bash")
    lines.append("PYTHONPATH=src python -m knn.pipeline --stage full \\")
    lines.append("  --out-dir models/knn \\")
    lines.append("  --reports-dir reports/knn")
    lines.append("```")
    lines.append("")
    lines.append(
        "Use `--stage plan` to inspect the sweep grid, `--stage sweeps` for array execution, and "
        "`--stage reports` to rebuild the documentation from existing metrics."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def generate_reports(repo_root: Path, report_bundle: ReportBundle) -> None:
    """
    Write refreshed Markdown reports under ``reports/knn``.

    :param repo_root: Repository root directory used for path resolution.

    :type repo_root: Path

    :param report_bundle: Aggregated data structure containing everything needed to emit reports.

    :type report_bundle: ReportBundle

    :returns: None.

    :rtype: None

    """
    reports_root = repo_root / "reports" / "knn"
    feature_spaces = report_bundle.feature_spaces
    allow_incomplete = report_bundle.allow_incomplete

    _build_catalog_report(
        reports_root,
        include_next_video=report_bundle.include_next_video,
        include_opinion=report_bundle.include_opinion,
    )

    if report_bundle.include_next_video or report_bundle.include_opinion:
        _build_hyperparameter_report(
            output_dir=reports_root / "hyperparameter_tuning",
            selections=report_bundle.selections if report_bundle.include_next_video else {},
            sweep_outcomes=report_bundle.sweep_outcomes
            if report_bundle.include_next_video
            else (),
            studies=report_bundle.studies,
            k_sweep=report_bundle.k_sweep,
            feature_spaces=feature_spaces,
            sentence_model=report_bundle.sentence_model,
            opinion_selections=(
                report_bundle.opinion_selections if report_bundle.include_opinion else {}
            ),
            opinion_sweep_outcomes=(
                report_bundle.opinion_sweep_outcomes
                if report_bundle.include_opinion
                else ()
            ),
            allow_incomplete=allow_incomplete,
            include_next_video=report_bundle.include_next_video,
            include_opinion=report_bundle.include_opinion,
        )
    if report_bundle.include_next_video:
        _build_next_video_report(
            output_dir=reports_root / "next_video",
            metrics_by_feature=report_bundle.metrics_by_feature,
            studies=report_bundle.studies,
            feature_spaces=feature_spaces,
            loso_metrics=report_bundle.loso_metrics,
            allow_incomplete=allow_incomplete,
        )

    if report_bundle.include_opinion:
        _build_opinion_report(
            output_path=reports_root / "opinion" / "README.md",
            metrics=report_bundle.opinion_metrics,
            studies=report_bundle.studies,
            allow_incomplete=allow_incomplete,
        )

__all__ = ["generate_reports"]
