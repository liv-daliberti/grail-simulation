"""Report generation helpers for the XGBoost pipeline."""
# pylint: disable=line-too-long,duplicate-code,too-many-lines

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from common.pipeline_formatters import (
    format_count as _format_count,
    format_delta as _format_delta,
    format_float as _format_float,
    format_optional_float as _format_optional_float,
    format_ratio as _format_ratio,
    safe_float as _safe_float,
    safe_int as _safe_int,
)
from common.pipeline_io import write_markdown_lines
from common.report_utils import extract_numeric_series, start_markdown_report

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt  # type: ignore[assignment]
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]

from .pipeline_context import (
    NextVideoMetricSummary,
    OpinionSummary,
    StudySelection,
    SweepOutcome,
    SweepConfig,
)

LOGGER = logging.getLogger("xgb.pipeline.reports")
HYPERPARAM_TABLE_TOP_N = 10
HYPERPARAM_LEADERBOARD_TOP_N = 5
def _extract_next_video_summary(data: Mapping[str, object]) -> NextVideoMetricSummary:
    """
    Normalise next-video metrics into a reusable summary structure.

    :param data: Raw metrics dictionary loaded from ``metrics.json``.
    :type data: Mapping[str, object]
    :returns: Dataclass containing typed fields for report generation.
    :rtype: NextVideoMetricSummary
    """

    accuracy = _safe_float(data.get("accuracy"))
    coverage = _safe_float(data.get("coverage"))
    evaluated = _safe_int(data.get("evaluated"))
    correct = _safe_int(data.get("correct"))
    known_hits = _safe_int(data.get("known_candidate_hits"))
    known_total = _safe_int(data.get("known_candidate_total"))
    known_availability = None
    if known_total is not None and evaluated:
        known_availability = known_total / evaluated if evaluated else None
    avg_probability = _safe_float(data.get("avg_probability"))
    dataset = data.get("dataset_source") or data.get("dataset")
    issue = data.get("issue")
    issue_label = data.get("issue_label")
    study_label = data.get("study_label") or data.get("study")
    return NextVideoMetricSummary(
        accuracy=accuracy,
        coverage=coverage,
        evaluated=evaluated,
        correct=correct,
        known_hits=known_hits,
        known_total=known_total,
        known_availability=known_availability,
        avg_probability=avg_probability,
        dataset=str(dataset) if dataset else None,
        issue=str(issue) if issue else None,
        issue_label=str(issue_label) if issue_label else None,
        study_label=str(study_label) if study_label else None,
    )


def _extract_opinion_summary(data: Mapping[str, object]) -> OpinionSummary:
    """
    Normalise opinion regression metrics into a reusable summary structure.

    :param data: Raw metrics dictionary emitted by the opinion pipeline.
    :type data: Mapping[str, object]
    :returns: Dataclass containing typed fields for opinion reporting.
    :rtype: OpinionSummary
    """

    metrics_block = data.get("metrics", {})
    baseline = data.get("baseline", {})
    mae_after = _safe_float(metrics_block.get("mae_after"))
    baseline_mae = _safe_float(baseline.get("mae_before") or baseline.get("mae_using_before"))
    mae_delta = None
    if mae_after is not None and baseline_mae is not None:
        mae_delta = baseline_mae - mae_after
    return OpinionSummary(
        mae_after=mae_after,
        rmse_after=_safe_float(metrics_block.get("rmse_after")),
        r2_after=_safe_float(metrics_block.get("r2_after")),
        baseline_mae=baseline_mae,
        mae_delta=mae_delta,
        participants=_safe_int(data.get("n_participants")),
        dataset=str(data.get("dataset")) if data.get("dataset") else None,
        split=str(data.get("split")) if data.get("split") else None,
        label=str(data.get("label")) if data.get("label") else None,
    )


def _next_video_dataset_info(metrics: Mapping[str, Mapping[str, object]]) -> str:
    """
    Determine the dataset identifier referenced by evaluation metrics.

    :param metrics: Mapping from study key to metrics dictionaries.
    :type metrics: Mapping[str, Mapping[str, object]]
    :returns: Dataset identifier or ``"unknown"`` when absent.
    :rtype: str
    """

    for payload in metrics.values():
        summary = _extract_next_video_summary(payload)
        if summary.dataset:
            return summary.dataset
    return "unknown"


def _next_video_observations(metrics: Mapping[str, Mapping[str, object]]) -> List[str]:
    """
    Generate bullet-point observations comparing study metrics.

    :param metrics: Mapping from study key to metrics dictionaries.
    :type metrics: Mapping[str, Mapping[str, object]]
    :returns: Markdown-formatted bullet points summarising key observations.
    :rtype: List[str]
    """

    if not metrics:
        return []
    lines: List[str] = ["## Observations", ""]
    accuracies: List[float] = []
    coverages: List[float] = []
    availabilities: List[float] = []
    for study_key in sorted(metrics.keys(), key=lambda key: (metrics[key].get("study_label") or key).lower()):
        summary = _extract_next_video_summary(metrics[study_key])
        accuracy_text = _format_optional_float(summary.accuracy)
        coverage_text = _format_optional_float(summary.coverage)
        availability_text = _format_optional_float(summary.known_availability)
        avg_prob_text = _format_optional_float(summary.avg_probability)
        lines.append(
            f"- {summary.study_label or study_key}: accuracy {accuracy_text}, "
            f"coverage {coverage_text}, known availability {availability_text}, "
            f"avg probability {avg_prob_text}."
        )
        if summary.accuracy is not None:
            accuracies.append(summary.accuracy)
        if summary.coverage is not None:
            coverages.append(summary.coverage)
        if summary.known_availability is not None:
            availabilities.append(summary.known_availability)
    if accuracies:
        lines.append(
            f"- Portfolio mean accuracy { _format_optional_float(sum(accuracies) / len(accuracies)) } "
            f"across {len(accuracies)} studies."
        )
    if coverages:
        lines.append(
            f"- Mean coverage { _format_optional_float(sum(coverages) / len(coverages)) }."
        )
    if availabilities:
        lines.append(
            f"- Known candidate availability averages "
            f"{ _format_optional_float(sum(availabilities) / len(availabilities)) }."
        )
    lines.append("")
    return lines


def _extract_curve_steps(curve_block: Mapping[str, object]) -> Tuple[List[int], List[float]]:
    """
    Extract sorted evaluation steps and accuracies from a curve payload.

    :param curve_block: Curve payload containing an ``accuracy_by_step`` mapping.
    :type curve_block: Mapping[str, object]
    :returns: Pair of ``(steps, accuracies)`` sorted by evaluation index.
    :rtype: Tuple[List[int], List[float]]
    """

    accuracy_map = curve_block.get("accuracy_by_step")
    if not isinstance(accuracy_map, Mapping):
        return ([], [])
    return extract_numeric_series(accuracy_map)


def _load_curve_bundle(payload: Mapping[str, object]) -> Optional[Mapping[str, object]]:
    """
    Load the stored curve metrics bundle, reading from disk when required.

    :param payload: Metrics dictionary potentially containing in-memory or on-disk curves.
    :type payload: Mapping[str, object]
    :returns: Curve metrics mapping or ``None`` when unavailable.
    :rtype: Optional[Mapping[str, object]]
    """

    curve_bundle = payload.get("curve_metrics")
    if isinstance(curve_bundle, Mapping):
        return curve_bundle
    curve_path = payload.get("curve_metrics_path")
    if not curve_path:
        return None
    try:
        with open(curve_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, Mapping):
                return loaded
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - logging aid
        LOGGER.warning("Unable to read curve metrics from %s: %s", curve_path, exc)
    return None


# pylint: disable=too-many-locals
def _plot_xgb_curve(
    *,
    directory: Path,
    study_label: str,
    study_key: str,
    payload: Mapping[str, object],
) -> Optional[str]:
    """
    Persist a training/validation accuracy curve plot for a study.

    :param directory: Report directory where plots are stored.
    :type directory: Path
    :param study_label: Human-readable study label.
    :type study_label: str
    :param study_key: Study identifier used for slug generation.
    :type study_key: str
    :param payload: Metrics payload containing curve information.
    :type payload: Mapping[str, object]
    :returns: Relative path to the generated image or ``None`` when plotting fails.
    :rtype: Optional[str]
    """

    if plt is None:  # pragma: no cover - optional dependency
        return None
    curve_bundle = _load_curve_bundle(payload)
    if not isinstance(curve_bundle, Mapping):
        return None
    eval_curve = curve_bundle.get("eval")
    if not isinstance(eval_curve, Mapping):
        return None
    eval_x, eval_y = _extract_curve_steps(eval_curve)
    if not eval_x:
        return None
    train_x: List[int] = []
    train_y: List[float] = []
    train_curve = curve_bundle.get("train")
    if isinstance(train_curve, Mapping):
        train_x, train_y = _extract_curve_steps(train_curve)

    curves_dir = directory / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    slug_source = study_label or study_key or "study"
    slug = slug_source.lower().replace(" ", "_").replace("/", "_")
    plot_path = curves_dir / f"{slug}.png"

    fig, axis = plt.subplots(figsize=(6, 3.5))  # type: ignore[attr-defined]
    axis.plot(eval_x, eval_y, marker="o", label="validation")
    if train_x and train_y:
        axis.plot(train_x, train_y, marker="o", linestyle="--", label="training")
    axis.set_title(study_label or study_key)
    axis.set_xlabel("Evaluated examples")
    axis.set_ylabel("Cumulative accuracy")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)  # type: ignore[attr-defined]
    plt.close(fig)  # type: ignore[attr-defined]
    try:
        return plot_path.relative_to(directory).as_posix()
    except ValueError:
        return plot_path.as_posix()


def _opinion_observations(metrics: Mapping[str, Mapping[str, object]]) -> List[str]:
    """
    Generate bullet-point takeaways for the opinion regression stage.

    :param metrics: Mapping from study key to opinion metrics dictionaries.
    :type metrics: Mapping[str, Mapping[str, object]]
    :returns: Markdown-formatted bullet points summarising opinion results.
    :rtype: List[str]
    """

    if not metrics:
        return []
    lines: List[str] = ["## Observations", ""]
    deltas: List[float] = []
    r2_scores: List[float] = []
    for study_key in sorted(metrics.keys(), key=lambda key: (metrics[key].get("label") or key).lower()):
        summary = _extract_opinion_summary(metrics[study_key])
        delta_text = _format_delta(summary.mae_delta)
        mae_text = _format_optional_float(summary.mae_after)
        r2_text = _format_optional_float(summary.r2_after)
        lines.append(
            f"- {summary.label or study_key}: MAE {mae_text} "
            f"(Δ vs. baseline {delta_text}), R² {r2_text}."
        )
        if summary.mae_delta is not None:
            deltas.append(summary.mae_delta)
        if summary.r2_after is not None:
            r2_scores.append(summary.r2_after)
    if deltas:
        lines.append(
            f"- Average MAE reduction { _format_delta(sum(deltas) / len(deltas)) } across "
            f"{len(deltas)} studies."
        )
    if r2_scores:
        lines.append(
            f"- Mean R² { _format_optional_float(sum(r2_scores) / len(r2_scores)) }."
        )
    lines.append("")
    return lines


def _next_video_portfolio_summary(
    metrics: Mapping[str, Mapping[str, object]],
) -> List[str]:
    """
    Generate weighted portfolio statistics for next-video evaluations.

    :param metrics: Mapping from study key to final evaluation metrics.
    :type metrics: Mapping[str, Mapping[str, object]]
    :returns: Markdown lines highlighting portfolio-level performance.
    :rtype: List[str]
    """

    total_correct = 0
    total_evaluated = 0
    total_known_hits = 0
    total_known_total = 0
    accuracy_entries: List[Tuple[float, str]] = []
    probability_values: List[float] = []

    for study_key, payload in metrics.items():
        summary = _extract_next_video_summary(payload)
        if summary.correct is not None and summary.evaluated is not None:
            total_correct += summary.correct
            total_evaluated += summary.evaluated
        if summary.known_hits is not None and summary.known_total is not None:
            total_known_hits += summary.known_hits
            total_known_total += summary.known_total
        label = summary.study_label or study_key
        if summary.accuracy is not None:
            accuracy_entries.append((summary.accuracy, label))
        if summary.avg_probability is not None:
            probability_values.append(summary.avg_probability)

    if not total_evaluated and not accuracy_entries:
        return []

    weighted_accuracy: Optional[float] = None
    if total_evaluated:
        weighted_accuracy = total_correct / total_evaluated
    weighted_coverage: Optional[float] = None
    if total_known_total:
        weighted_coverage = total_known_hits / total_known_total
    weighted_availability: Optional[float] = None
    if total_evaluated:
        weighted_availability = total_known_total / total_evaluated
    mean_probability: Optional[float] = None
    if probability_values:
        mean_probability = sum(probability_values) / len(probability_values)

    lines: List[str] = ["## Portfolio Summary", ""]
    if total_evaluated:
        lines.append(
            f"- Weighted accuracy {_format_optional_float(weighted_accuracy)} "
            f"across {_format_count(total_evaluated)} evaluated slates."
        )
    if total_known_total:
        lines.append(
            f"- Weighted known-candidate coverage {_format_optional_float(weighted_coverage)} "
            f"over {_format_count(total_known_total)} eligible slates."
        )
    if weighted_availability is not None:
        lines.append(
            f"- Known-candidate availability {_format_optional_float(weighted_availability)} "
            f"relative to all evaluated slates."
        )
    if mean_probability is not None:
        probability_count = len(probability_values)
        study_label = "study" if probability_count == 1 else "studies"
        lines.append(
            f"- Mean predicted probability on known candidates "
            f"{_format_optional_float(mean_probability)} "
            f"(across {probability_count} {study_label} with recorded probabilities)."
        )

    if accuracy_entries:
        best_accuracy, best_label = max(accuracy_entries, key=lambda item: item[0])
        lines.append(
            f"- Highest study accuracy: {best_label} ({_format_optional_float(best_accuracy)})."
        )
        if len(accuracy_entries) > 1:
            worst_accuracy, worst_label = min(accuracy_entries, key=lambda item: item[0])
            lines.append(
                f"- Lowest study accuracy: {worst_label} ({_format_optional_float(worst_accuracy)})."
            )
    lines.append("")
    return lines


# pylint: disable=too-many-arguments
def _write_reports(
    *,
    reports_dir: Path,
    outcomes: Sequence[SweepOutcome],
    selections: Mapping[str, StudySelection],
    final_metrics: Mapping[str, Mapping[str, object]],
    opinion_metrics: Mapping[str, Mapping[str, object]],
    allow_incomplete: bool,
    include_next_video: bool = True,
    include_opinion: bool = True,
) -> None:
    """
    Write the full report bundle capturing sweep and evaluation artefacts.

    :param reports_dir: Base directory receiving generated Markdown files.
    :type reports_dir: Path
    :param outcomes: Sweep outcomes considered during model selection.
    :type outcomes: Sequence[SweepOutcome]
    :param selections: Mapping from study key to selected sweep outcome.
    :type selections: Mapping[str, StudySelection]
    :param final_metrics: Mapping from study key to final evaluation metrics.
    :type final_metrics: Mapping[str, Mapping[str, object]]
    :param opinion_metrics: Mapping from study key to opinion regression metrics.
    :type opinion_metrics: Mapping[str, Mapping[str, object]]
    :param allow_incomplete: Flag controlling whether missing artefacts are tolerated.
    :type allow_incomplete: bool
    :param include_next_video: Flag enabling next-video sections.
    :type include_next_video: bool
    :param include_opinion: Flag enabling opinion sections.
    :type include_opinion: bool
    """

    reports_dir.mkdir(parents=True, exist_ok=True)

    legacy_hyper_file = reports_dir / "hyperparameter_tuning.md"
    legacy_next_file = reports_dir / "next_video.md"
    if legacy_hyper_file.exists():
        legacy_hyper_file.unlink()
    if legacy_next_file.exists():
        legacy_next_file.unlink()

    _write_catalog_report(
        reports_dir,
        include_next_video=include_next_video,
        include_opinion=include_opinion,
    )
    if include_next_video:
        _write_hyperparameter_report(
            reports_dir / "hyperparameter_tuning",
            outcomes,
            selections,
            allow_incomplete=allow_incomplete,
        )
        _write_next_video_report(
            reports_dir / "next_video",
            final_metrics,
            selections,
            allow_incomplete=allow_incomplete,
        )
    else:
        _write_disabled_report(
            reports_dir / "hyperparameter_tuning",
            "Hyper-parameter Tuning",
            "Next-video sweeps were disabled for this run.",
        )
        _write_disabled_report(
            reports_dir / "next_video",
            "Next-Video Evaluation",
            "Next-video evaluation was skipped because the task was not selected.",
        )
    if include_opinion:
        _write_opinion_report(
            reports_dir / "opinion",
            opinion_metrics,
            allow_incomplete=allow_incomplete,
        )
    else:
        _write_disabled_report(
            reports_dir / "opinion",
            "Opinion Regression",
            "Opinion sweeps were disabled for this run.",
        )


def _write_catalog_report(
    reports_dir: Path,
    *,
    include_next_video: bool,
    include_opinion: bool,
) -> None:
    """
    Create the catalog README summarising generated artefacts.

    :param reports_dir: Directory where the catalog README is written.
    :type reports_dir: Path
    :param include_next_video: Flag indicating whether next-video sections are enabled.
    :type include_next_video: bool
    :param include_opinion: Flag indicating whether opinion sections are enabled.
    :type include_opinion: bool
    """

    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "README.md"
    lines: List[str] = []
    lines.append("# XGBoost Report Catalog")
    lines.append("")
    lines.append(
        "The Markdown artefacts in this directory are produced by `python -m xgb.pipeline` "
        "(or `training/training-xgb.sh`) and track the XGBoost baselines that accompany the simulation:"
    )
    lines.append("")
    if include_next_video:
        lines.append("- `hyperparameter_tuning/README.md` – sweep grids, configuration deltas, and parameter frequency summaries.")
        lines.append("- `next_video/README.md` – validation accuracy, coverage, and probability diagnostics for the slate-ranking task.")
    if include_opinion:
        lines.append("- `opinion/README.md` – post-study opinion regression metrics with MAE deltas versus the no-change baseline.")
    lines.append("")
    lines.append("Raw metrics, model checkpoints, and intermediate artefacts referenced by these reports live beneath `models/xgb/…`.")
    lines.append("")
    lines.append("## Refreshing Reports")
    lines.append("")
    lines.append("```bash")
    lines.append("PYTHONPATH=src python -m xgb.pipeline --stage full \\")
    lines.append("  --out-dir models/xgb \\")
    lines.append("  --reports-dir reports/xgb")
    lines.append("```")
    lines.append("")
    lines.append("Stages can be invoked individually (`plan`, `sweeps`, `finalize`, `reports`) to match existing SLURM workflows.")
    lines.append("")
    write_markdown_lines(path, lines)


# pylint: disable=too-many-locals,too-many-statements
def _write_hyperparameter_report(
    directory: Path,
    outcomes: Sequence[SweepOutcome],
    selections: Mapping[str, StudySelection],
    *,
    allow_incomplete: bool,
) -> None:
    """
    Create the hyper-parameter sweep summary document.

    :param directory: Directory where the report and assets are written.
    :type directory: Path
    :param outcomes: Sweep outcomes considered during selection.
    :type outcomes: Sequence[SweepOutcome]
    :param selections: Mapping from study key to selected sweep outcome.
    :type selections: Mapping[str, StudySelection]
    :param allow_incomplete: Flag controlling placeholder messaging when artefacts are missing.
    :type allow_incomplete: bool
    """

    path, lines = start_markdown_report(directory, title="Hyper-parameter Tuning")
    lines.append(
        (
            f"This summary lists the top-performing configurations uncovered for each participant study "
            f"(showing up to {HYPERPARAM_TABLE_TOP_N} rows per study). "
            "Selections promoted to the final pipeline are highlighted in bold. "
            "See the leaderboard section below for ranked deltas."
        )
    )
    lines.append("")
    per_study: Dict[str, List[SweepOutcome]] = {}
    for outcome in outcomes:
        per_study.setdefault(outcome.study.key, []).append(outcome)

    if not per_study:
        lines.append("No sweep runs were available when this report was generated.")
        if allow_incomplete:
            lines.append(
                "Run the XGBoost pipeline with `--stage sweeps` or `--stage full` once artefacts are ready."
            )
        lines.append("")
        write_markdown_lines(path, lines)
        return

    def _study_label(study_key: str) -> str:
        """
        Resolve a human-readable study label for sorting purposes.

        :param study_key: Key identifying the participant study.
        :type study_key: str
        :returns: Study label derived from selections or available outcomes.
        :rtype: str
        """
        selection = selections.get(study_key)
        if selection is not None:
            return selection.study.label
        study_outcomes = per_study.get(study_key)
        if study_outcomes:
            return study_outcomes[0].study.label
        return study_key

    sorted_study_outcomes: Dict[str, List[SweepOutcome]] = {}
    for study_key in sorted(per_study.keys(), key=lambda key: _study_label(key).lower()):
        study_outcomes = per_study[study_key]
        selection = selections.get(study_key)
        study_label = _study_label(study_key)
        issue_label = study_outcomes[0].study.issue.replace("_", " ").title()
        lines.append(f"## {study_label}")
        lines.append("")
        lines.append(f"*Issue:* {issue_label}")
        lines.append("")
        lines.append("| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |")
        lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: |")
        ordered = sorted(
            study_outcomes,
            key=lambda item: (item.accuracy, item.coverage, item.evaluated),
            reverse=True,
        )
        sorted_study_outcomes[study_key] = ordered
        display_limit = max(1, HYPERPARAM_TABLE_TOP_N)
        displayed = ordered[:display_limit]
        selected_outcome = None
        if selection is not None:
            for candidate in ordered:
                if candidate.config == selection.config:
                    selected_outcome = candidate
                    break
        if selected_outcome is not None and selected_outcome not in displayed:
            displayed.append(selected_outcome)
            displayed.sort(key=lambda item: (item.accuracy, item.coverage, item.evaluated), reverse=True)
            displayed = displayed[:display_limit]
        for outcome in displayed:
            label = outcome.config.label()
            formatted = (
                f"**{label}**" if selection and outcome.config == selection.config else label
            )
            summary = _extract_next_video_summary(outcome.metrics)
            lines.append(
                f"| {formatted} | {_format_optional_float(summary.accuracy)} | "
                f"{_format_optional_float(summary.coverage)} | "
                f"{_format_ratio(summary.known_hits, summary.known_total)} | "
                f"{_format_optional_float(summary.known_availability)} | "
                f"{_format_optional_float(summary.avg_probability)} | "
                f"{_format_count(summary.evaluated)} |"
            )
        if len(ordered) > display_limit:
            lines.append(
                f"*Showing top {display_limit} of {len(ordered)} configurations.*"
            )
        lines.append("")

    lines.extend(
        _xgb_leaderboard_section(
            per_study_sorted=sorted_study_outcomes,
            selections=selections,
            top_n=HYPERPARAM_LEADERBOARD_TOP_N,
        )
    )

    if selections:
        lines.extend(_xgb_selection_summary_section(sorted_study_outcomes, selections))
        lines.extend(_xgb_parameter_frequency_section(selections))
    write_markdown_lines(path, lines)


# pylint: disable=too-many-locals
def _xgb_leaderboard_section(
    *,
    per_study_sorted: Mapping[str, Sequence[SweepOutcome]],
    selections: Mapping[str, StudySelection],
    top_n: int,
) -> List[str]:
    """
    Render ranked leaderboards mirroring the KNN report format.

    :param per_study_sorted: Mapping from study key to ordered sweep outcomes.
    :type per_study_sorted: Mapping[str, Sequence[SweepOutcome]]
    :param selections: Mapping from study key to selected sweep outcome.
    :type selections: Mapping[str, StudySelection]
    :param top_n: Maximum number of leaderboard entries per study.
    :type top_n: int
    :returns: Markdown lines representing the leaderboard section.
    :rtype: List[str]
    """

    if not per_study_sorted:
        return []

    def _descriptor(study_key: str) -> str:
        """
        Resolve the display label for leaderboard headings.

        :param study_key: Key identifying the participant study.
        :type study_key: str
        :returns: Human-readable study label.
        :rtype: str
        """
        selection = selections.get(study_key)
        if selection is not None:
            return selection.study.label
        outcomes = per_study_sorted.get(study_key, ())
        if outcomes:
            return outcomes[0].study.label
        return study_key

    lines: List[str] = ["### Configuration Leaderboards", ""]
    for study_key in sorted(per_study_sorted.keys(), key=lambda key: _descriptor(key).lower()):
        outcomes = per_study_sorted[study_key]
        if not outcomes:
            continue
        selection = selections.get(study_key)
        limit = max(1, top_n)
        best = outcomes[0]
        best_accuracy = best.accuracy
        best_coverage = best.coverage
        lines.append(f"#### {_descriptor(study_key)}")
        lines.append("")
        lines.append("| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for idx, outcome in enumerate(outcomes[:limit], start=1):
            label = outcome.config.label()
            formatted = f"**{label}**" if selection and outcome.config == selection.config else label
            delta_acc = None
            if best_accuracy is not None and outcome.accuracy is not None:
                delta_acc = max(0.0, best_accuracy - outcome.accuracy)
            delta_cov = None
            if best_coverage is not None and outcome.coverage is not None:
                delta_cov = max(0.0, best_coverage - outcome.coverage)
            acc_text = _format_optional_float(outcome.accuracy)
            cov_text = _format_optional_float(outcome.coverage)
            delta_acc_text = _format_optional_float(delta_acc)
            delta_cov_text = _format_optional_float(delta_cov)
            evaluated_text = _format_count(outcome.evaluated)
            lines.append(
                f"| {idx} | {formatted} | {acc_text} | {delta_acc_text} | "
                f"{cov_text} | {delta_cov_text} | {evaluated_text} |"
            )
        if len(outcomes) > limit:
            lines.append(f"*Showing top {limit} of {len(outcomes)} configurations.*")
        lines.append("")
    return lines


def _xgb_selection_summary_section(
    per_study_sorted: Mapping[str, Sequence[SweepOutcome]],
    selections: Mapping[str, StudySelection],
) -> List[str]:
    """
    Render a bullet summary comparing winning configurations to runner-ups.

    :param per_study_sorted: Mapping from study key to ordered sweep outcomes.
    :type per_study_sorted: Mapping[str, Sequence[SweepOutcome]]
    :param selections: Mapping from study key to selected sweep outcome.
    :type selections: Mapping[str, StudySelection]
    :returns: Markdown bullet list highlighting improvements over runner-ups.
    :rtype: List[str]
    """

    lines: List[str] = ["### Selection Summary", ""]
    for study_key in sorted(per_study_sorted.keys(), key=lambda key: selections.get(key).study.label.lower() if selections.get(key) else per_study_sorted[key][0].study.label.lower()):
        selection = selections.get(study_key)
        ordered = per_study_sorted.get(study_key, [])
        if not ordered:
            continue
        descriptor = selection.study.label if selection is not None else ordered[0].study.label
        issue_label = ordered[0].study.issue.replace("_", " ").title()
        descriptor_full = f"{descriptor} (issue {issue_label})"
        if selection is None:
            top = ordered[0]
            lines.append(
                f"- **{descriptor_full}**: accuracy {_format_float(top.accuracy)} "
                f"with {_summarise_xgb_config(top.config)}."
            )
            continue
        best = selection.outcome
        summary = _summarise_xgb_config(selection.config)
        runner_up = ordered[1] if len(ordered) > 1 else None
        if runner_up is not None:
            delta_acc = best.accuracy - runner_up.accuracy
            delta_cov = best.coverage - runner_up.coverage
            lines.append(
                f"- **{descriptor_full}**: accuracy {_format_float(best.accuracy)} "
                f"(coverage {_format_float(best.coverage)}) using {summary}. "
                f"Δ accuracy vs. runner-up {_format_delta(delta_acc)}; Δ coverage {_format_delta(delta_cov)}."
            )
        else:
            lines.append(
                f"- **{descriptor_full}**: accuracy {_format_float(best.accuracy)} "
                f"(coverage {_format_float(best.coverage)}) using {summary}."
            )
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
    :type config: SweepConfig
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


# pylint: disable=too-many-locals
def _write_next_video_report(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
    selections: Mapping[str, StudySelection],
    *,
    allow_incomplete: bool,
) -> None:
    """
    Create the next-video evaluation summary document.

    :param directory: Directory where the report and assets are written.
    :type directory: Path
    :param metrics: Mapping from study key to final evaluation metrics.
    :type metrics: Mapping[str, Mapping[str, object]]
    :param selections: Mapping from study key to selected sweep outcome.
    :type selections: Mapping[str, StudySelection]
    :param allow_incomplete: Flag controlling placeholder messaging when artefacts are missing.
    :type allow_incomplete: bool
    """

    path, lines = start_markdown_report(directory, title="XGBoost Next-Video Baseline")
    if metrics:
        dataset_name = _next_video_dataset_info(metrics)
        lines.append("Slate-ranking accuracy for the selected XGBoost configuration.")
        lines.append("")
        lines.append(f"- Dataset: `{dataset_name}`")
        lines.append("- Split: validation")
        lines.append("- Metrics: accuracy, coverage of known candidates, and availability of known neighbours.")
        lines.append("")
        lines.extend(_next_video_portfolio_summary(metrics))
    else:
        lines.append("Accuracy on the validation split for the selected slate configuration.")
        lines.append("")

    if not metrics:
        lines.append("No finalized evaluation metrics were available when this report was generated.")
        if allow_incomplete:
            lines.append(
                "Run the pipeline with `--stage finalize` once sufficient artefacts exist to refresh this table."
            )
        lines.append("")
        write_markdown_lines(path, lines)
        return

    lines.append("| Study | Issue | Accuracy ↑ | Correct / evaluated | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ |")
    lines.append("| --- | --- | ---: | --- | ---: | --- | ---: | ---: |")

    def _study_label(study_key: str) -> str:
        """
        Resolve the display label for a study row in the report table.

        :param study_key: Key identifying the participant study.
        :type study_key: str
        :returns: Human-readable study label.
        :rtype: str
        """
        selection = selections.get(study_key)
        if selection is not None:
            return selection.study.label
        return study_key

    def _issue_label(study_key: str, fallback: str) -> str:
        """
        Resolve the issue label for a study row in the report table.

        :param study_key: Key identifying the participant study.
        :type study_key: str
        :param fallback: Fallback issue string from the metrics payload.
        :type fallback: str
        :returns: Human-readable issue label.
        :rtype: str
        """
        selection = selections.get(study_key)
        if selection is not None:
            return selection.study.issue.replace("_", " ").title()
        if fallback:
            return str(fallback).replace("_", " ").title()
        return ""

    for study_key in sorted(metrics.keys(), key=lambda key: _study_label(key).lower()):
        payload = metrics[study_key]
        summary = _extract_next_video_summary(payload)
        study_label = summary.study_label or _study_label(study_key)
        issue_label = summary.issue_label or _issue_label(
            study_key, summary.issue or study_key
        )
        lines.append(
            f"| {study_label} | {issue_label or _issue_label(study_key, '')} | "
            f"{_format_optional_float(summary.accuracy)} | "
            f"{_format_ratio(summary.correct, summary.evaluated)} | "
            f"{_format_optional_float(summary.coverage)} | "
            f"{_format_ratio(summary.known_hits, summary.known_total)} | "
            f"{_format_optional_float(summary.known_availability)} | "
            f"{_format_optional_float(summary.avg_probability)} |"
        )
    lines.append("")
    curve_lines: List[str] = []
    if plt is not None:
        for study_key in sorted(metrics.keys(), key=lambda key: _study_label(key).lower()):
            payload = metrics[study_key]
            label = _study_label(study_key)
            rel_path = _plot_xgb_curve(
                directory=directory,
                study_label=label,
                study_key=study_key,
                payload=payload,
            )
            if rel_path:
                if not curve_lines:
                    curve_lines.extend(["## Accuracy Curves", ""])
                curve_lines.append(f"![{label}]({rel_path})")
                curve_lines.append("")
    if curve_lines:
        lines.extend(curve_lines)
    lines.extend(_next_video_observations(metrics))
    write_markdown_lines(path, lines)


def _write_opinion_report(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
    *,
    allow_incomplete: bool,
) -> None:
    """
    Create the opinion regression summary document.

    :param directory: Directory where the report and assets are written.
    :type directory: Path
    :param metrics: Mapping from study key to opinion metrics.
    :type metrics: Mapping[str, Mapping[str, object]]
    :param allow_incomplete: Flag controlling placeholder messaging when artefacts are missing.
    :type allow_incomplete: bool
    """

    path, lines = start_markdown_report(directory, title="XGBoost Opinion Regression")
    if not metrics:
        lines.append("No opinion runs were produced during this pipeline invocation.")
        if allow_incomplete:
            lines.append(
                "Rerun the pipeline with `--stage finalize` to populate this section once opinion metrics are available."
            )
        lines.append("")
        write_markdown_lines(path, lines)
        return
    lines.append("MAE / RMSE / R² scores for predicting the post-study opinion index.")
    lines.append("")
    dataset_name = "unknown"
    split_name = "validation"
    for payload in metrics.values():
        summary = _extract_opinion_summary(payload)
        if dataset_name == "unknown" and summary.dataset:
            dataset_name = summary.dataset
        if summary.split:
            split_name = summary.split
        if dataset_name != "unknown" and summary.split:
            break
    lines.append(f"- Dataset: `{dataset_name}`")
    lines.append(f"- Split: {split_name}")
    lines.append("- Metrics: MAE, RMSE, and R² compared against a no-change baseline (pre-study opinion).")
    lines.append("")
    lines.append("| Study | Participants | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Baseline MAE ↓ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for study_key, payload in sorted(metrics.items()):
        summary = _extract_opinion_summary(payload)
        lines.append(
            f"| {summary.label or study_key} | {_format_count(summary.participants)} | "
            f"{_format_optional_float(summary.mae_after)} | {_format_delta(summary.mae_delta)} | "
            f"{_format_optional_float(summary.rmse_after)} | {_format_optional_float(summary.r2_after)} | "
            f"{_format_optional_float(summary.baseline_mae)} |"
        )
    lines.append("")
    lines.extend(_opinion_observations(metrics))
    write_markdown_lines(path, lines)


def _write_disabled_report(directory: Path, title: str, message: str) -> None:
    """
    Emit a placeholder report clarifying why a section is absent.

    :param directory: Directory where the placeholder README is written.
    :type directory: Path
    :param title: Report title communicated to readers.
    :type title: str
    :param message: Explanatory message describing the omission.
    :type message: str
    """

    path, lines = start_markdown_report(directory, title=title)
    lines.append(message)
    lines.append("")
    write_markdown_lines(path, lines)

__all__ = [
    '_format_float',
    '_format_optional_float',
    '_format_delta',
    '_format_count',
    '_format_ratio',
    '_extract_next_video_summary',
    '_extract_opinion_summary',
    '_write_reports',
    '_write_disabled_report',
]
