"""Report generation helpers for the modular KNN pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from common.report_utils import extract_numeric_series

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt  # type: ignore[assignment]
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]

from .pipeline_context import OpinionSummary, ReportBundle, StudySelection, StudySpec, SweepOutcome
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


def _hyperparameter_report_intro(
    k_sweep: str,
    feature_spaces: Sequence[str],
    sentence_model: Optional[str],
) -> List[str]:
    """Return the Markdown header introducing the hyperparameter report."""

    feature_label = ", ".join(space.replace("_", "-").upper() for space in feature_spaces)
    lines = [
        "# KNN Hyperparameter Tuning Notes",
        "",
        "This document consolidates the selected grid searches for the KNN baselines.",
        "",
        "## Next-Video Prediction",
        "",
        f"The latest sweeps cover the {feature_label} feature spaces with:",
        f"- `k ∈ {{{k_sweep}}}`",
        "- Distance metrics: cosine and L2",
        "- Text-field augmentations: none, `viewer_profile,state_text`",
    ]
    if "word2vec" in feature_spaces:
        lines.append("- Word2Vec variants: vector size ∈ {128, 256}, window ∈ {5, 10}, min_count ∈ {1}")
    if "sentence_transformer" in feature_spaces and sentence_model:
        lines.append(f"- Sentence-transformer model: `{sentence_model}`")
    lines.extend(
        [
            "",
            "| Feature space | Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    return lines


def _hyperparameter_feature_rows(
    feature_space: str,
    per_study: Mapping[str, StudySelection],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Return table rows covering ``feature_space`` selections."""

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
    """Format a Markdown table row summarising a sweep selection."""

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
    """Render the hyperparameter summary table for each feature space."""

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
    """Return detailed leaderboards for the top-performing sweep configurations."""

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
                    "| {rank} | {label} | {acc} | {delta} | {k} | {eligible} |".format(
                        rank=idx,
                        label=label_display,
                        acc=format_float(outcome.accuracy),
                        delta=format_float(delta),
                        k=outcome.best_k,
                        eligible=outcome.eligible,
                    )
                )
            lines.append("")
        lines.append("")
    return lines


def _describe_text_fields(fields: Sequence[str]) -> str:
    """Return a readable description of text-field augmentations."""

    if not fields:
        return "base prompt only"
    return ", ".join(snake_to_title(field) for field in fields)


def _hyperparameter_observations_section(
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Summarise key takeaways from the hyperparameter sweeps."""

    lines: List[str] = ["### Observations", ""]
    for feature_space in ("tfidf", "word2vec", "sentence_transformer"):
        per_feature = selections.get(feature_space)
        if not per_feature:
            continue
        bullet_bits: List[str] = []
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
        if bullet_bits:
            lines.append(f"- {feature_space.upper()}: " + "; ".join(bullet_bits) + ".")
    lines.append("")
    return lines


def _hyperparameter_opinion_section() -> List[str]:
    """Return the blurb linking to opinion-regression sweeps."""

    return [
        "",
        "## Post-Study Opinion Regression",
        "",
        "Opinion runs reuse the per-study slate configurations gathered above.",
        "See `reports/knn/opinion/README.md` for detailed metrics and plots.",
        "",
    ]


def _feature_space_heading(feature_space: str) -> str:
    """Return the Markdown heading for ``feature_space``."""

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
    """Extract a representative dataset name and split from metrics payloads."""

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
    """Return the first uncertainty payload available for reporting."""

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
    """Return the introductory Markdown section for the next-video report."""

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


def _format_ci(ci_value: object) -> str:
    """Format a 95% confidence interval if present."""

    ci = parse_ci(ci_value)
    if ci is None:
        return "—"
    low, high = ci
    return f"[{low:.3f}, {high:.3f}]"


def _extract_curve_series(curve_block: Mapping[str, object]) -> Tuple[List[int], List[float]]:
    """Return sorted k/accuracy pairs extracted from ``curve_block``."""

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
    """Save a train/validation accuracy curve plot for ``study`` when possible."""

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
    """Render the next-video metrics table for ``feature_space``."""

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
    """Render Markdown sections embedding train/validation accuracy curves."""

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
    """Summarise per-feature observations for next-video metrics."""

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
    """Return a section summarising leave-one-study-out accuracy."""

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
) -> None:
    """Write the hyperparameter tuning summary under ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "README.md"
    lines: List[str] = []
    lines.extend(_hyperparameter_report_intro(k_sweep, feature_spaces, sentence_model))
    lines.extend(_hyperparameter_table_section(selections, studies))
    lines.extend(
        _hyperparameter_leaderboard_section(
            sweep_outcomes=sweep_outcomes,
            selections=selections,
            studies=studies,
            top_n=3,
        )
    )
    lines.extend(_hyperparameter_observations_section(selections, studies))
    lines.extend(_hyperparameter_opinion_section())
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
    """Compose the next-video evaluation report under ``output_dir``."""

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
    """Return the introductory Markdown section for the opinion report."""

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
    """Extract dataset metadata from the opinion metrics bundle."""

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
    """Render opinion metrics tables grouped by feature space."""

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
    """Return a Markdown table row for opinion metrics."""

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
    """Return the Markdown section referencing opinion heatmaps."""

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
    """Generate takeaway bullets comparing opinion performance."""

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


def _build_opinion_report(
    *,
    output_path: Path,
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
    allow_incomplete: bool = False,
) -> None:
    """Compose the opinion regression report at ``output_path``."""

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
    lines.extend(_opinion_takeaways(metrics, studies))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_catalog_report(
    reports_root: Path,
    include_next_video: bool,
    include_opinion: bool,
) -> None:
    """Create the catalog README summarising generated artefacts."""

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
    """Write refreshed Markdown reports under ``reports/knn``."""

    reports_root = repo_root / "reports" / "knn"
    feature_spaces = report_bundle.feature_spaces
    allow_incomplete = report_bundle.allow_incomplete

    _build_catalog_report(
        reports_root,
        include_next_video=report_bundle.include_next_video,
        include_opinion=report_bundle.include_opinion,
    )

    if report_bundle.include_next_video:
        _build_hyperparameter_report(
            output_dir=reports_root / "hyperparameter_tuning",
            selections=report_bundle.selections,
            sweep_outcomes=report_bundle.sweep_outcomes,
            studies=report_bundle.studies,
            k_sweep=report_bundle.k_sweep,
            feature_spaces=feature_spaces,
            sentence_model=report_bundle.sentence_model,
        )
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
