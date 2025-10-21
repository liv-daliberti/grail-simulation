"""Top-level orchestration for the political sciences replication report."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import DatasetDict

from clean_data.clean_data import dedupe_by_participant_issue

from .analysis import (
    assemble_study_specs,
    compute_treatment_regression,
    dataframe_from_splits,
    histogram2d_counts,
    load_assignment_frame,
    prepare_study_frame,
    summarise_assignments,
    summarise_shift,
)
from .markdown import build_markdown
from .plotting import plot_heatmap, plot_assignment_panels
from .stratified import analyze_preregistered_effects


def _policy_summary_rows(stratified: pd.DataFrame) -> List[Dict[str, object]]:
    if stratified.empty:
        return []

    outcome_labels = {
        "study1": ("gun_index_w2", "Gun policy index"),
        "study2": ("mw_index_w2", "Minimum wage index"),
        "study3": ("mw_index_w2", "Minimum wage index"),
    }

    rows: List[Dict[str, object]] = []
    for study_key, (outcome, outcome_label) in outcome_labels.items():
        subset = stratified[
            (stratified["study_key"] == study_key)
            & (stratified["family_key"] == "policy")
            & (stratified["outcome"] == outcome)
            & (stratified.get("contrast_display", True))
        ].sort_values(["contrast_label"])
        for _, record in subset.iterrows():
            rows.append(
                {
                    "study_label": record["study_label"],
                    "contrast_label": record["contrast_label"],
                    "outcome_label": outcome_label,
                    "estimate": float(record.get("estimate", float("nan"))),
                    "ci_low": float(record.get("ci_low", float("nan"))),
                    "ci_high": float(record.get("ci_high", float("nan"))),
                    "mde": float(record.get("mde", float("nan"))),
                    "p_adjusted": float(record.get("p_adjusted", float("nan"))),
                    "n": int(record.get("n", 0)),
                }
            )
    return rows


def generate_research_article_report(  # pylint: disable=too-many-locals
    dataset: DatasetDict,
    output_dir: Path | str,
    *,
    heatmap_bins: int = 10,
) -> Dict[str, object]:
    """Generate heatmaps and Markdown summarising opinion shifts per study."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    deduped = dedupe_by_participant_issue(dataset)
    combined = dataframe_from_splits(deduped)

    study_entries: List[Dict[str, object]] = []
    heatmap_paths: List[Path] = []
    table_rows: List[Dict[str, float]] = []

    for spec in assemble_study_specs():
        study_frame = prepare_study_frame(combined, spec)
        hist, edges = histogram2d_counts(
            study_frame,
            spec.before_column,
            spec.after_column,
            heatmap_bins,
        )
        heatmap_path = output_path / spec.heatmap_filename
        plot_heatmap(hist, edges, spec.label, heatmap_path)
        heatmap_paths.append(heatmap_path)
        summary = summarise_shift(study_frame, spec.before_column, spec.after_column)
        study_entries.append(
            {
                "spec": spec,
                "summary": summary,
                "frame": study_frame,
                "heatmap_path": heatmap_path,
            }
        )

    assignment_panels: List[tuple[str, List[Dict[str, float]]]] = []
    regression_frames: List[pd.DataFrame] = []

    for entry in study_entries:
        spec = entry["spec"]
        assignment_frame = load_assignment_frame(spec)
        summaries = summarise_assignments(assignment_frame)
        assignment_panels.append((spec.label, summaries))
        control_entry = next(
            (item for item in summaries if item["assignment"] == "control"),
            None,
        )
        treatment_entry = next(
            (item for item in summaries if item["assignment"] == "treatment"),
            None,
        )
        if control_entry or treatment_entry:
            table_rows.append(
                {
                    "label": spec.label,
                    "control_mean_change": (
                        control_entry["mean_change"] if control_entry else float("nan")
                    ),
                    "treatment_mean_change": (
                        treatment_entry["mean_change"]
                        if treatment_entry
                        else float("nan")
                    ),
                }
            )
        if not assignment_frame.empty:
            regression_frames.append(
                assignment_frame.assign(study_key=spec.key, study_label=spec.label)
            )

    regression_df = (
        pd.concat(regression_frames, ignore_index=True) if regression_frames else pd.DataFrame()
    )
    regression_stats = compute_treatment_regression(regression_df)

    stratified_rows: List[Dict[str, object]] = []
    stratified_paths = analyze_preregistered_effects(output_path)
    combined_path = stratified_paths.get("combined") if stratified_paths else None
    if combined_path and Path(combined_path).exists():
        stratified_df = pd.read_csv(combined_path)
        if not stratified_df.empty:
            stratified_rows = _policy_summary_rows(stratified_df)
    else:
        stratified_df = pd.DataFrame()

    mean_change_plot = output_path / "mean_opinion_change.png"
    plot_assignment_panels(
        study_panels=assignment_panels,
        regression=regression_stats,
        output_path=mean_change_plot,
    )

    markdown_lines = build_markdown(
        output_dir=output_path,
        study_rows=[
            {"label": entry["spec"].label, "summary": entry["summary"]}
            for entry in study_entries
        ],
        heatmap_paths=heatmap_paths,
        mean_change_path=mean_change_plot,
        assignment_rows=table_rows,
        regression_summary=regression_stats,
        policy_rows=stratified_rows,
    )
    (output_path / "README.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )

    return {
        "summaries": {
            entry["spec"].label: entry["summary"] for entry in study_entries
        },
        "heatmaps": [str(path) for path in heatmap_paths],
        "mean_change_plot": str(mean_change_plot),
        "markdown": str(output_path / "README.md"),
        "stratified_paths": {key: str(path) for key, path in stratified_paths.items()},
        "stratified_policy_rows": stratified_rows,
    }
