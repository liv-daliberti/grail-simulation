"""Top-level orchestration for the political sciences replication report."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from datasets import DatasetDict

from clean_data.clean_data import dedupe_by_participant_issue

from .analysis import (
    assemble_study_specs,
    dataframe_from_splits,
    histogram2d_counts,
    prepare_study_frame,
    summarise_shift,
)
from .markdown import build_markdown
from .plotting import plot_heatmap, plot_mean_change


def generate_research_article_report(
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

    mean_change_plot = output_path / "mean_opinion_change.png"
    plot_mean_change(
        summaries=[entry["summary"] for entry in study_entries],
        labels=[entry["spec"].label for entry in study_entries],
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
    }
