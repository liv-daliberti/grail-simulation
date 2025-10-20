"""Unit tests for the political sciences replication helpers in clean_data."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from clean_data.research_article_political_sciences.analysis import (
    assemble_study_specs,
    dataframe_from_splits,
    histogram2d_counts,
    prepare_study_frame,
    summarise_shift,
    to_numeric,
)
from clean_data.research_article_political_sciences.markdown import build_markdown
from clean_data.research_article_political_sciences.plotting import plot_heatmap, plot_mean_change
from clean_data.research_article_political_sciences.report import generate_research_article_report

pytestmark = pytest.mark.clean_data


def test_dataframe_from_splits_combines_rows():
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"participant_study": ["study1"], "issue": ["gun_control"]}),
            "validation": Dataset.from_dict({"participant_study": ["study2"], "issue": ["minimum_wage"]}),
        }
    )
    combined = dataframe_from_splits(dataset)
    assert sorted(combined["participant_study"].tolist()) == ["study1", "study2"]


def test_dataframe_from_splits_empty_dataset_returns_empty_frame():
    dataset = DatasetDict({})
    combined = dataframe_from_splits(dataset)
    assert combined.empty


def test_prepare_study_frame_filters_and_converts_numeric():
    frame = pd.DataFrame(
        {
            "participant_study": ["study1", "study1", "study1"],
            "issue": ["gun_control", "gun_control", "minimum_wage"],
            "gun_index": ["0.25", "not-a-number", "0.8"],
            "gun_index_2": ["0.50", "0.75", ""],
        }
    )
    spec = next(spec for spec in assemble_study_specs() if spec.key == "study1")
    result = prepare_study_frame(frame, spec)
    assert len(result) == 1
    assert result.iloc[0][spec.before_column] == pytest.approx(0.25)
    assert result.iloc[0][spec.after_column] == pytest.approx(0.5)


def test_histogram2d_counts_handles_non_empty_frame():
    df = pd.DataFrame({"pre": [0.1, 0.9], "post": [0.2, 0.8]})
    hist, edges = histogram2d_counts(df, "pre", "post", bins=2)
    assert hist.shape == (2, 2)
    assert hist.sum() == 2
    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(1.0)


def test_histogram2d_counts_returns_zero_matrix_for_empty_frame():
    df = pd.DataFrame(columns=["pre", "post"])
    hist, edges = histogram2d_counts(df, "pre", "post", bins=3)
    assert hist.shape == (3, 3)
    assert not hist.any()
    np.testing.assert_allclose(edges, np.linspace(0.0, 1.0, 4))


def test_summarise_shift_returns_nan_for_empty_frame():
    summary = summarise_shift(pd.DataFrame(columns=["pre", "post"]), "pre", "post")
    assert summary["n"] == 0
    for key, value in summary.items():
        if key != "n":
            assert math.isnan(value)


def test_summarise_shift_computes_expected_statistics():
    df = pd.DataFrame({"pre": [0.1, 0.4, 0.3], "post": [0.2, 0.6, 0.1]})
    summary = summarise_shift(df, "pre", "post")
    changes = np.array([0.1, 0.2, -0.2])
    assert summary["n"] == pytest.approx(3.0)
    assert summary["mean_change"] == pytest.approx(changes.mean())
    assert summary["median_change"] == pytest.approx(np.median(changes))
    assert summary["std_change"] == pytest.approx(np.std(changes, ddof=1))
    assert summary["share_increase"] == pytest.approx(np.mean(changes > 0.0))
    assert summary["share_decrease"] == pytest.approx(np.mean(changes < 0.0))
    assert summary["share_small_change"] == pytest.approx(np.mean(np.abs(changes) <= 0.05))


def test_to_numeric_preserves_nans():
    series = pd.Series(["0.1", "not-a-number", None])
    result = to_numeric(series)
    assert math.isnan(result.iloc[1])
    assert math.isnan(result.iloc[2])


def test_build_markdown_formats_nan_entries(tmp_path):
    output_dir = tmp_path / "report"
    output_dir.mkdir()

    heatmap_path = output_dir / "heatmap_study1.png"
    heatmap_path.write_text("", encoding="utf-8")
    mean_change_path = output_dir / "mean_change.png"
    mean_change_path.write_text("", encoding="utf-8")

    study_rows = [
        {
            "label": "Study 1 – Example",
            "summary": {
                "n": 12.0,
                "mean_before": 0.2,
                "mean_after": 0.25,
                "mean_change": 0.05,
                "median_change": 0.05,
                "std_change": 0.01,
                "share_increase": float("nan"),
                "share_decrease": 0.2,
                "share_small_change": float("nan"),
            },
        }
    ]

    lines = build_markdown(
        output_dir=output_dir,
        study_rows=study_rows,
        heatmap_paths=[heatmap_path],
        mean_change_path=mean_change_path,
    )

    table_line = next(line for line in lines if line.startswith("| Study 1 – Example"))
    assert "| 12 |" in table_line
    assert "n/a" in table_line
    assert lines[-1].startswith("Replication notes")


def test_plot_heatmap_creates_png(tmp_path):
    output_path = tmp_path / "figures" / "heatmap.png"
    hist = np.array([[2, 0], [1, 3]])
    edges = np.linspace(0.0, 1.0, 3)
    plot_heatmap(hist, edges, "Test Heatmap", output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_mean_change_handles_empty_input(tmp_path):
    output_path = tmp_path / "figures" / "mean.png"
    plot_mean_change([], [], output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_mean_change_with_data(tmp_path):
    output_path = tmp_path / "figures" / "mean_non_empty.png"
    summaries = [
        {"mean_change": 0.05, "std_change": 0.02, "n": 10.0},
        {"mean_change": -0.03, "std_change": 0.01, "n": 8.0},
    ]
    labels = ["Study A", "Study B"]
    plot_mean_change(summaries, labels, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_research_article_report_creates_artifacts(tmp_path):
    train = Dataset.from_dict(
        {
            "participant_id": ["p1", "p1", "p4", "p2", "p5", "p3", "p6"],
            "participant_study": ["study1", "study1", "study1", "study2", "study2", "study3", "study3"],
            "issue": [
                "gun_control",
                "gun_control",
                "gun_control",
                "minimum_wage",
                "minimum_wage",
                "minimum_wage",
                "minimum_wage",
            ],
            "gun_index": [0.2, 0.6, 0.3, float("nan"), float("nan"), float("nan"), float("nan")],
            "gun_index_2": [0.4, 0.7, 0.45, float("nan"), float("nan"), float("nan"), float("nan")],
            "mw_index_w1": [float("nan"), float("nan"), float("nan"), 0.35, 0.4, 0.55, 0.6],
            "mw_index_w2": [float("nan"), float("nan"), float("nan"), 0.45, 0.5, 0.65, 0.7],
        }
    )
    dataset = DatasetDict({"train": train})

    result = generate_research_article_report(dataset, tmp_path, heatmap_bins=2)

    study_labels = [spec.label for spec in assemble_study_specs()]
    assert set(result["summaries"]) == set(study_labels)
    for path in result["heatmaps"]:
        assert Path(path).exists()
    assert Path(result["mean_change_plot"]).exists()
    markdown_path = Path(result["markdown"])
    assert markdown_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Study 1 – Gun Control" in markdown

    summary = result["summaries"]["Study 1 – Gun Control (MTurk)"]
    assert summary["n"] == pytest.approx(2.0)
    assert summary["mean_before"] == pytest.approx(np.mean([0.2, 0.3]))
