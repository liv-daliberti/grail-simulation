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

"""Plotting helpers for the political sciences replication report.

These functions generate the heatmaps and opinion-shift bar charts used to
replicate the published study's figures. The visualisation utilities are
licensed under the repository's Apache 2.0 terms; see LICENSE for details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


def _ensure_output_dir(path: Path) -> None:
    """Create parent directories for ``path`` if they do not already exist.

    :param path: Target output file path used by the plotting helpers.
    """

    path.parent.mkdir(parents=True, exist_ok=True)


def plot_mean_change(  # pylint: disable=too-many-locals
    summaries: Iterable[Mapping[str, float]],
    labels: Iterable[str],
    output_path: Path,
) -> None:
    """Render mean opinion shifts with 95% CI-style bars (σ/√n).

    :param summaries: Iterable of summary dictionaries with ``mean_change``,
        ``std_change``, and ``n`` keys.
    :param labels: Iterable of x-axis labels aligned with ``summaries``.
    :param output_path: Destination path where the plot image is written.
    """

    output_path = Path(output_path)
    _ensure_output_dir(output_path)

    summaries = list(summaries)
    labels = list(labels)

    fig, axes_obj = plt.subplots(figsize=(8, 4.5))

    if not summaries:
        axes_obj.text(
            0.5,
            0.5,
            "No summary statistics available",
            ha="center",
            va="center",
            fontsize=11,
        )
        axes_obj.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return

    if labels and len(labels) != len(summaries):
        raise ValueError("labels length must match summaries length")

    means = []
    stderr = []
    annotations = []

    for summary in summaries:
        mean_change = summary.get("mean_change", float("nan"))
        std_change = summary.get("std_change", float("nan"))
        sample_size = summary.get("n", float("nan"))

        means.append(mean_change)

        if np.isnan(std_change) or np.isnan(sample_size) or sample_size <= 0:
            stderr.append(float("nan"))
        else:
            stderr.append(std_change / np.sqrt(sample_size))

        if sample_size and not np.isnan(sample_size):
            annotations.append(f"n={int(sample_size)}")
        else:
            annotations.append("n/a")

    positions = np.arange(len(summaries))
    yerr = np.array([stderr, stderr])

    bars = axes_obj.bar(
        positions,
        means,
        yerr=yerr,
        capsize=6,
        color="#4c72b0",
        alpha=0.9,
    )
    axes_obj.axhline(0.0, color="#222222", linewidth=1, linestyle="--")

    tick_labels = labels if labels else [str(idx + 1) for idx in range(len(summaries))]
    axes_obj.set_xticks(positions)
    axes_obj.set_xticklabels(tick_labels, rotation=15, ha="right")
    axes_obj.set_ylabel("Mean Δ (post - pre)")
    axes_obj.set_title("Mean opinion change by study")

    for bar_patch, note in zip(bars, annotations):
        height = bar_patch.get_height()
        axes_obj.text(
            bar_patch.get_x() + bar_patch.get_width() / 2,
            height,
            note,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_heatmap(
    hist: np.ndarray,
    bin_edges: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Render a heatmap from a 2D histogram of pre/post opinion indices.

    :param hist: 2D histogram counts (before x after).
    :param bin_edges: Bin edges used along both axes.
    :param title: Plot title describing the study context.
    :param output_path: Destination path where the heatmap is saved.
    """

    output_path = Path(output_path)
    _ensure_output_dir(output_path)
    fig, axes_obj = plt.subplots(figsize=(6, 5))

    mesh = axes_obj.imshow(
        hist.T,
        origin="lower",
        extent=[
            bin_edges[0],
            bin_edges[-1],
            bin_edges[0],
            bin_edges[-1],
        ],
        aspect="auto",
        cmap="magma",
    )
    axes_obj.set_xlabel("Pre-study opinion index")
    axes_obj.set_ylabel("Post-study opinion index")
    axes_obj.set_title(title)
    fig.colorbar(mesh, ax=axes_obj, label="Participants")

    # Overlay bin counts to make low-volume cells interpretable.
    if hist.size:
        x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        y_centers = x_centers
        for row_idx, x_center in enumerate(x_centers):
            for col_idx, y_center in enumerate(y_centers):
                value = hist[row_idx, col_idx]
                if value:
                    axes_obj.text(
                        x_center,
                        y_center,
                        f"{value}",
                        color="white",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_assignment_panels(  # pylint: disable=too-many-locals,too-many-statements
    study_panels: Iterable[tuple[str, Iterable[Mapping[str, float]]]],
    regression: Mapping[str, float],
    output_path: Path,
) -> None:
    """Render control/treatment mean-change panels with a regression summary.

    :param study_panels: Iterable pairing panel titles with per-assignment summaries.
    :param regression: Dictionary containing pooled regression metrics.
    :param output_path: Destination path where the composite figure is saved.
    """

    output_path = Path(output_path)
    _ensure_output_dir(output_path)

    panels = [(label, list(items)) for label, items in study_panels]
    fig, axes_grid = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes_grid.flatten()

    colors = {
        "control": "#7f7f7f",
        "treatment": "#4c72b0",
    }

    for idx, (label, entries) in enumerate(panels):
        panel_axes = axes_flat[idx]
        if not entries:
            panel_axes.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=11)
            panel_axes.set_title(label)
            panel_axes.axis("off")
            continue

        sorted_entries = sorted(
            entries,
            key=lambda item: (
                0 if item["assignment"] == "control" else 1,
                item["assignment"],
            ),
        )
        means = [entry["mean_change"] for entry in sorted_entries]
        confidence_intervals = [entry.get("ci95", float("nan")) for entry in sorted_entries]
        n_values = [entry.get("n", 0) for entry in sorted_entries]
        assign_names = [entry["assignment"] for entry in sorted_entries]
        colors_used = [colors.get(name, "#55a868") for name in assign_names]

        positions = np.arange(len(sorted_entries))
        yerr = np.zeros((2, len(sorted_entries)))
        for i, value in enumerate(confidence_intervals):
            if value is not None and not np.isnan(value):
                yerr[0, i] = value
                yerr[1, i] = value

        panel_axes.bar(
            positions,
            means,
            yerr=yerr,
            color=colors_used,
            alpha=0.9,
            capsize=6,
        )
        panel_axes.axhline(0.0, color="#222222", linewidth=1, linestyle="--")
        tick_labels = [
            f"{name.title()} (n={int(n)})" for name, n in zip(assign_names, n_values)
        ]
        panel_axes.set_xticks(positions)
        panel_axes.set_xticklabels(tick_labels, rotation=15, ha="right")
        panel_axes.set_ylabel("Mean Δ (post - pre)")
        panel_axes.set_title(label)

    # Regression panel sits in the final subplot.
    regression_axes = axes_flat[3]
    coef = regression.get("coefficient", float("nan"))
    ci_low = regression.get("ci_low", float("nan"))
    ci_high = regression.get("ci_high", float("nan"))
    p_value = regression.get("p_value", float("nan"))

    if np.isnan(coef):
        regression_axes.text(
            0.5,
            0.5,
            "Regression unavailable",
            ha="center",
            va="center",
            fontsize=11,
        )
        regression_axes.axis("off")
    else:
        lower_err = coef - ci_low if not np.isnan(ci_low) else 0.0
        upper_err = ci_high - coef if not np.isnan(ci_high) else 0.0
        yerr = np.array([[max(lower_err, 0.0)], [max(upper_err, 0.0)]])
        regression_axes.errorbar(
            [0],
            [coef],
            yerr=yerr,
            fmt="o",
            color="#2ca02c",
            ecolor="#2ca02c",
            capsize=6,
        )
        regression_axes.axhline(0.0, color="#222222", linewidth=1, linestyle="--")
        regression_axes.set_xticks([0])
        regression_axes.set_xticklabels(["Treatment effect"])
        regression_axes.set_xlim(-0.75, 0.75)
        regression_axes.set_ylabel("β (Δ opinion)")
        regression_axes.set_title("Regression: Treatment vs. Control")
        regression_axes.grid(axis="y", alpha=0.2, linestyle="--")

        text_lines = [f"β = {coef:.3f}"]
        if not np.isnan(ci_low) and not np.isnan(ci_high):
            text_lines.append(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
        if not np.isnan(p_value):
            text_lines.append(f"p = {p_value:.3f}")
        regression_axes.text(
            0.02,
            0.95,
            "\n".join(text_lines),
            transform=regression_axes.transAxes,
            va="top",
            ha="left",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
