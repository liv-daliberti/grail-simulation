"""Plotting helpers for the political sciences replication report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


def _ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_mean_change(  # pylint: disable=too-many-locals
    summaries: Iterable[Mapping[str, float]],
    labels: Iterable[str],
    output_path: Path,
) -> None:
    """Render mean opinion shifts with 95% CI-style bars (σ/√n)."""

    output_path = Path(output_path)
    _ensure_output_dir(output_path)

    summaries = list(summaries)
    labels = list(labels)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if not summaries:
        ax.text(0.5, 0.5, "No summary statistics available", ha="center", va="center", fontsize=11)
        ax.axis("off")
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
        n = summary.get("n", float("nan"))

        means.append(mean_change)

        if np.isnan(std_change) or np.isnan(n) or n <= 0:
            stderr.append(float("nan"))
        else:
            stderr.append(std_change / np.sqrt(n))

        if n and not np.isnan(n):
            annotations.append(f"n={int(n)}")
        else:
            annotations.append("n/a")

    positions = np.arange(len(summaries))
    yerr = np.array([stderr, stderr])

    bars = ax.bar(
        positions,
        means,
        yerr=yerr,
        capsize=6,
        color="#4c72b0",
        alpha=0.9,
    )
    ax.axhline(0.0, color="#222222", linewidth=1, linestyle="--")

    tick_labels = labels if labels else [str(idx + 1) for idx in range(len(summaries))]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=15, ha="right")
    ax.set_ylabel("Mean Δ (post - pre)")
    ax.set_title("Mean opinion change by study")

    for bar_patch, note in zip(bars, annotations):
        height = bar_patch.get_height()
        ax.text(
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
    """Render a heatmap from a 2D histogram of pre/post opinion indices."""

    output_path = Path(output_path)
    _ensure_output_dir(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))

    mesh = ax.imshow(
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
    ax.set_xlabel("Pre-study opinion index")
    ax.set_ylabel("Post-study opinion index")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="Participants")

    # Overlay bin counts to make low-volume cells interpretable.
    if hist.size:
        x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        y_centers = x_centers
        for i, x in enumerate(x_centers):
            for j, y in enumerate(y_centers):
                value = hist[i, j]
                if value:
                    ax.text(
                        x,
                        y,
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
    """Render control/treatment mean-change panels with a regression summary."""

    output_path = Path(output_path)
    _ensure_output_dir(output_path)

    panels = [(label, list(items)) for label, items in study_panels]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.flatten()

    colors = {
        "control": "#7f7f7f",
        "treatment": "#4c72b0",
    }

    for idx, (label, entries) in enumerate(panels):
        ax = axes_flat[idx]
        if not entries:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=11)
            ax.set_title(label)
            ax.axis("off")
            continue

        sorted_entries = sorted(
            entries,
            key=lambda item: (
                0 if item["assignment"] == "control" else 1,
                item["assignment"],
            ),
        )
        means = [entry["mean_change"] for entry in sorted_entries]
        ci = [entry.get("ci95", float("nan")) for entry in sorted_entries]
        n_values = [entry.get("n", 0) for entry in sorted_entries]
        assign_names = [entry["assignment"] for entry in sorted_entries]
        colors_used = [colors.get(name, "#55a868") for name in assign_names]

        positions = np.arange(len(sorted_entries))
        yerr = np.zeros((2, len(sorted_entries)))
        for i, value in enumerate(ci):
            if value is not None and not np.isnan(value):
                yerr[0, i] = value
                yerr[1, i] = value

        ax.bar(
            positions,
            means,
            yerr=yerr,
            color=colors_used,
            alpha=0.9,
            capsize=6,
        )
        ax.axhline(0.0, color="#222222", linewidth=1, linestyle="--")
        tick_labels = [
            f"{name.title()} (n={int(n)})" for name, n in zip(assign_names, n_values)
        ]
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=15, ha="right")
        ax.set_ylabel("Mean Δ (post - pre)")
        ax.set_title(label)

    # Regression panel sits in the final subplot.
    reg_ax = axes_flat[3]
    coef = regression.get("coefficient", float("nan"))
    ci_low = regression.get("ci_low", float("nan"))
    ci_high = regression.get("ci_high", float("nan"))
    p_value = regression.get("p_value", float("nan"))

    if np.isnan(coef):
        reg_ax.text(0.5, 0.5, "Regression unavailable", ha="center", va="center", fontsize=11)
        reg_ax.axis("off")
    else:
        lower_err = coef - ci_low if not np.isnan(ci_low) else 0.0
        upper_err = ci_high - coef if not np.isnan(ci_high) else 0.0
        yerr = np.array([[max(lower_err, 0.0)], [max(upper_err, 0.0)]])
        reg_ax.errorbar(
            [0],
            [coef],
            yerr=yerr,
            fmt="o",
            color="#2ca02c",
            ecolor="#2ca02c",
            capsize=6,
        )
        reg_ax.axhline(0.0, color="#222222", linewidth=1, linestyle="--")
        reg_ax.set_xticks([0])
        reg_ax.set_xticklabels(["Treatment effect"])
        reg_ax.set_xlim(-0.75, 0.75)
        reg_ax.set_ylabel("β (Δ opinion)")
        reg_ax.set_title("Regression: Treatment vs. Control")
        reg_ax.grid(axis="y", alpha=0.2, linestyle="--")

        text_lines = [f"β = {coef:.3f}"]
        if not np.isnan(ci_low) and not np.isnan(ci_high):
            text_lines.append(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
        if not np.isnan(p_value):
            text_lines.append(f"p = {p_value:.3f}")
        reg_ax.text(
            0.02,
            0.95,
            "\n".join(text_lines),
            transform=reg_ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
