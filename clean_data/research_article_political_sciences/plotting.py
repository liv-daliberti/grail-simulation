"""Plotting helpers for the political sciences replication report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def plot_mean_change(
    summaries: Iterable[Mapping[str, float]],
    labels: Iterable[str],
    output_path: Path,
) -> None:
    """Plot the mean opinion change per study with 95% CI error bars."""

    output_path = Path(output_path)
    _ensure_output_dir(output_path)
    summaries = list(summaries)
    labels = list(labels)

    if not summaries:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return

    means = np.array([entry["mean_change"] for entry in summaries], dtype=float)
    stds = np.array([entry["std_change"] for entry in summaries], dtype=float)
    counts = np.array([entry["n"] for entry in summaries], dtype=float)
    stderr = np.divide(stds, np.sqrt(np.maximum(counts, 1)), out=np.zeros_like(stds), where=counts > 0)
    ci95 = 1.96 * stderr

    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(means))
    ax.bar(positions, means, yerr=ci95, color="#4c72b0", alpha=0.85, capsize=6)
    ax.axhline(0.0, color="#222222", linewidth=1, linestyle="--")
    ax.set_ylabel("Mean post - pre opinion index")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Average opinion shift by study (95% CI)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
