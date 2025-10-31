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

"""Curve utilities and plotting helpers for KNN evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from common.evaluation.utils import safe_div
from common.visualization.matplotlib import plt


def compute_auc_from_curve(
    k_values: Sequence[int],
    accuracy_by_k: Dict[int, float],
) -> Tuple[float, float]:
    """
    Compute the area under the accuracy curve across ``k`` values.

    :param k_values: Iterable of evaluated ``k`` values.
    :param accuracy_by_k: Accuracy observed for each ``k``.
    :returns: Tuple of (area, normalised_area).
    """

    if not k_values:
        return 0.0, 0.0
    sorted_k = sorted({int(k) for k in k_values})
    accuracy_values = [float(accuracy_by_k.get(k, 0.0)) for k in sorted_k]
    if len(sorted_k) == 1:
        value = accuracy_values[0]
        return value, value
    area = float(np.trapz(accuracy_values, sorted_k))
    span = float(sorted_k[-1] - sorted_k[0]) or 1.0
    return area, area / span


def plot_elbow(
    k_values: Sequence[int],
    accuracy_by_k: Mapping[int, float],
    best_k: int,
    output_path: Path,
    *,
    data_split: str = "validation",
) -> None:
    """
    Plot the accuracy curve (acc@k) for the evaluation stage and mark ``best_k``.

    :param k_values: Iterable of ``k`` values that were evaluated.
    :param accuracy_by_k: Accuracy keyed by ``k`` for the evaluation split.
    :param best_k: Selected ``k`` to highlight in the plot.
    :param output_path: Destination path for the generated PNG artefact.
    :param data_split: Split label used for the legend title.
    """

    if plt is None:  # pragma: no cover - optional dependency
        logging.warning("[KNN] Skipping elbow plot (matplotlib not installed)")
        return

    if not k_values:
        logging.warning("[KNN] Skipping elbow plot (no k values supplied)")
        return

    sorted_k = list(k_values)
    plt.figure(figsize=(6, 4))
    error_rates = [1.0 - float(accuracy_by_k.get(k, 0.0)) for k in sorted_k]
    plt.plot(sorted_k, error_rates, marker="o", label="Error rate")
    if best_k in accuracy_by_k:
        best_error = 1.0 - float(accuracy_by_k[best_k])
        plt.axvline(best_k, color="red", linestyle="--", alpha=0.6)
        plt.scatter([best_k], [best_error], color="red", label="Selected k")
    split_label = data_split.strip() or "validation"
    plt.title(f"KNN error vs k ({split_label} split)")
    plt.xlabel("k")
    plt.ylabel("Error rate")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles, labels)
    plt.figtext(
        0.5,
        -0.05,
        f"Error computed on {split_label} data (eligible examples only)",
        ha="center",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def curve_summary(
    *,
    k_values: Sequence[int],
    accuracy_by_k: Dict[int, float],
    per_k_stats: Dict[int, Dict[str, int]],
    best_k: int,
    n_examples: int,
) -> Dict[str, object]:
    """
    Return a serialisable summary for accuracy-vs-k curves.

    :param k_values: Iterable of ``k`` values to evaluate or report.
    :param accuracy_by_k: Mapping from each ``k`` to its measured validation accuracy.
    :param per_k_stats: Detailed per-``k`` statistics derived from the evaluation curve.
    :param best_k: Neighbourhood size selected as optimal for the evaluation.
    :param n_examples: Total number of evaluation examples summarised in the bundle.
    :returns: Serialised curve summary including accuracy, eligibility, and AUC.
    """

    area, normalised = compute_auc_from_curve(k_values, accuracy_by_k)
    sorted_k = sorted({int(k) for k in k_values})
    accuracy_serialised = {
        str(k): float(accuracy_by_k.get(k, 0.0))
        for k in sorted_k
    }
    eligible_serialised = {
        str(k): int(per_k_stats[k]["eligible"])
        for k in sorted_k
    }
    correct_serialised = {
        str(k): int(per_k_stats[k]["correct"])
        for k in sorted_k
    }
    return {
        "accuracy_by_k": accuracy_serialised,
        "eligible_by_k": eligible_serialised,
        "correct_by_k": correct_serialised,
        "auc_area": float(area),
        "auc_normalized": float(normalised),
        "best_k": int(best_k),
        "best_accuracy": float(accuracy_by_k.get(best_k, 0.0)),
        "n_examples": int(n_examples),
    }


@dataclass(frozen=True)
class ValidationLogContext:
    """Bundle containing the data required to log validation summaries."""

    issue_slug: str
    feature_space: str
    best_k: int
    accuracy_by_k: Mapping[int, float]
    per_k_stats: Mapping[int, Mapping[str, int]]
    n_examples: int


def log_validation_summary(context: ValidationLogContext) -> None:
    """
    Log the primary validation metrics captured during evaluation.

    :param context: Summary bundle describing the validation phase output.
    """

    stats = context.per_k_stats.get(context.best_k, {})
    eligible = int(stats.get("eligible", 0))
    correct = int(stats.get("correct", 0))
    evaluated = int(context.n_examples)
    accuracy = float(context.accuracy_by_k.get(context.best_k, 0.0))
    coverage = safe_div(eligible, evaluated) if evaluated else 0.0
    logging.info(
        "[KNN][%s][%s] best_k=%d accuracy=%.4f coverage=%.4f (eligible=%d/%d, correct=%d)",
        context.feature_space,
        context.issue_slug,
        context.best_k,
        accuracy,
        coverage,
        eligible,
        evaluated,
        correct,
    )


__all__ = [
    "compute_auc_from_curve",
    "curve_summary",
    "ValidationLogContext",
    "log_validation_summary",
    "plot_elbow",
]
