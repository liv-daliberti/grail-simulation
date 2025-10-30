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

"""CSV export helpers for next-video report metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

from ...context import StudySpec
from ...utils import extract_metric_summary
from .helpers import iter_metric_payloads


def _write_next_video_metrics_csv(
    output_dir: Path,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> None:
    """
    Write per-study next-video metrics to metrics.csv for all feature spaces.

    :param output_dir: Directory where ``metrics.csv`` should be written.
    :param metrics_by_feature: Nested mapping of metrics keyed by feature space and study.
    :param studies: Ordered study specifications used for label resolution.
    """

    if not metrics_by_feature:
        return
    fieldnames = [
        "feature_space",
        "study",
        "accuracy",
        "accuracy_all_rows",
        "baseline_accuracy",
        "random_baseline_accuracy",
        "best_k",
        "eligible",
        "total",
        "accuracy_ci_low",
        "accuracy_ci_high",
    ]
    with open(output_dir / "metrics.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature_space, per_feature in metrics_by_feature.items():
            for study, summary, payload in iter_metric_payloads(per_feature, studies):
                confidence_interval = summary.accuracy_ci
                writer.writerow(
                    {
                        "feature_space": feature_space,
                        "study": study.label,
                        "accuracy": summary.accuracy,
                        "accuracy_all_rows": payload.get("accuracy_overall_all_rows"),
                        "baseline_accuracy": summary.baseline,
                        "random_baseline_accuracy": summary.random_baseline,
                        "best_k": summary.best_k,
                        "eligible": summary.n_eligible,
                        "total": summary.n_total,
                        "accuracy_ci_low": (
                            confidence_interval[0] if confidence_interval else None
                        ),
                        "accuracy_ci_high": (
                            confidence_interval[1] if confidence_interval else None
                        ),
                    }
                )


def _write_next_video_loso_csv(
    output_dir: Path,
    loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> None:
    """
    Write LOSO next-video metrics to loso_metrics.csv for all feature spaces.

    :param output_dir: Directory where ``loso_metrics.csv`` should be written.
    :param loso_metrics: Nested mapping of LOSO metrics keyed by feature space and study.
    :param studies: Ordered study specifications used for label resolution.
    """

    if not loso_metrics:
        return
    out_path = output_dir / "loso_metrics.csv"
    fieldnames = [
        "feature_space",
        "holdout_study",
        "accuracy",
        "delta_vs_baseline",
        "baseline_accuracy",
        "best_k",
        "eligible",
    ]
    study_by_key = {spec.key: spec for spec in studies}
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature_space, per_feature in loso_metrics.items():
            for study_key, payload in per_feature.items():
                spec = study_by_key.get(study_key)
                if spec is None:
                    continue
                summary = extract_metric_summary(payload)
                delta = (
                    summary.accuracy - summary.baseline
                    if summary.accuracy is not None and summary.baseline is not None
                    else None
                )
                writer.writerow(
                    {
                        "feature_space": feature_space,
                        "holdout_study": spec.label,
                        "accuracy": summary.accuracy,
                        "delta_vs_baseline": delta,
                        "baseline_accuracy": summary.baseline,
                        "best_k": summary.best_k,
                        "eligible": summary.n_eligible,
                    }
                )


__all__ = [
    "_write_next_video_loso_csv",
    "_write_next_video_metrics_csv",
]
