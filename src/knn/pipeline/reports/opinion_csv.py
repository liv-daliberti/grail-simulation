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

"""CSV export utilities for the opinion pipeline report."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

from common.opinion.metrics import OPINION_CSV_BASE_FIELDS, build_opinion_csv_base_row

from ..context import StudySpec
from ..utils import extract_opinion_summary

__all__ = ["_write_knn_opinion_csv"]


def _write_knn_opinion_csv(
    output_dir: Path,
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> None:
    """
    Write KNN opinion metrics to ``opinion_metrics.csv`` across feature spaces.

    :param output_dir: Directory in which to place the generated CSV file.
    :type output_dir: Path
    :param metrics: Nested mapping of feature space -> study key -> raw metrics payload.
    :type metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param studies: Sequence of declared study specifications.
    :type studies: Sequence[~knn.pipeline.context.StudySpec]
    :returns: ``None``. The CSV file is written to disk as a side effect.
    :rtype: None
    """

    if not metrics:
        return
    out_path = output_dir / "opinion_metrics.csv"
    fieldnames = ["feature_space", *OPINION_CSV_BASE_FIELDS]
    study_by_key = {spec.key: spec for spec in studies}
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature_space, per_feature in metrics.items():
            for study_key, payload in per_feature.items():
                spec = study_by_key.get(study_key)
                if spec is None:
                    continue
                summary = extract_opinion_summary(payload)
                row = dict(
                    build_opinion_csv_base_row(summary, study_label=spec.label)
                )
                row["feature_space"] = feature_space
                writer.writerow(row)
