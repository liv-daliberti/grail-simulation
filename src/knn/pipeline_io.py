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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .pipeline_context import StudySpec
from .pipeline_data import issue_slug_for_study

def load_metrics(run_dir: Path, issue_slug: str) -> tuple[Mapping[str, object], Path]:
    """
    Load the evaluation metrics JSON for a specific study/issue from ``run_dir``.

    :param run_dir: Directory containing per-issue subdirectories with metrics artefacts.
    :type run_dir: Path
    :param issue_slug: Issue slug used to locate the metrics file on disk.
    :type issue_slug: str
    :returns: Tuple containing the parsed metrics payload and the filesystem path used.
    :rtype: tuple[Mapping[str, object], Path]
    :raises FileNotFoundError: If the expected metrics file does not exist.
    """
    metrics_path = run_dir / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle), metrics_path

def load_opinion_metrics(out_dir: Path, feature_space: str) -> Dict[str, Mapping[str, object]]:
    """
    Collect opinion-task metrics for every study under ``out_dir``.

    :param out_dir: Root directory containing opinion evaluation artefacts.
    :type out_dir: Path
    :param feature_space: Feature space identifier (e.g. ``tfidf`` or ``word2vec``)
        used to scope the search.
    :type feature_space: str
    :returns: Mapping keyed by study slug with the deserialised metrics dictionary.
    :rtype: Dict[str, Mapping[str, object]]
    """
    result: Dict[str, Mapping[str, object]] = {}
    base_dir = out_dir / "opinion" / feature_space
    if not base_dir.exists():
        return result
    for study_dir in sorted(base_dir.iterdir()):
        if not study_dir.is_dir():
            continue
        metrics_path = study_dir / f"opinion_knn_{study_dir.name}_validation_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as handle:
            result[study_dir.name] = json.load(handle)
    return result

def load_final_metrics_from_disk(
    *,
    out_dir: Path,
    feature_spaces: Sequence[str],
    studies: Sequence[StudySpec],
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """
    Load slate metrics written by prior runs instead of recomputing them.

    :param out_dir: Directory where per-feature-space results are persisted.
    :type out_dir: Path
    :param feature_spaces: Iterable of feature space names to inspect.
    :type feature_spaces: Sequence[str]
    :param studies: Studies to look up within each feature directory.
    :type studies: Sequence[StudySpec]
    :returns: Nested mapping ``feature_space -> study -> metrics`` for all cached
        results found.
    :rtype: Dict[str, Dict[str, Mapping[str, object]]]
    """
    metrics_by_feature: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space in feature_spaces:
        feature_dir = out_dir / feature_space
        if not feature_dir.exists():
            continue
        per_study: Dict[str, Mapping[str, object]] = {}
        for study in studies:
            study_dir = feature_dir / study.study_slug
            try:
                metrics, _ = load_metrics(study_dir, issue_slug_for_study(study))
            except FileNotFoundError:
                continue
            per_study[study.key] = metrics
        if per_study:
            metrics_by_feature[feature_space] = per_study
    return metrics_by_feature

def load_loso_metrics_from_disk(
    *,
    out_dir: Path,
    feature_spaces: Sequence[str],
    studies: Sequence[StudySpec],
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """
    Load leave-one-study-out metrics produced by previous pipeline runs.

    :param out_dir: Directory containing persisted evaluation artefacts.
    :type out_dir: Path
    :param feature_spaces: Feature space configurations to inspect for cached
        LOSO runs.
    :type feature_spaces: Sequence[str]
    :param studies: Studies that were treated as hold-outs in the LOSO evaluation.
    :type studies: Sequence[StudySpec]
    :returns: Mapping ``feature_space -> study -> metrics`` for every cached
        LOSO evaluation located.
    :rtype: Dict[str, Dict[str, Mapping[str, object]]]
    """
    cross_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space in feature_spaces:
        loso_dir = out_dir / feature_space / "loso"
        if not loso_dir.exists():
            continue
        per_study: Dict[str, Mapping[str, object]] = {}
        for study in studies:
            holdout_dir = loso_dir / study.study_slug
            try:
                metrics, _ = load_metrics(holdout_dir, issue_slug_for_study(study))
            except FileNotFoundError:
                continue
            per_study[study.key] = metrics
        if per_study:
            cross_metrics[feature_space] = per_study
    return cross_metrics

__all__ = [
    "load_final_metrics_from_disk",
    "load_loso_metrics_from_disk",
    "load_metrics",
    "load_opinion_metrics",
]
