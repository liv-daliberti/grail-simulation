"""Filesystem helpers for loading persisted KNN metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .pipeline_context import StudySpec
from .pipeline_data import issue_slug_for_study


def load_metrics(run_dir: Path, issue_slug: str) -> tuple[Mapping[str, object], Path]:
    """Load the evaluation metrics JSON for ``issue_slug``."""

    metrics_path = run_dir / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle), metrics_path


def load_opinion_metrics(out_dir: Path, feature_space: str) -> Dict[str, Mapping[str, object]]:
    """Return metrics keyed by study for the opinion task."""

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
    """Load slate metrics written by prior runs instead of recomputing them."""

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
    """Load leave-one-study-out metrics produced by previous pipeline runs."""

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
