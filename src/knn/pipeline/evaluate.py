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

"""Evaluation runners for the Grail Simulation KNN pipeline.

Wraps the KNN CLI so orchestration code can execute final evaluations,
leave-one-study-out checks, and opinion regressions while respecting the
pipeline's caching semantics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence, List

from common.pipeline.utils import (
    compose_cli_args,
    ensure_final_stage_overwrite_with_context,
)

from .context import EvaluationContext, OpinionStudySelection, StudySelection, StudySpec
from .data import issue_slug_for_study
from .io import (
    load_metrics,
    load_opinion_metrics,
    load_loso_metrics_from_disk as load_loso_metrics_from_disk,
)
from .sweeps import run_knn_cli
from .utils import ensure_dir

LOGGER = logging.getLogger("knn.pipeline.evaluate")


def _opinion_prediction_paths(base_dir: Path, feature_space: str, study_key: str) -> List[Path]:
    """
    Return candidate locations for cached opinion prediction archives.

    Older runs stored predictions under ``<out>/<feature>/`` whereas newer
    runs use ``<out>/opinion/<feature>/``. Report regeneration expects the
    JSONL artefact to exist in either location, so we probe both when
    deciding whether we can safely reuse cached metrics.
    """

    filename = f"opinion_knn_{study_key}_validation.jsonl"
    return [
        base_dir / "opinion" / feature_space / study_key / filename,
        base_dir / feature_space / study_key / filename,
    ]

def _maybe_word2vec_model_dir(
    *, feature_space: str, base_dir: Path, study_slug: str
) -> Path | None:
    """
    Return the Word2Vec cache directory for a study when applicable.

    This centralises the conditional logic for building an on-disk cache
    directory used by ``word2vec`` feature-space evaluations.

    :param feature_space: Feature space identifier (e.g. ``tfidf`` or ``word2vec``).
    :type feature_space: str
    :param base_dir: Root directory for Word2Vec caches for the current task.
    :type base_dir: ~pathlib.Path
    :param study_slug: Filesystem-friendly study slug.
    :type study_slug: str
    :returns: The materialised cache directory when ``feature_space`` is ``word2vec``; otherwise ``None``.
    :rtype: Optional[~pathlib.Path]
    """

    if feature_space != "word2vec":
        return None
    return ensure_dir(base_dir / study_slug)

def _load_cached_final_metrics(
    *,
    root: Path,
    issue_slug: str,
    metrics_path: Path,
    feature_space: str,
    study_key: str,
    reuse_existing: bool,
) -> Mapping[str, object] | None:
    """
    Attempt to load cached next-video metrics, logging outcomes consistently.

    :param root: Output directory where per-issue artefacts are written.
    :type root: ~pathlib.Path
    :param issue_slug: Normalised issue identifier used for filenames.
    :type issue_slug: str
    :param metrics_path: Expected JSON metrics path on disk.
    :type metrics_path: ~pathlib.Path
    :param feature_space: Feature space label used for logging.
    :type feature_space: str
    :param study_key: Study key used for logging.
    :type study_key: str
    :param reuse_existing: Whether cached artefacts should be reused when present.
    :type reuse_existing: bool
    :returns: Metrics mapping when successfully loaded; otherwise ``None``.
    :rtype: Optional[Mapping[str, object]]
    """

    if not (reuse_existing and metrics_path.exists()):
        return None
    try:
        metrics, _ = load_metrics(root, issue_slug)
    except FileNotFoundError:
        LOGGER.warning(
            (
                "[FINAL][MISS] feature=%s study=%s expected cached metrics at %s but none found."
            ),
            feature_space,
            study_key,
            metrics_path,
        )
        return None
    LOGGER.info(
        "[FINAL][SKIP] feature=%s study=%s (metrics cached).",
        feature_space,
        study_key,
    )
    return metrics

def run_final_evaluations(
    *,
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
    context: EvaluationContext,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """
    Run final slate evaluations and return metrics grouped by feature space.

    :param selections: Winning sweep selections keyed by feature space and study key.
    :type selections: Mapping[str, Mapping[str, ~knn.pipeline.context.StudySelection]]
    :param studies: Ordered list of studies to evaluate.
    :type studies: Sequence[~knn.pipeline.context.StudySpec]
    :param context: Shared CLI/runtime parameters used for all evaluations.
    :type context: ~knn.pipeline.context.EvaluationContext
    :returns: Nested mapping ``feature_space -> study_key -> metrics`` for the final evaluations.
    :rtype: Dict[str, Dict[str, Mapping[str, object]]]
    """
    metrics_by_feature: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space, per_study in selections.items():
        feature_metrics: Dict[str, Mapping[str, object]] = {}
        for study in studies:
            selection = per_study.get(study.key)
            if selection is None:
                continue
            LOGGER.info(
                "[FINAL] feature=%s study=%s issue=%s accuracy=%.3f",
                feature_space,
                study.key,
                study.issue,
                selection.accuracy,
            )
            feature_out_dir = ensure_dir(
                context.next_video_out_dir / feature_space / study.study_slug
            )
            issue_slug = issue_slug_for_study(study)
            metrics_path = (
                feature_out_dir / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
            )
            cached = _load_cached_final_metrics(
                root=feature_out_dir,
                issue_slug=issue_slug,
                metrics_path=metrics_path,
                feature_space=feature_space,
                study_key=study.key,
                reuse_existing=context.reuse_existing,
            )
            if cached is not None:
                feature_metrics[study.key] = cached
                continue
            args = compose_cli_args(
                context.base_cli,
                selection.config.cli_args(
                    word2vec_model_dir=_maybe_word2vec_model_dir(
                        feature_space=feature_space,
                        base_dir=context.next_video_word2vec_dir,
                        study_slug=study.study_slug,
                    )
                ),
                ["--issues", study.issue],
                ["--participant-studies", study.key],
                ["--train-participant-studies", study.key],
                ["--out-dir", str(feature_out_dir)],
                ["--knn-k", str(selection.best_k)],
                context.extra_cli,
            )
            ensure_final_stage_overwrite_with_context(
                args,
                metrics_path,
                logger=LOGGER,
                feature=feature_space,
                study=study.key,
            )
            run_knn_cli(args)
            metrics, _ = load_metrics(feature_out_dir, issue_slug)
            feature_metrics[study.key] = metrics
        if feature_metrics:
            metrics_by_feature[feature_space] = feature_metrics
    return metrics_by_feature

def run_opinion_evaluations(
    *,
    selections: Mapping[str, Mapping[str, OpinionStudySelection]],
    studies: Sequence[StudySpec],
    context: EvaluationContext,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """
    Run opinion regression for each feature space and return metrics.

    :param selections: Winning opinion selections keyed by feature space and study key.
    :type selections: Mapping[str, Mapping[str, ~knn.pipeline.context.OpinionStudySelection]]
    :param studies: Ordered list of opinion studies to evaluate.
    :type studies: Sequence[~knn.pipeline.context.StudySpec]
    :param context: Shared CLI/runtime parameters used for all evaluations.
    :type context: ~knn.pipeline.context.EvaluationContext
    :returns: Nested mapping ``feature_space -> study_key -> metrics`` for the opinion evaluations.
    :rtype: Dict[str, Dict[str, Mapping[str, object]]]
    """
    metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space, per_study in selections.items():
        LOGGER.info("[OPINION] feature=%s", feature_space)
        out_dir = ensure_dir(context.opinion_out_dir)
        cached_metrics = (
            load_opinion_metrics(out_dir, feature_space) if context.reuse_existing else {}
        )
        for study in studies:
            selection = per_study.get(study.key)
            if selection is None:
                continue
            predictions_cached = any(
                path.exists()
                for path in _opinion_prediction_paths(out_dir, feature_space, study.key)
            )
            if context.reuse_existing and study.key in cached_metrics and predictions_cached:
                LOGGER.info(
                    "[OPINION][SKIP] feature=%s study=%s (metrics cached).",
                    feature_space,
                    study.key,
                )
                continue
            if context.reuse_existing and study.key in cached_metrics and not predictions_cached:
                LOGGER.info(
                    "[OPINION][REFRESH] feature=%s study=%s cached metrics found but "
                    "predictions missing; rerunning evaluation.",
                    feature_space,
                    study.key,
                )
            LOGGER.info("[OPINION] study=%s issue=%s", study.key, study.issue)
            run_knn_cli(
                compose_cli_args(
                    context.base_cli,
                    selection.config.cli_args(
                        word2vec_model_dir=_maybe_word2vec_model_dir(
                            feature_space=feature_space,
                            base_dir=context.opinion_word2vec_dir,
                            study_slug=study.study_slug,
                        )
                    ),
                    ["--task", "opinion"],
                    ["--out-dir", str(out_dir)],
                    ["--knn-k", str(selection.best_k)],
                    ["--opinion-studies", study.key],
                    context.extra_cli,
                )
            )
        metrics[feature_space] = load_opinion_metrics(out_dir, feature_space)
    return metrics

def run_opinion_from_next_evaluations(
    *,
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
    context: EvaluationContext,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """
    Score opinion change using the next-video configuration.

    :param selections: Winning next-video selections keyed by feature space and study key.
    :type selections: Mapping[str, Mapping[str, ~knn.pipeline.context.StudySelection]]
    :param studies: Ordered list of opinion studies to evaluate.
    :type studies: Sequence[~knn.pipeline.context.StudySpec]
    :param context: Shared CLI/runtime parameters used for all evaluations.
    :type context: ~knn.pipeline.context.EvaluationContext
    :returns: Next-video metrics keyed by ``feature_space`` then ``study_key``.
    :rtype: Dict[str, Dict[str, Mapping[str, object]]]
    """

    if not selections:
        return {}

    base_out_dir = ensure_dir(context.opinion_out_dir / "from_next")
    metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}

    for feature_space, per_study in selections.items():
        LOGGER.info("[OPINION][FROM-NEXT] feature=%s", feature_space)
        cached_metrics = (
            load_opinion_metrics(base_out_dir, feature_space) if context.reuse_existing else {}
        )
        for study in studies:
            selection = per_study.get(study.key)
            if selection is None:
                continue
            predictions_cached = any(
                path.exists()
                for path in _opinion_prediction_paths(base_out_dir, feature_space, study.key)
            )
            if context.reuse_existing and study.key in cached_metrics and predictions_cached:
                LOGGER.info(
                    "[OPINION][FROM-NEXT][SKIP] feature=%s study=%s (metrics cached).",
                    feature_space,
                    study.key,
                )
                continue
            if context.reuse_existing and study.key in cached_metrics and not predictions_cached:
                LOGGER.info(
                    "[OPINION][FROM-NEXT][REFRESH] feature=%s study=%s cached metrics "
                    "found but predictions missing; rerunning evaluation.",
                    feature_space,
                    study.key,
                )
            LOGGER.info(
                "[OPINION][FROM-NEXT] study=%s issue=%s",
                study.key,
                study.issue,
            )
            run_knn_cli(
                compose_cli_args(
                    context.base_cli,
                    selection.config.cli_args(
                        word2vec_model_dir=_maybe_word2vec_model_dir(
                            feature_space=feature_space,
                            base_dir=context.next_video_word2vec_dir,
                            study_slug=study.study_slug,
                        )
                    ),
                    ["--task", "opinion"],
                    ["--out-dir", str(base_out_dir)],
                    ["--knn-k", str(selection.best_k)],
                    ["--opinion-studies", study.key],
                    context.extra_cli,
                )
            )
        refreshed = load_opinion_metrics(base_out_dir, feature_space)
        if refreshed:
            metrics[feature_space] = refreshed
    return metrics

def run_cross_study_evaluations(
    *,
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
    context: EvaluationContext,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """
    Run per-study evaluation sweeps grouped by feature space.

    Historically this performed leave-one-study-out (LOSO) checks, but training
    is now restricted to the holdout study to avoid cross-study mixtures. This
    implementation primarily supports reuse of cached metrics (sufficient for tests).

    :param selections: Mapping from feature space to per-study selections that
        determine which configuration to evaluate.
    :param studies: Ordered list of study specifications considered in LOSO checks.
    :param context: Evaluation context providing filesystem roots and flags.
    :returns: Nested mapping ``{feature_space: {study_key: metrics}}`` with any
        cached results found (may be empty when unavailable).
    """

    if len(studies) <= 1:
        LOGGER.warning("[LOSO] Skipping cross-study evaluations (no alternate studies).")
        return {}

    feature_spaces = tuple(selections.keys())
    cached = (
        load_loso_metrics_from_disk(
            out_dir=context.next_video_out_dir,
            feature_spaces=feature_spaces,
            studies=studies,
        )
        if context.reuse_existing
        else None
    )
    if cached:
        LOGGER.info(
            "[LOSO] Reusing cached metrics for feature spaces: %s",
            ",".join(feature_spaces),
        )
        return cached

    # Execute per-study runs while keeping training and evaluation aligned.
    cross_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space, per_study in selections.items():
        LOGGER.info("[LOSO] feature=%s", feature_space)
        per_holdout: Dict[str, Mapping[str, object]] = {}
        for holdout in studies:
            selection = per_study.get(holdout.key)
            if selection is None:
                continue
            loso_root = ensure_dir(
                context.next_video_out_dir / feature_space / "loso" / holdout.study_slug
            )
            issue_slug = issue_slug_for_study(holdout)
            metrics_path = loso_root / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
            if context.reuse_existing and metrics_path.exists():
                try:
                    metrics, _ = load_metrics(loso_root, issue_slug)
                except FileNotFoundError:
                    LOGGER.warning(
                        (
                            "[LOSO][MISS] feature=%s holdout=%s expected cached metrics at %s "
                            "but none found."
                        ),
                        feature_space,
                        holdout.key,
                        metrics_path,
                    )
                else:
                    per_holdout[holdout.key] = metrics
                    LOGGER.info(
                        "[LOSO][SKIP] feature=%s holdout=%s (metrics cached).",
                        feature_space,
                        holdout.key,
                    )
                    continue
            LOGGER.info(
                "[LOSO] feature=%s holdout=%s issue=%s best_k=%d",
                feature_space,
                holdout.key,
                holdout.issue,
                int(selection.best_k),
            )
            run_knn_cli(
                compose_cli_args(
                    context.base_cli,
                    selection.config.cli_args(
                        word2vec_model_dir=_maybe_word2vec_model_dir(
                            feature_space=feature_space,
                            base_dir=context.next_video_word2vec_dir,
                            study_slug=holdout.study_slug,
                        )
                    ),
                    ["--issues", holdout.issue],
                    # Evaluate on the holdout only and train on the same cohort.
                    ["--participant-studies", holdout.key],
                    ["--train-participant-studies", holdout.key],
                    ["--out-dir", str(loso_root)],
                    ["--knn-k", str(selection.best_k)],
                    context.extra_cli,
                )
            )
            try:
                metrics, _ = load_metrics(loso_root, issue_slug)
                per_holdout[holdout.key] = metrics
            except FileNotFoundError:
                LOGGER.warning(
                    "[LOSO][MISS] feature=%s holdout=%s failed to produce metrics at %s",
                    feature_space,
                    holdout.key,
                    metrics_path,
                )
        if per_holdout:
            cross_metrics[feature_space] = per_holdout
    return cross_metrics

__all__ = [
    "run_final_evaluations",
    "run_opinion_evaluations",
    "run_opinion_from_next_evaluations",
    "run_cross_study_evaluations",
    "load_loso_metrics_from_disk",
]
