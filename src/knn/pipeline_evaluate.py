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
from typing import Dict, Mapping, Sequence

from .pipeline_context import EvaluationContext, OpinionStudySelection, StudySelection, StudySpec
from .pipeline_data import issue_slug_for_study
from .pipeline_io import load_loso_metrics_from_disk, load_metrics, load_opinion_metrics
from .pipeline_sweeps import run_knn_cli
from .pipeline_utils import ensure_dir

LOGGER = logging.getLogger("knn.pipeline.evaluate")

# pylint: disable=too-many-locals
def run_final_evaluations(
    *,
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
    context: EvaluationContext,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """
    Run final slate evaluations and return metrics grouped by feature space.

    :param selections: Winning sweep selections keyed by feature space and study key.
    :type selections: Mapping[str, Mapping[str, StudySelection]]
    :param studies: Ordered list of studies to evaluate.
    :type studies: Sequence[StudySpec]
    :param context: Shared CLI/runtime parameters used for all evaluations.
    :type context: EvaluationContext
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
            model_dir = None
            if feature_space == "word2vec":
                model_dir = ensure_dir(context.next_video_word2vec_dir / study.study_slug)
            issue_slug = issue_slug_for_study(study)
            metrics_path = (
                feature_out_dir / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
            )
            if context.reuse_existing and metrics_path.exists():
                try:
                    metrics, _ = load_metrics(feature_out_dir, issue_slug)
                except FileNotFoundError:
                    LOGGER.warning(
                        "[FINAL][MISS] feature=%s study=%s expected cached metrics at %s "
                        "but none found.",
                        feature_space,
                        study.key,
                        metrics_path,
                    )
                else:
                    feature_metrics[study.key] = metrics
                    LOGGER.info(
                        "[FINAL][SKIP] feature=%s study=%s (metrics cached).",
                        feature_space,
                        study.key,
                    )
                    continue
            cli_args: list[str] = []
            cli_args.extend(context.base_cli)
            cli_args.extend(selection.config.cli_args(word2vec_model_dir=model_dir))
            cli_args.extend(["--issues", study.issue])
            cli_args.extend(["--participant-studies", study.key])
            cli_args.extend(["--out-dir", str(feature_out_dir)])
            cli_args.extend(["--knn-k", str(selection.best_k)])
            cli_args.extend(context.extra_cli)
            run_knn_cli(cli_args)
            metrics, _ = load_metrics(feature_out_dir, issue_slug)
            feature_metrics[study.key] = metrics
        if feature_metrics:
            metrics_by_feature[feature_space] = feature_metrics
    return metrics_by_feature

# pylint: disable=too-many-locals
def run_opinion_evaluations(
    *,
    selections: Mapping[str, Mapping[str, OpinionStudySelection]],
    studies: Sequence[StudySpec],
    context: EvaluationContext,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """
    Run opinion regression for each feature space and return metrics.

    :param selections: Winning opinion selections keyed by feature space and study key.
    :type selections: Mapping[str, Mapping[str, OpinionStudySelection]]
    :param studies: Ordered list of opinion studies to evaluate.
    :type studies: Sequence[StudySpec]
    :param context: Shared CLI/runtime parameters used for all evaluations.
    :type context: EvaluationContext
    :returns: Nested mapping ``feature_space -> study_key -> metrics`` for the opinion evaluations.
    :rtype: Dict[str, Dict[str, Mapping[str, object]]]
    """
    metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space, per_study in selections.items():
        LOGGER.info("[OPINION] feature=%s", feature_space)
        feature_out_dir = ensure_dir(context.opinion_out_dir)
        cached_metrics = (
            load_opinion_metrics(feature_out_dir, feature_space)
            if context.reuse_existing
            else {}
        )
        for study in studies:
            selection = per_study.get(study.key)
            if selection is None:
                continue
            if context.reuse_existing and study.key in cached_metrics:
                LOGGER.info(
                    "[OPINION][SKIP] feature=%s study=%s (metrics cached).",
                    feature_space,
                    study.key,
                )
                continue
            LOGGER.info("[OPINION] study=%s issue=%s", study.key, study.issue)
            model_dir = None
            if feature_space == "word2vec":
                model_dir = ensure_dir(context.opinion_word2vec_dir / study.study_slug)
            cli_args: list[str] = []
            cli_args.extend(context.base_cli)
            cli_args.extend(selection.config.cli_args(word2vec_model_dir=model_dir))
            cli_args.extend(["--task", "opinion"])
            cli_args.extend(["--out-dir", str(feature_out_dir)])
            cli_args.extend(["--knn-k", str(selection.best_k)])
            cli_args.extend(["--opinion-studies", study.key])
            cli_args.extend(context.extra_cli)
            run_knn_cli(cli_args)
        metrics[feature_space] = load_opinion_metrics(feature_out_dir, feature_space)
    return metrics

# pylint: disable=too-many-locals
def run_cross_study_evaluations(
    *,
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
    context: EvaluationContext,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """
    Run leave-one-study-out evaluations and return metrics grouped by feature space.

    :param selections: Winning sweep selections keyed by feature space and study key.
    :type selections: Mapping[str, Mapping[str, StudySelection]]
    :param studies: Ordered list of studies to iterate when selecting hold-outs.
    :type studies: Sequence[StudySpec]
    :param context: Shared CLI/runtime parameters used for all evaluations.
    :type context: EvaluationContext
    :returns: Nested mapping ``feature_space -> holdout_study -> metrics`` for the LOSO evaluations.
    :rtype: Dict[str, Dict[str, Mapping[str, object]]]
    """
    cross_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    cached_cross = (
        load_loso_metrics_from_disk(
            out_dir=context.next_video_out_dir,
            feature_spaces=tuple(selections.keys()),
            studies=studies,
        )
        if context.reuse_existing
        else {}
    )
    for feature_space, per_study in selections.items():
        feature_metrics: Dict[str, Mapping[str, object]] = dict(cached_cross.get(feature_space, {}))
        feature_out_dir = ensure_dir(context.next_video_out_dir / feature_space / "loso")
        for study in studies:
            selection = per_study.get(study.key)
            if selection is None:
                continue
            if context.reuse_existing and study.key in feature_metrics:
                LOGGER.info(
                    "[LOSO][SKIP] feature=%s holdout=%s (metrics cached).",
                    feature_space,
                    study.key,
                )
                continue
            train_studies = [spec.key for spec in studies if spec.key != study.key]
            if not train_studies:
                LOGGER.warning(
                    "[LOSO] Skipping feature=%s holdout=%s (no alternate studies).",
                    feature_space,
                    study.key,
                )
                continue

            model_dir = None
            if feature_space == "word2vec":
                model_dir = ensure_dir(
                    context.next_video_word2vec_dir / "loso" / study.study_slug
                )

            holdout_out_dir = ensure_dir(feature_out_dir / study.study_slug)
            cli_args: list[str] = []
            cli_args.extend(context.base_cli)
            cli_args.extend(selection.config.cli_args(word2vec_model_dir=model_dir))
            cli_args.extend(["--out-dir", str(holdout_out_dir)])
            cli_args.extend(["--knn-k", str(selection.best_k)])
            cli_args.extend(["--train-participant-studies", ",".join(train_studies)])
            cli_args.extend(["--eval-participant-studies", study.key])
            cli_args.extend(["--train-issues", "all"])
            cli_args.extend(["--eval-issues", study.issue])
            cli_args.extend(context.extra_cli)

            LOGGER.info(
                "[LOSO] feature=%s holdout=%s train_studies=%s",
                feature_space,
                study.key,
                ",".join(train_studies),
            )
            run_knn_cli(cli_args)

            issue_slug = issue_slug_for_study(study)
            try:
                metrics, metrics_path = load_metrics(holdout_out_dir, issue_slug)
            except FileNotFoundError:
                LOGGER.warning(
                    "[LOSO] Missing metrics for feature=%s holdout=%s (expected slug=%s)",
                    feature_space,
                    study.key,
                    issue_slug,
                )
                continue
            feature_metrics[study.key] = metrics
            LOGGER.info(
                "[LOSO] feature=%s holdout=%s metrics=%s",
                feature_space,
                study.key,
                metrics_path,
            )
        if feature_metrics:
            cross_metrics[feature_space] = feature_metrics
    return cross_metrics

__all__ = [
    "run_cross_study_evaluations",
    "run_final_evaluations",
    "run_opinion_evaluations",
]
