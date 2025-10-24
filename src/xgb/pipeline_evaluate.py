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

"""Evaluation runners for the Grail Simulation XGBoost pipeline.

Wraps the CLI entry points that execute the selected sweeps, perform
cross-study checks, and run opinion regressions while honouring reuse
settings configured by ``xgb.pipeline``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping

from .opinion import DEFAULT_SPECS, OpinionEvalRequest, OpinionTrainConfig, run_opinion_eval
from .pipeline_context import (
    FinalEvalContext,
    OpinionStageConfig,
    OpinionStudySelection,
    StudySelection,
)
from .pipeline_sweeps import _run_xgb_cli, _load_metrics

LOGGER = logging.getLogger("xgb.pipeline.finalize")


def _run_final_evaluations(
    *,
    selections: Mapping[str, StudySelection],
    context: FinalEvalContext,
) -> Dict[str, Mapping[str, object]]:
    """
    Run the final next-video evaluations for each selected configuration.

    :param selections: Mapping from study key to selected configuration.
    :type selections: Mapping[str, StudySelection]
    :param context: Runtime configuration describing CLI arguments and output paths.
    :type context: FinalEvalContext
    :returns: Mapping from study key to the loaded metrics payload.
    :rtype: Dict[str, Mapping[str, object]]
    """

    metrics_by_study: Dict[str, Mapping[str, object]] = {}
    context.out_dir.mkdir(parents=True, exist_ok=True)

    for study_key, selection in selections.items():
        metrics_path = context.out_dir / selection.evaluation_slug / "metrics.json"
        if context.reuse_existing and metrics_path.exists():
            try:
                metrics = dict(_load_metrics(metrics_path))
            except FileNotFoundError:
                LOGGER.warning(
                    "[FINAL][MISS] issue=%s study=%s expected cached metrics at %s but none found.",
                    selection.study.issue,
                    selection.study.key,
                    metrics_path,
                )
            else:
                metrics.setdefault("issue", selection.study.issue)
                metrics.setdefault("issue_label", selection.study.issue.replace("_", " ").title())
                metrics.setdefault("study", selection.study.key)
                metrics.setdefault("study_label", selection.study.label)
                metrics_by_study[study_key] = metrics
                LOGGER.info(
                    "[FINAL][SKIP] issue=%s study=%s (metrics cached).",
                    selection.study.issue,
                    selection.study.key,
                )
                continue
        cli_args: List[str] = []
        cli_args.extend(context.base_cli)
        cli_args.extend(selection.config.cli_args(context.tree_method))
        cli_args.extend(["--issues", selection.study.issue])
        cli_args.extend(["--participant_studies", selection.study.key])
        cli_args.extend(["--out_dir", str(context.out_dir)])
        if context.save_model_dir is not None:
            cli_args.extend(["--save_model", str(context.save_model_dir)])
        cli_args.extend(context.extra_cli)
        LOGGER.info(
            "[FINAL] issue=%s study=%s config=%s",
            selection.study.issue,
            selection.study.key,
            selection.config.label(),
        )
        _run_xgb_cli(cli_args)
        metrics = dict(_load_metrics(metrics_path))
        metrics.setdefault("issue", selection.study.issue)
        metrics.setdefault("issue_label", selection.study.issue.replace("_", " ").title())
        metrics.setdefault("study", selection.study.key)
        metrics.setdefault("study_label", selection.study.label)
        metrics_by_study[study_key] = metrics
    return metrics_by_study


def _run_opinion_stage(
    *,
    selections: Mapping[str, OpinionStudySelection],
    config: OpinionStageConfig,
) -> Dict[str, Dict[str, object]]:
    """
    Execute the optional opinion regression stage for selected participant studies.

    :param selections: Mapping from study key to selected opinion configuration.
    :type selections: Mapping[str, OpinionStudySelection]
    :param config: Opinion stage configuration describing dataset and feature options.
    :type config: OpinionStageConfig
    :returns: Mapping from study key to the resulting metrics payload.
    :rtype: Dict[str, Dict[str, object]]
    """

    if not selections:
        LOGGER.warning("Skipping opinion stage; no selections available.")
        return {}

    opinion_out_dir = config.base_out_dir / "opinion"
    requested = [token for token in config.studies if token and token.lower() != "all"]
    if not requested:
        requested = [spec.key for spec in DEFAULT_SPECS]

    results: Dict[str, Dict[str, object]] = {}
    for study_key in requested:
        selection = selections.get(study_key)
        if selection is None:
            LOGGER.warning(
                "Skipping opinion study for study=%s (no selection available).",
                study_key,
            )
            continue
        feature_dir = opinion_out_dir / "tfidf"
        study_dir = feature_dir / study_key
        metrics_path = study_dir / f"opinion_xgb_{study_key}_validation_metrics.json"
        if config.reuse_existing and metrics_path.exists():
            try:
                payload = dict(_load_metrics(metrics_path))
            except FileNotFoundError:
                LOGGER.warning(
                    "Opinion metrics expected at %s but missing; rerunning evaluation.",
                    metrics_path,
                )
            else:
                results[study_key] = payload
                LOGGER.info(
                    "[OPINION][SKIP] study=%s issue=%s (metrics cached).",
                    study_key,
                    selection.study.issue,
                )
                continue
        opinion_config = OpinionTrainConfig(
            max_participants=config.max_participants,
            seed=config.seed,
            max_features=config.max_features,
            booster=selection.config.booster_params(config.tree_method),
        )
        payload = run_opinion_eval(
            request=OpinionEvalRequest(
                dataset=config.dataset,
                cache_dir=config.cache_dir,
                out_dir=opinion_out_dir,
                feature_space="tfidf",
                extra_fields=config.extra_fields,
                train_config=opinion_config,
                overwrite=config.overwrite,
            ),
            studies=[study_key],
        )
        results.update(payload)
    return results

__all__ = ['_run_final_evaluations','_run_opinion_stage']
