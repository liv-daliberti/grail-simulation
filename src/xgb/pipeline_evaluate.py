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
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set

from .opinion import DEFAULT_SPECS, OpinionEvalRequest, OpinionTrainConfig, run_opinion_eval
from .pipeline_context import (
    FinalEvalContext,
    OpinionStageConfig,
    OpinionStudySelection,
    StudySelection,
    StudySpec,
)
from .pipeline_sweeps import (
    DEFAULT_OPINION_FEATURE_SPACE,
    _inject_study_metadata,
    _load_loso_metrics_from_disk,
    _load_metrics,
    _load_metrics_with_log,
    _load_opinion_from_next_metrics_from_disk,
    _run_xgb_cli,
)

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
                _inject_study_metadata(metrics, selection.study)
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
        _inject_study_metadata(metrics, selection.study)
        metrics_by_study[study_key] = metrics
    return metrics_by_study


def _run_cross_study_evaluations(
    *,
    selections: Mapping[str, StudySelection],
    studies: Sequence[StudySpec],
    context: FinalEvalContext,
) -> Dict[str, Mapping[str, object]]:
    """
    Execute leave-one-study-out evaluations for the selected configurations.

    :param selections: Mapping from study key to chosen sweep outcome.
    :type selections: Mapping[str, StudySelection]
    :param studies: Ordered list of studies used to determine hold-out rotations.
    :type studies: Sequence[StudySpec]
    :param context: Runtime configuration describing CLI arguments and output paths.
    :type context: FinalEvalContext
    :returns: Mapping from hold-out study key to metrics payload.
    :rtype: Dict[str, Mapping[str, object]]
    """

    if not selections:
        return {}

    loso_root = context.out_dir / "loso"
    loso_root.mkdir(parents=True, exist_ok=True)

    cached: Dict[str, Mapping[str, object]] = {}
    if context.reuse_existing:
        cached = _load_loso_metrics_from_disk(
            next_video_dir=context.out_dir,
            studies=studies,
        )
    results: Dict[str, Mapping[str, object]] = dict(cached)

    for spec in studies:
        selection = selections.get(spec.key)
        if selection is None:
            continue
        if context.reuse_existing and spec.key in cached:
            LOGGER.info(
                "[LOSO][SKIP] issue=%s study=%s (metrics cached).",
                spec.issue,
                spec.key,
            )
            continue
        train_studies = [other.key for other in studies if other.key != spec.key]
        if not train_studies:
            LOGGER.warning(
                "[LOSO] Skipping issue=%s study=%s (no alternate training studies).",
                spec.issue,
                spec.key,
            )
            continue
        cli_args: List[str] = []
        cli_args.extend(context.base_cli)
        cli_args.extend(selection.config.cli_args(context.tree_method))
        cli_args.extend(["--issues", spec.issue])
        cli_args.extend(["--train_participant_studies", ",".join(train_studies)])
        cli_args.extend(["--participant_studies", spec.key])
        cli_args.extend(["--out_dir", str(loso_root)])
        cli_args.extend(context.extra_cli)
        LOGGER.info(
            "[LOSO] issue=%s holdout=%s train_studies=%s",
            spec.issue,
            spec.key,
            ",".join(train_studies),
        )
        _run_xgb_cli(cli_args)
        metrics_path = loso_root / spec.evaluation_slug / "metrics.json"
        metrics = _load_metrics_with_log(
            metrics_path,
            spec,
            log_level=logging.WARNING,
            message="[LOSO] issue=%s study=%s missing metrics at %s.",
        )
        if metrics is None:
            continue
        _inject_study_metadata(metrics, spec)
        results[spec.key] = metrics
    return results


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

    opinion_out_dir = config.base_out_dir
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
        feature_space = selection.outcome.config.text_vectorizer.lower()
        feature_dir = opinion_out_dir / feature_space
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
                    "[OPINION][SKIP] study=%s issue=%s feature=%s (metrics cached).",
                    study_key,
                    selection.study.issue,
                    feature_space,
                )
                continue
        opinion_config = OpinionTrainConfig(
            max_participants=config.max_participants,
            seed=config.seed,
            max_features=config.max_features if feature_space == "tfidf" else None,
            booster=selection.config.booster_params(config.tree_method),
        )
        tfidf_config = config.tfidf_config if feature_space == "tfidf" else None
        word2vec_config = None
        if feature_space == "word2vec":
            base_cfg = config.word2vec_config
            if config.word2vec_model_base is not None:
                model_dir = (
                    config.word2vec_model_base
                    / "opinion_stage"
                    / feature_space
                    / selection.study.issue_slug
                    / study_key
                )
            else:
                model_dir = study_dir / "word2vec_model"
            word2vec_config = replace(
                base_cfg,
                model_dir=str(model_dir),
                seed=config.seed,
            )
        sentence_config = (
            config.sentence_transformer_config
            if feature_space == "sentence_transformer"
            else None
        )
        payload = run_opinion_eval(
            request=OpinionEvalRequest(
                dataset=str(config.dataset) if config.dataset else None,
                cache_dir=str(config.cache_dir) if config.cache_dir else None,
                out_dir=opinion_out_dir,
                feature_space=feature_space,
                extra_fields=config.extra_fields,
                train_config=opinion_config,
                tfidf_config=tfidf_config,
                word2vec_config=word2vec_config,
                sentence_transformer_config=sentence_config,
                overwrite=config.overwrite,
            ),
            studies=[study_key],
        )
        results.update(payload)
    return results


def _requested_opinion_specs(
    studies: Sequence[StudySpec],
    config: OpinionStageConfig,
) -> List[StudySpec]:
    """Resolve the ordered studies that should run opinion-from-next."""

    tokens = [token for token in config.studies if token and token.lower() != "all"]
    if not tokens:
        return list(studies)

    study_by_key = {spec.key: spec for spec in studies}
    selected: List[StudySpec] = []
    for token in tokens:
        spec = study_by_key.get(token)
        if spec is None:
            LOGGER.warning("Skipping opinion-from-next for study=%s (unknown study).", token)
            continue
        selected.append(spec)
    return selected


def _cached_opinion_from_next_metrics(
    config: OpinionStageConfig,
    studies: Sequence[StudySpec],
) -> Dict[str, Dict[str, object]]:
    """Load cached opinion-from-next metrics when reuse is enabled."""

    if not config.reuse_existing:
        return {}
    return _load_opinion_from_next_metrics_from_disk(
        opinion_dir=config.base_out_dir,
        studies=studies,
    )


def _evaluate_opinion_from_next_spec(
    *,
    spec: StudySpec,
    selections: Mapping[str, StudySelection],
    config: OpinionStageConfig,
    cached_keys: Set[str],
    allow_incomplete: bool,
) -> Dict[str, Dict[str, object]] | None:
    """Run the opinion regression for a specific study selection."""

    selection = selections.get(spec.key)
    if selection is None:
        LOGGER.warning(
            "Skipping opinion-from-next for study=%s (no next-video selection).",
            spec.key,
        )
        return None
    feature_space = selection.config.text_vectorizer.lower()
    if config.reuse_existing and spec.key in cached_keys:
        LOGGER.info(
            "[OPINION][FROM-NEXT][SKIP] study=%s feature=%s (metrics cached).",
            spec.key,
            feature_space,
        )
        return None

    base_out_dir = config.base_out_dir / "from_next"
    opinion_config = OpinionTrainConfig(
        max_participants=config.max_participants,
        seed=config.seed,
        max_features=config.max_features if feature_space == "tfidf" else None,
        booster=selection.config.booster_params(config.tree_method),
    )
    tfidf_config = config.tfidf_config if feature_space == "tfidf" else None
    word2vec_config = None
    if feature_space == "word2vec":
        base_cfg = config.word2vec_config
        if config.word2vec_model_base is not None:
            model_dir = (
                config.word2vec_model_base
                / "opinion_from_next"
                / feature_space
                / spec.issue_slug
                / spec.key
            )
        else:
            model_dir = base_out_dir / feature_space / spec.key / "word2vec_model"
        word2vec_config = replace(
            base_cfg,
            model_dir=str(model_dir),
            seed=config.seed,
        )
    sentence_config = (
        config.sentence_transformer_config
        if feature_space == "sentence_transformer"
        else None
    )
    try:
        return run_opinion_eval(
            request=OpinionEvalRequest(
                dataset=str(config.dataset) if config.dataset else None,
                cache_dir=str(config.cache_dir) if config.cache_dir else None,
                out_dir=base_out_dir,
                feature_space=feature_space,
                extra_fields=config.extra_fields,
                train_config=opinion_config,
                tfidf_config=tfidf_config,
                word2vec_config=word2vec_config,
                sentence_transformer_config=sentence_config,
                overwrite=config.overwrite,
            ),
            studies=[spec.key],
        )
    except FileNotFoundError as exc:
        if allow_incomplete:
            LOGGER.warning(
                "[OPINION][FROM-NEXT][SKIP] study=%s dataset missing (%s). "
                "Continuing because allow-incomplete mode is enabled.",
                spec.key,
                exc,
            )
            return None
        raise
    except ImportError as exc:
        if allow_incomplete:
            LOGGER.warning(
                "[OPINION][FROM-NEXT][SKIP] study=%s missing dependency (%s). "
                "Continuing because allow-incomplete mode is enabled.",
                spec.key,
                exc,
            )
            return None
        raise


def _run_opinion_from_next_stage(
    *,
    selections: Mapping[str, StudySelection],
    studies: Sequence[StudySpec],
    config: OpinionStageConfig,
    allow_incomplete: bool = False,
) -> Dict[str, Dict[str, object]]:
    """
    Execute the opinion-regression stage using next-video sweep selections.

    :param selections: Mapping from study key to next-video selection.
    :type selections: Mapping[str, StudySelection]
    :param studies: Ordered list of study specifications considered by the pipeline.
    :type studies: Sequence[StudySpec]
    :param config: Opinion stage configuration describing dataset and feature options.
    :type config: OpinionStageConfig
    :param allow_incomplete: When true, skips missing datasets instead of raising.
    :type allow_incomplete: bool
    :returns: Mapping from study key to metrics payload generated by the stage.
    :rtype: Dict[str, Dict[str, object]]
    """

    if not selections:
        LOGGER.warning("Skipping opinion-from-next stage; no selections available.")
        return {}

    requested_specs = _requested_opinion_specs(studies, config)
    if not requested_specs:
        return {}

    cached = _cached_opinion_from_next_metrics(config, requested_specs)
    cached_keys: Set[str] = set(cached)
    base_out_dir = config.base_out_dir / "from_next"
    base_out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, object]] = dict(cached)

    for spec in requested_specs:
        payload = _evaluate_opinion_from_next_spec(
            spec=spec,
            selections=selections,
            config=config,
            cached_keys=cached_keys,
            allow_incomplete=allow_incomplete,
        )
        if payload:
            results.update(payload)
    return results


__all__ = [
    "_run_cross_study_evaluations",
    "_run_final_evaluations",
    "_run_opinion_from_next_stage",
    "_run_opinion_stage",
]
