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

"""Opinion sweep orchestration helpers."""

from __future__ import annotations

import logging
from functools import partial
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from common.opinion.sweep_types import AccuracySummary, MetricsArtifact
from common.pipeline.executor import execute_indexed_tasks

from ...core.opinion import (
    OpinionEvalRequest,
    OpinionTrainConfig,
    OpinionVectorizerConfig,
    run_opinion_eval,
)
from ..context import (
    OpinionStudySelection,
    OpinionSweepOutcome,
    OpinionSweepRunContext,
    OpinionSweepTask,
    StudySpec,
    SweepConfig,
)
from .common import (
    LOGGER,
    ensure_metrics_with_placeholder,
    get_sweeps_attr,
    merge_sweep_outcomes,
)


def _opinion_sweep_outcome_from_metrics(
    task: OpinionSweepTask,
    metrics: Mapping[str, object],
    metrics_path: Path,
) -> OpinionSweepOutcome:
    """
    Convert cached opinion sweep metrics into an outcome instance.

    :param task: Opinion sweep task metadata associated with the metrics.
    :type task: OpinionSweepTask
    :param metrics: Metrics dictionary loaded from disk.
    :type metrics: Mapping[str, object]
    :param metrics_path: Filesystem path to the metrics artefact.
    :type metrics_path: Path
    :returns: Opinion sweep outcome capturing MAE, RMSE, and R².
    :rtype: ~xgb.pipeline.context.OpinionSweepOutcome
    """

    def _safe_float(value: object, default: float = 0.0) -> float:
        """
        Convert ``value`` to ``float`` with a default fallback.

        :param value: Input value to coerce.
        :type value: object
        :param default: Fallback returned when coercion fails.
        :type default: float
        :returns: Floating-point value or ``default`` when conversion fails.
        :rtype: float
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    metrics_block = metrics.get("metrics", {})
    baseline_block = metrics.get("baseline", {})

    def _optional_float(value: object) -> Optional[float]:
        """
        Convert numeric-like values into ``float`` instances when possible.

        :param value: Candidate value extracted from the metrics payload.
        :type value: object
        :returns: Floating-point value or ``None`` when conversion is unsupported.
        :rtype: Optional[float]
        """
        if isinstance(value, (int, float)):
            return float(value)
        return None

    accuracy_value = _optional_float(metrics_block.get("direction_accuracy"))
    baseline_value = _optional_float(baseline_block.get("direction_accuracy"))
    accuracy_delta = None
    if accuracy_value is not None and baseline_value is not None:
        accuracy_delta = accuracy_value - baseline_value

    def _optional_int(value: object) -> Optional[int]:
        """
        Convert numeric-like values into ``int`` instances when possible.

        :param value: Candidate value extracted from the metrics payload.
        :type value: object
        :returns: Integer value or ``None`` when conversion is unsupported.
        :rtype: Optional[int]
        """
        if isinstance(value, (int, float)):
            return int(value)
        return None

    eligible_value = _optional_int(metrics.get("eligible"))
    if eligible_value is None:
        eligible_value = _optional_int(metrics_block.get("eligible"))
    return OpinionSweepOutcome(
        order_index=task.index,
        study=task.study,
        config=task.config,
        mae=_safe_float(metrics_block.get("mae_after")),
        rmse=_safe_float(metrics_block.get("rmse_after")),
        r_squared=_safe_float(metrics_block.get("r2_after")),
        artifact=MetricsArtifact(path=metrics_path, payload=metrics),
        accuracy_summary=AccuracySummary(
            value=accuracy_value,
            baseline=baseline_value,
            delta=accuracy_delta,
            eligible=eligible_value,
        ),
    )


def _build_opinion_vectorizer_config(
    *,
    config: SweepConfig,
    context: OpinionSweepRunContext,
    run_root: Path,
    study: StudySpec,
    extra_fields: Sequence[str],
) -> OpinionVectorizerConfig:
    """
    Construct vectoriser settings for a single opinion sweep.

    :param config: Sweep configuration that determines the feature space.
    :type config: SweepConfig
    :param context: Shared opinion sweep execution context.
    :type context: OpinionSweepRunContext
    :param run_root: Root directory receiving sweep artefacts for the task.
    :type run_root: Path
    :param study: Study metadata used to derive filesystem slugs.
    :type study: StudySpec
    :param extra_fields: Prompt document fields appended to the vectoriser inputs.
    :type extra_fields: Sequence[str]
    :returns: Serialisable vectoriser configuration consumed by the evaluator.
    :rtype: OpinionVectorizerConfig
    """

    feature_space = config.text_vectorizer.lower()
    vectorizer_args: Dict[str, object] = {
        "feature_space": feature_space,
        "extra_fields": extra_fields,
    }
    if feature_space == "tfidf":
        vectorizer_args["tfidf"] = context.tfidf_config
    elif feature_space == "word2vec":
        base_cfg = context.word2vec_config
        model_dir = (
            context.word2vec_model_base
            / "opinion_sweeps"
            / study.issue_slug
            / study.study_slug
            / config.label()
            if context.word2vec_model_base is not None
            else run_root / feature_space / "word2vec_model"
        )
        vectorizer_args["word2vec"] = replace(
            base_cfg,
            model_dir=str(model_dir),
            seed=context.seed,
        )
    elif feature_space == "sentence_transformer":
        vectorizer_args["sentence_transformer"] = context.sentence_transformer_config
    return OpinionVectorizerConfig(**vectorizer_args)


def _build_opinion_request_args(
    *,
    context: OpinionSweepRunContext,
    config: SweepConfig,
    vectorizer: OpinionVectorizerConfig,
    run_root: Path,
) -> Dict[str, object]:
    """
    Compose keyword arguments for :class:`~open_r1.core.opinion.OpinionEvalRequest`.

    :param run_root: Root directory for sweep artefacts.
    :type run_root: Path
    :param context: Shared opinion sweep execution context.
    :type context: OpinionSweepRunContext
    :param config: Sweep configuration that produced the task.
    :type config: SweepConfig
    :param vectorizer: Vectoriser configuration derived from the sweep config.
    :type vectorizer: OpinionVectorizerConfig
    :returns: Dictionary forwarded to :func:`run_opinion_eval`.
    :rtype: Dict[str, object]
    """

    dataset = str(context.dataset) if context.dataset else None
    cache_dir = str(context.cache_dir) if context.cache_dir else None

    return {
        "dataset": dataset,
        "cache_dir": cache_dir,
        "out_dir": run_root,
        "train_config": OpinionTrainConfig(
            max_participants=context.max_participants,
            seed=context.seed,
            max_features=context.max_features,
            booster=config.booster_params(context.tree_method),
        ),
        "vectorizer": vectorizer,
        "overwrite": True,
    }


def _build_opinion_task(
    *,
    config: SweepConfig,
    study: StudySpec,
    context: OpinionSweepRunContext,
    task_index: int,
) -> OpinionSweepTask:
    """
    Create a fully-populated :class:`OpinionSweepTask` for a single study/config pair.

    :param config: Sweep configuration describing the model variant to run.
    :type config: SweepConfig
    :param study: Study metadata describing participants and labelling.
    :type study: StudySpec
    :param context: Shared opinion sweep execution context.
    :type context: OpinionSweepRunContext
    :param task_index: Deterministic order index for the task.
    :type task_index: int
    :returns: Opinion sweep task ready for execution.
    :rtype: OpinionSweepTask
    """

    vectorizer_builder = cast(
        Callable[..., OpinionVectorizerConfig],
        get_sweeps_attr("_build_opinion_vectorizer_config"),
    )
    extra_fields = tuple(context.extra_fields)
    run_root = context.sweep_dir / study.issue_slug / study.study_slug / config.label()
    vectorizer = vectorizer_builder(
        config=config,
        context=context,
        run_root=run_root,
        study=study,
        extra_fields=extra_fields,
    )
    feature_space = vectorizer.feature_space
    metrics_path = (
        run_root
        / feature_space
        / study.key
        / f"opinion_xgb_{study.key}_validation_metrics.json"
    )
    request_args = _build_opinion_request_args(
        context=context,
        config=config,
        vectorizer=vectorizer,
        run_root=run_root,
    )
    return OpinionSweepTask(
        index=task_index,
        study=study,
        config=config,
        feature_space=feature_space,
        metrics_path=metrics_path,
        request_args=request_args,
    )


def _iter_opinion_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: OpinionSweepRunContext,
) -> Sequence[OpinionSweepTask]:
    """
    Yield opinion sweep tasks in a deterministic order.

    :param studies: Participant studies targeted by the opinion sweeps.
    :type studies: Sequence[StudySpec]
    :param configs: Hyper-parameter configurations evaluated for each study.
    :type configs: Sequence[SweepConfig]
    :param context: Shared opinion sweep execution context.
    :type context: OpinionSweepRunContext
    :returns: Iterable sequence of opinion sweep tasks sorted by a stable index.
    :rtype: Sequence[OpinionSweepTask]
    """

    tasks: List[OpinionSweepTask] = []
    task_index = 0
    for config in configs:
        for study in studies:
            tasks.append(
                _build_opinion_task(
                    config=config,
                    study=study,
                    context=context,
                    task_index=task_index,
                )
            )
            task_index += 1
    return tasks


def _prepare_opinion_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: OpinionSweepRunContext,
    reuse_existing: bool,
) -> Tuple[List[OpinionSweepTask], List[OpinionSweepOutcome]]:
    """Return pending opinion sweep tasks and cached outcomes."""

    pending: List[OpinionSweepTask] = []
    cached: List[OpinionSweepOutcome] = []
    load_metrics = cast(
        Callable[[Path], Mapping[str, object]],
        get_sweeps_attr("_load_metrics"),
    )
    outcome_factory = cast(
        Callable[[OpinionSweepTask, Mapping[str, object], Path], OpinionSweepOutcome],
        get_sweeps_attr("_opinion_sweep_outcome_from_metrics"),
    )
    for task in _iter_opinion_sweep_tasks(
        studies=studies,
        configs=configs,
        context=context,
    ):
        metrics_path = task.metrics_path
        if reuse_existing and metrics_path.exists():
            LOGGER.info(
                "[OPINION][SWEEP][SKIP] study=%s issue=%s feature=%s config=%s (cached).",
                task.study.key,
                task.study.issue,
                task.feature_space,
                task.config.label(),
            )
            metrics = load_metrics(metrics_path)
            cached.append(outcome_factory(task, metrics, metrics_path))
            continue
        pending.append(task)
    return pending, cached


_merge_opinion_sweep_outcomes = cast(
    Callable[
        [Sequence[OpinionSweepOutcome], Sequence[OpinionSweepOutcome]],
        List[OpinionSweepOutcome],
    ],
    partial(
        merge_sweep_outcomes,
        duplicate_message=(
            "Duplicate opinion sweep outcome for index=%d; replacing cached result."
        ),
    ),
)
_merge_opinion_sweep_outcomes.__doc__ = """
Merge cached and freshly executed opinion sweep outcomes while preserving order indices.

:param cached: Previously cached opinion sweep outcomes loaded from disk.
:type cached: Sequence[OpinionSweepOutcome]
:param executed: Newly generated opinion sweep outcomes from the current run.
:type executed: Sequence[OpinionSweepOutcome]
:returns: Ordered list containing the merged opinion sweep outcomes.
:rtype: List[OpinionSweepOutcome]
"""


def _execute_opinion_sweep_task(task: OpinionSweepTask) -> OpinionSweepOutcome:
    """Execute a single opinion sweep task and return the resulting metrics."""

    load_metrics_with_log = cast(
        Callable[..., Dict[str, object] | None],
        get_sweeps_attr("_load_metrics_with_log"),
    )
    outcome_factory = cast(
        Callable[[OpinionSweepTask, Mapping[str, object], Path], OpinionSweepOutcome],
        get_sweeps_attr("_opinion_sweep_outcome_from_metrics"),
    )
    args = dict(task.request_args)
    out_dir = Path(args["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    args["out_dir"] = out_dir
    request = OpinionEvalRequest(**args)

    LOGGER.info(
        "[OPINION][SWEEP] study=%s issue=%s feature=%s config=%s",
        task.study.key,
        task.study.issue,
        request.feature_space,
        task.config.label(),
    )
    run_opinion_eval(request=request, studies=[task.study.key])

    # Opinion evaluations may be skipped (e.g., no train/eval rows).
    # Tolerate missing metrics by logging and returning a placeholder outcome.
    metrics = ensure_metrics_with_placeholder(
        lambda: load_metrics_with_log(
            task.metrics_path,
            task.study,
            log_level=logging.WARNING,
            message=(
                "[OPINION][SWEEP][MISS] issue=%s study=%s expected metrics at %s; "
                "recording placeholder outcome."
            ),
        ),
        placeholder_factory=lambda: {
            "model": "xgb_opinion",
            "feature_space": task.feature_space,
            "study": task.study.key,
            "issue": task.study.issue,
            "split": "validation",
            "metrics": {},
            "baseline": {},
            "skipped": True,
        },
        metrics_path=task.metrics_path,
        debug_message="[OPINION][SWEEP][MISS] Unable to write placeholder metrics at %s",
    )
    return outcome_factory(task, metrics, task.metrics_path)


def _execute_opinion_sweep_tasks(
    tasks: Sequence[OpinionSweepTask],
    *,
    jobs: int,
) -> List[OpinionSweepOutcome]:
    """Execute opinion sweep tasks, optionally in parallel."""

    return execute_indexed_tasks(
        tasks,
        _execute_opinion_sweep_task,
        jobs=jobs,
        logger=LOGGER,
        label="opinion sweep",
    )


# Backwards compatibility: allow legacy imports without the leading underscore.
execute_opinion_sweep_tasks = _execute_opinion_sweep_tasks


def _load_opinion_metrics_from_disk(
    *,
    opinion_dir: Path,
    studies: Sequence[StudySpec],
) -> Dict[str, Dict[str, object]]:
    """
    Load opinion regression metrics saved by previous runs.

    :param opinion_dir: Directory containing opinion artefacts.
    :type opinion_dir: Path
    :param studies: Participant studies to load.
    :type studies: Sequence[~common.pipeline.types.StudySpec]
    :returns: Mapping from study key to opinion metrics payload.
    :rtype: Dict[str, Dict[str, object]]
    """

    results: Dict[str, Dict[str, object]] = {}
    load_metrics = cast(
        Callable[[Path], Mapping[str, object]],
        get_sweeps_attr("_load_metrics"),
    )
    filename_template = "opinion_xgb_{study}_validation_metrics.json"
    for spec in studies:
        filename = filename_template.format(study=spec.key)
        candidate_dirs = [
            opinion_dir / space / spec.key
            for space in ("tfidf", "word2vec", "sentence_transformer")
        ] + [
            opinion_dir / spec.key,
            opinion_dir / "opinion" / spec.key,
        ]
        metrics_path: Path | None = None
        for base_dir in candidate_dirs:
            path = base_dir / filename
            if path.exists():
                metrics_path = path
                break
        if metrics_path is None:
            continue
        metrics = dict(load_metrics(metrics_path))
        results[spec.key] = metrics
    return results


def _load_opinion_from_next_metrics_from_disk(
    *,
    opinion_dir: Path,
    studies: Sequence[StudySpec],
) -> Dict[str, Dict[str, object]]:
    """
    Load opinion regression metrics produced using next-video configurations.

    :param opinion_dir: Directory containing opinion artefacts.
    :type opinion_dir: Path
    :param studies: Participant studies to load.
    :type studies: Sequence[~common.pipeline.types.StudySpec]
    :returns: Mapping from study key to opinion-from-next metrics payload.
    :rtype: Dict[str, Dict[str, object]]
    """

    results: Dict[str, Dict[str, object]] = {}
    load_metrics = cast(
        Callable[[Path], Mapping[str, object]],
        get_sweeps_attr("_load_metrics"),
    )
    feature_spaces = ("tfidf", "word2vec", "sentence_transformer")
    base_dirs = [opinion_dir / "from_next" / space for space in feature_spaces]
    base_dirs.append(opinion_dir / "from_next")
    filename_template = "opinion_xgb_{study}_validation_metrics.json"
    for spec in studies:
        filename = filename_template.format(study=spec.key)
        metrics_path: Path | None = None
        for base_dir in base_dirs:
            path = base_dir / spec.key / filename
            if path.exists():
                metrics_path = path
                break
        if metrics_path is None:
            continue
        try:
            metrics = dict(load_metrics(metrics_path))
        except FileNotFoundError:
            LOGGER.debug(
                "[OPINION-NEXT][MISS] study=%s expected metrics at %s but none found.",
                spec.key,
                metrics_path,
            )
            continue
        results[spec.key] = metrics
    return results


def _select_best_opinion_configs(
    outcomes: Sequence[OpinionSweepOutcome],
) -> Dict[str, OpinionStudySelection]:
    """
    Pick the best opinion configuration per study prioritising MAE, RMSE, then R².

    :param outcomes: Opinion sweep outcomes spanning studies and configurations.
    :type outcomes: Sequence[~xgb.pipeline.context.OpinionSweepOutcome]
    :returns: Mapping from study key to the chosen opinion configuration.
    :rtype: Dict[str, ~xgb.pipeline.context.OpinionStudySelection]
    """

    selections: Dict[str, OpinionStudySelection] = {}
    for outcome in outcomes:
        current = selections.get(outcome.study.key)
        if current is None:
            selections[outcome.study.key] = OpinionStudySelection(
                study=outcome.study,
                outcome=outcome,
            )
            continue
        incumbent = current.outcome
        if outcome.mae < incumbent.mae - 1e-9:
            selections[outcome.study.key] = OpinionStudySelection(
                study=outcome.study,
                outcome=outcome,
            )
            continue
        if abs(outcome.mae - incumbent.mae) <= 1e-9:
            if outcome.rmse < incumbent.rmse - 1e-9:
                selections[outcome.study.key] = OpinionStudySelection(
                    study=outcome.study,
                    outcome=outcome,
                )
            elif (
                abs(outcome.rmse - incumbent.rmse) <= 1e-9
                and outcome.r_squared > incumbent.r_squared + 1e-9
            ):
                selections[outcome.study.key] = OpinionStudySelection(
                    study=outcome.study,
                    outcome=outcome,
                )
    return selections


__all__ = [
    "execute_opinion_sweep_tasks",
    "_build_opinion_vectorizer_config",
    "_execute_opinion_sweep_task",
    "_execute_opinion_sweep_tasks",
    "_iter_opinion_sweep_tasks",
    "_load_opinion_from_next_metrics_from_disk",
    "_load_opinion_metrics_from_disk",
    "_merge_opinion_sweep_outcomes",
    "_opinion_sweep_outcome_from_metrics",
    "_prepare_opinion_sweep_tasks",
    "_select_best_opinion_configs",
]
