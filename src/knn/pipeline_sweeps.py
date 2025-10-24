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

# pylint: disable=line-too-long
from __future__ import annotations

import json
import logging
import os
from itertools import product
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from common.pipeline_executor import execute_indexed_tasks, execute_sequential_tasks
from common.pipeline_utils import merge_ordered

from knn.cli import build_parser as build_knn_parser
from knn.evaluate import run_eval

from .opinion import run_opinion_eval
from .pipeline_context import (
    OpinionStudySelection,
    OpinionSweepOutcome,
    OpinionSweepTask,
    PipelineContext,
    StudySelection,
    StudySpec,
    SweepConfig,
    SweepOutcome,
    SweepTask,
    SweepTaskContext,
)
from .pipeline_data import issue_slug_for_study
from .pipeline_io import load_metrics
from .pipeline_utils import ensure_dir, extract_metric_summary, extract_opinion_summary

LOGGER = logging.getLogger("knn.pipeline.sweeps")

def run_knn_cli(argv: Sequence[str]) -> None:
    """
    Execute the KNN CLI entry point with ``argv``.

    :param argv: Optional argument vector override used when invoking the CLI programmatically.

    :type argv: Sequence[str]

    :returns: None.

    :rtype: None

    """
    parser = build_knn_parser()
    namespace = parser.parse_args(list(argv))
    if getattr(namespace, "task", "slate") == "slate":
        run_eval(namespace)
        return
    if namespace.task == "opinion":
        run_opinion_eval(namespace)
        return
    raise ValueError(f"Unsupported task '{namespace.task}'.")

def build_sweep_configs(context: PipelineContext) -> List[SweepConfig]:
    """
    Return the grid of configurations evaluated during sweeps.

    :param context: Pipeline context encapsulating dataset paths and configuration flags.

    :type context: PipelineContext

    :returns: the grid of configurations evaluated during sweeps

    :rtype: List[SweepConfig]

    """
    # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks
    text_options: Tuple[Tuple[str, ...], ...] = ((), ("viewer_profile", "state_text"))
    feature_spaces = context.feature_spaces

    configs: List[SweepConfig] = []

    if "tfidf" in feature_spaces:
        for metric in ("cosine", "l2"):
            for fields in text_options:
                configs.append(
                    SweepConfig(
                        feature_space="tfidf",
                        metric=metric,
                        text_fields=fields,
                    )
                )

    if "word2vec" in feature_spaces:
        word2vec_metrics = ("cosine", "l2")
        word2vec_sizes = tuple(
            int(token)
            for token in os.environ.get("WORD2VEC_SWEEP_SIZES", "128,256").split(",")
            if token.strip()
        )
        word2vec_windows = tuple(
            int(token)
            for token in os.environ.get("WORD2VEC_SWEEP_WINDOWS", "5,10").split(",")
            if token.strip()
        )
        word2vec_min_counts = tuple(
            int(token)
            for token in os.environ.get("WORD2VEC_SWEEP_MIN_COUNTS", "1").split(",")
            if token.strip()
        )
        word2vec_epochs_options = tuple(
            int(token)
            for token in os.environ.get(
                "WORD2VEC_SWEEP_EPOCHS", str(context.word2vec_epochs)
            ).split(",")
            if token.strip()
        )
        word2vec_workers_options = tuple(
            int(token)
            for token in os.environ.get(
                "WORD2VEC_SWEEP_WORKERS", str(context.word2vec_workers)
            ).split(",")
            if token.strip()
        )
        for metric in word2vec_metrics:
            for fields in text_options:
                for size in word2vec_sizes:
                    for window in word2vec_windows:
                        for min_count in word2vec_min_counts:
                            for epochs in word2vec_epochs_options:
                                for workers in word2vec_workers_options:
                                    configs.append(
                                        SweepConfig(
                                            feature_space="word2vec",
                                            metric=metric,
                                            text_fields=fields,
                                            word2vec_size=size,
                                            word2vec_window=window,
                                            word2vec_min_count=min_count,
                                            word2vec_epochs=epochs,
                                            word2vec_workers=workers,
                                        )
                                    )

    if "sentence_transformer" in feature_spaces:
        for metric in ("cosine", "l2"):
            for fields in text_options:
                configs.append(
                    SweepConfig(
                        feature_space="sentence_transformer",
                        metric=metric,
                        text_fields=fields,
                        sentence_transformer_model=context.sentence_model,
                        sentence_transformer_device=context.sentence_device,
                        sentence_transformer_batch_size=context.sentence_batch_size,
                        sentence_transformer_normalize=context.sentence_normalize,
                    )
                )

    return configs

def prepare_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: SweepTaskContext,
    reuse_existing: bool,
) -> Tuple[List[SweepTask], List[SweepOutcome]]:
    """
    Return next-video sweep tasks requiring execution and cached outcomes.

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :param configs: Iterable of sweep configurations scheduled for execution.

    :type configs: Sequence[SweepConfig]

    :param context: Shared CLI/runtime parameters reused across sweep invocations.
    :type context: SweepTaskContext

    :param reuse_existing: Whether to reuse cached results instead of recomputing them.

    :type reuse_existing: bool

    :returns: next-video sweep tasks requiring execution and cached outcomes

    :rtype: Tuple[List[SweepTask], List[SweepOutcome]]

    """
    # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks
    pending_tasks: List[SweepTask] = []
    cached_outcomes: List[SweepOutcome] = []
    base_cli_tuple = tuple(context.base_cli)
    extra_cli_tuple = tuple(context.extra_cli)

    task_index = 0
    for config in configs:
        for study in studies:
            issue_slug = issue_slug_for_study(study)
            run_root = context.sweep_dir / config.feature_space / study.study_slug / config.label()
            metrics_path = run_root / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
            word2vec_model_dir = None
            if config.feature_space == "word2vec":
                word2vec_model_dir = (
                    context.word2vec_model_base / "sweeps" / study.study_slug / config.label()
                )
            task = SweepTask(
                index=task_index,
                study=study,
                config=config,
                base_cli=base_cli_tuple,
                extra_cli=extra_cli_tuple,
                run_root=run_root,
                word2vec_model_dir=word2vec_model_dir,
                issue=study.issue,
                issue_slug=issue_slug,
                metrics_path=metrics_path,
            )
            task_index += 1
            if reuse_existing and metrics_path.exists():
                try:
                    metrics, cached_path = load_metrics(run_root, issue_slug)
                except FileNotFoundError:
                    LOGGER.debug("Expected cached metrics at %s but none found.", metrics_path)
                else:
                    LOGGER.info(
                        "[SWEEP][SKIP] feature=%s study=%s label=%s (metrics cached).",
                        config.feature_space,
                        study.key,
                        config.label(),
                    )
                    cached_outcomes.append(sweep_outcome_from_metrics(task, metrics, cached_path))
                    continue
            pending_tasks.append(task)
    return pending_tasks, cached_outcomes


def _build_opinion_task(
    *,
    index: int,
    config: SweepConfig,
    study: StudySpec,
    context: SweepTaskContext,
) -> Tuple[OpinionSweepTask, Path]:
    run_root = (
        context.sweep_dir
        / "opinion"
        / config.feature_space
        / study.study_slug
        / config.label()
    )
    metrics_path = (
        run_root
        / "opinion"
        / config.feature_space
        / study.key
        / f"opinion_knn_{study.key}_validation_metrics.json"
    )
    word2vec_model_dir = None
    if config.feature_space == "word2vec":
        word2vec_model_dir = (
            context.word2vec_model_base / "sweeps_opinion" / study.study_slug / config.label()
        )
    task = OpinionSweepTask(
        index=index,
        study=study,
        config=config,
        base_cli=tuple(context.base_cli),
        extra_cli=tuple(context.extra_cli),
        run_root=run_root,
        word2vec_model_dir=word2vec_model_dir,
        metrics_path=metrics_path,
    )
    return task, metrics_path


def prepare_opinion_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: SweepTaskContext,
    reuse_existing: bool,
) -> Tuple[List[OpinionSweepTask], List[OpinionSweepOutcome]]:
    """
    Return opinion sweep tasks requiring execution and cached outcomes.

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :param configs: Iterable of sweep configurations scheduled for execution.

    :type configs: Sequence[SweepConfig]

    :param context: Shared CLI/runtime parameters reused across sweep invocations.
    :type context: SweepTaskContext

    :param reuse_existing: Whether to reuse cached results instead of recomputing them.

    :type reuse_existing: bool

    :returns: opinion sweep tasks requiring execution and cached outcomes

    :rtype: Tuple[List[OpinionSweepTask], List[OpinionSweepOutcome]]

    """
    # pylint: disable=too-many-branches
    pending_tasks: List[OpinionSweepTask] = []
    cached_outcomes: List[OpinionSweepOutcome] = []

    for task_index, (config, study) in enumerate(product(configs, studies)):
        task, metrics_path = _build_opinion_task(
            index=task_index,
            config=config,
            study=study,
            context=context,
        )
        if reuse_existing and metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
            except FileNotFoundError:
                LOGGER.debug("Expected cached opinion metrics at %s but none found.", metrics_path)
            else:
                LOGGER.info(
                    "[OPINION][SKIP] feature=%s study=%s label=%s (metrics cached).",
                    config.feature_space,
                    study.key,
                    config.label(),
                )
                cached_outcomes.append(
                    opinion_sweep_outcome_from_metrics(task, metrics, metrics_path)
                )
                continue
        pending_tasks.append(task)
    return pending_tasks, cached_outcomes

def sweep_outcome_from_metrics(
    task: SweepTask,
    metrics: Mapping[str, object],
    metrics_path: Path,
) -> SweepOutcome:
    """
    Translate cached metrics into a :class:`SweepOutcome`.

    :param task: Individual sweep task describing an execution unit.

    :type task: SweepTask

    :param metrics: Metrics dictionary captured from a previous pipeline stage.

    :type metrics: Mapping[str, object]

    :param metrics_path: Filesystem path where the metrics JSON artefact resides.

    :type metrics_path: Path

    :returns: Slate sweep outcome reconstructed from cached metrics.

    :rtype: SweepOutcome

    """
    summary = extract_metric_summary(metrics)
    eligible = summary.n_eligible if summary.n_eligible is not None else int(metrics.get("n_eligible", 0))
    best_k = summary.best_k if summary.best_k is not None else int(metrics.get("best_k", 0))
    accuracy = summary.accuracy if summary.accuracy is not None else float(metrics.get("accuracy_overall", 0.0))
    return SweepOutcome(
        order_index=task.index,
        study=task.study,
        feature_space=task.config.feature_space,
        config=task.config,
        accuracy=accuracy,
        best_k=best_k,
        eligible=eligible,
        metrics_path=metrics_path,
        metrics=metrics,
    )

def opinion_sweep_outcome_from_metrics(
    task: OpinionSweepTask,
    metrics: Mapping[str, object],
    metrics_path: Path,
) -> OpinionSweepOutcome:
    """
    Translate cached opinion metrics into an :class:`OpinionSweepOutcome`.

    :param task: Individual sweep task describing an execution unit.

    :type task: OpinionSweepTask

    :param metrics: Metrics dictionary captured from a previous pipeline stage.

    :type metrics: Mapping[str, object]

    :param metrics_path: Filesystem path where the metrics JSON artefact resides.

    :type metrics_path: Path

    :returns: Opinion sweep outcome reconstructed from cached metrics.

    :rtype: OpinionSweepOutcome

    """
    summary = extract_opinion_summary(metrics)
    best_metrics = metrics.get("best_metrics", {})
    mae = summary.mae if summary.mae is not None else float(metrics.get("best_mae", float("inf")))
    rmse = summary.rmse if summary.rmse is not None else float(best_metrics.get("rmse_after", 0.0))
    r2_score = summary.r2_score
    if r2_score is None:
        r2_score = best_metrics.get("r2_after", 0.0)
        r2_score = float(r2_score) if r2_score is not None else 0.0
    participants = summary.participants if summary.participants is not None else int(
        metrics.get("n_participants", 0)
    )
    best_k = summary.best_k if summary.best_k is not None else int(metrics.get("best_k", 0))
    baseline_mae = summary.baseline_mae
    if baseline_mae is None:
        baseline_mae_raw = metrics.get("baseline", {}).get("mae_using_before")
        baseline_mae = float(baseline_mae_raw) if baseline_mae_raw is not None else None
    mae_delta = summary.mae_delta
    if mae_delta is None and baseline_mae is not None:
        mae_delta = float(mae) - float(baseline_mae)
    accuracy = summary.accuracy
    if accuracy is None:
        accuracy = best_metrics.get("direction_accuracy")
        accuracy = float(accuracy) if accuracy is not None else None
    baseline_accuracy = summary.baseline_accuracy
    if baseline_accuracy is None:
        baseline_accuracy = metrics.get("baseline", {}).get("direction_accuracy")
        baseline_accuracy = (
            float(baseline_accuracy) if baseline_accuracy is not None else None
        )
    accuracy_delta = summary.accuracy_delta
    if accuracy_delta is None and accuracy is not None and baseline_accuracy is not None:
        accuracy_delta = accuracy - baseline_accuracy
    eligible = summary.eligible
    if eligible is None:
        eligible = metrics.get("eligible")
        eligible = int(eligible) if isinstance(eligible, (int, float)) else None
    if eligible is None:
        eligible = participants
    return OpinionSweepOutcome(
        order_index=task.index,
        study=task.study,
        config=task.config,
        feature_space=task.config.feature_space,
        mae=float(mae),
        rmse=float(rmse),
        r2_score=float(r2_score),
        baseline_mae=baseline_mae,
        mae_delta=mae_delta,
        accuracy=accuracy,
        baseline_accuracy=baseline_accuracy,
        accuracy_delta=accuracy_delta,
        best_k=best_k,
        participants=participants,
        eligible=eligible,
        metrics_path=metrics_path,
        metrics=metrics,
    )

def merge_sweep_outcomes(
    cached: Sequence[SweepOutcome],
    executed: Sequence[SweepOutcome],
) -> List[SweepOutcome]:
    """
    Combine cached and freshly executed sweep outcomes preserving order.

    :param cached: Previously computed artefacts available for reuse.

    :type cached: Sequence[SweepOutcome]

    :param executed: Iterable of tasks that were actually executed during the run.

    :type executed: Sequence[SweepOutcome]

    :returns: Mapping of feature spaces to merged slate sweep outcomes.

    :rtype: List[SweepOutcome]

    """
    def _on_replace(_existing: SweepOutcome, incoming: SweepOutcome) -> None:
        """
        Emit a warning when a cached sweep outcome is replaced.

        :param _existing: Outcome currently registered for the task index.
        :type _existing: SweepOutcome
        :param incoming: Newly produced outcome that will replace ``_existing``.
        :type incoming: SweepOutcome
        :returns: ``None``. The function logs a warning for visibility.
        :rtype: None
        """

        LOGGER.warning(
            "Duplicate sweep outcome detected for index=%d (feature=%s study=%s). Overwriting.",
            incoming.order_index,
            incoming.feature_space,
            incoming.study.key,
        )

    return merge_ordered(
        cached,
        executed,
        order_key=lambda outcome: outcome.order_index,
        on_replace=_on_replace,
    )

def merge_opinion_sweep_outcomes(
    cached: Sequence[OpinionSweepOutcome],
    executed: Sequence[OpinionSweepOutcome],
) -> List[OpinionSweepOutcome]:
    """
    Combine cached and freshly executed opinion outcomes preserving order.

    :param cached: Previously computed artefacts available for reuse.

    :type cached: Sequence[OpinionSweepOutcome]

    :param executed: Iterable of tasks that were actually executed during the run.

    :type executed: Sequence[OpinionSweepOutcome]

    :returns: Mapping of feature spaces to merged opinion sweep outcomes.

    :rtype: List[OpinionSweepOutcome]

    """
    def _on_replace(_existing: OpinionSweepOutcome, incoming: OpinionSweepOutcome) -> None:
        """
        Emit a warning when an opinion sweep outcome is replaced.

        :param _existing: Previously cached opinion outcome.
        :type _existing: OpinionSweepOutcome
        :param incoming: Newly produced outcome that supersedes ``_existing``.
        :type incoming: OpinionSweepOutcome
        :returns: ``None``. The function logs a warning for visibility.
        :rtype: None
        """

        LOGGER.warning(
            "Duplicate opinion sweep outcome detected for index=%d (study=%s). Overwriting.",
            incoming.order_index,
            incoming.study.key,
        )

    return merge_ordered(
        cached,
        executed,
        order_key=lambda outcome: outcome.order_index,
        on_replace=_on_replace,
    )

def execute_sweep_tasks(
    tasks: Sequence[SweepTask],
    *,
    jobs: int,
) -> List[SweepOutcome]:
    """
    Run the supplied sweep tasks (possibly in parallel).

    :param tasks: Collection of sweep tasks scheduled for execution.

    :type tasks: Sequence[SweepTask]

    :param jobs: Maximum number of parallel workers to schedule.

    :type jobs: int

    :returns: List of slate sweep outcomes generated from the provided tasks.

    :rtype: List[SweepOutcome]

    """
    return execute_indexed_tasks(tasks, execute_sweep_task, jobs=jobs, logger=LOGGER, label="sweep")

def execute_opinion_sweep_tasks(tasks: Sequence[OpinionSweepTask]) -> List[OpinionSweepOutcome]:
    """
    Run the supplied opinion sweep tasks sequentially.

    :param tasks: Collection of sweep tasks scheduled for execution.

    :type tasks: Sequence[OpinionSweepTask]

    :returns: List of opinion sweep outcomes generated from the provided tasks.

    :rtype: List[OpinionSweepOutcome]

    """
    if not tasks:
        return []
    return execute_sequential_tasks(tasks, execute_opinion_sweep_task)

def execute_sweep_task(task: SweepTask) -> SweepOutcome:
    """
    Execute a single sweep task and return the captured metrics.

    :param task: Individual sweep task describing an execution unit.

    :type task: SweepTask

    :returns: Slate sweep outcome produced by executing the given task.

    :rtype: SweepOutcome

    """
    run_root = ensure_dir(task.run_root)
    model_dir = None
    if task.config.feature_space == "word2vec":
        if task.word2vec_model_dir is None:
            raise RuntimeError("Word2Vec sweep task missing model directory.")
        model_dir = ensure_dir(task.word2vec_model_dir)

    cli_args: List[str] = list(task.base_cli)
    cli_args.extend(task.config.cli_args(word2vec_model_dir=model_dir))
    cli_args.extend(["--issues", task.issue])
    cli_args.extend(["--participant-studies", task.study.key])
    cli_args.extend(["--out-dir", str(run_root)])
    cli_args.extend(task.extra_cli)

    LOGGER.info(
        "[SWEEP] feature=%s study=%s issue=%s label=%s",
        task.config.feature_space,
        task.study.key,
        task.study.issue,
        task.config.label(),
    )
    run_knn_cli(cli_args)
    metrics, metrics_path = load_metrics(run_root, task.issue_slug)
    return sweep_outcome_from_metrics(task, metrics, metrics_path)

def execute_opinion_sweep_task(task: OpinionSweepTask) -> OpinionSweepOutcome:
    """
    Execute a single opinion sweep task and return the captured metrics.

    :param task: Individual sweep task describing an execution unit.

    :type task: OpinionSweepTask

    :returns: Opinion sweep outcome produced by executing the given task.

    :rtype: OpinionSweepOutcome

    """
    run_root = ensure_dir(task.run_root)
    outputs_root = run_root / "opinion" / task.config.feature_space
    model_dir = None
    if task.config.feature_space == "word2vec":
        if task.word2vec_model_dir is None:
            raise RuntimeError("Word2Vec opinion sweep task missing model directory.")
        model_dir = ensure_dir(task.word2vec_model_dir)

    cli_args: List[str] = list(task.base_cli)
    cli_args.extend(task.config.cli_args(word2vec_model_dir=model_dir))
    cli_args.extend(["--task", "opinion"])
    cli_args.extend(["--out-dir", str(run_root)])
    cli_args.extend(["--opinion-studies", task.study.key])
    cli_args.extend(task.extra_cli)

    LOGGER.info(
        "[OPINION][SWEEP] feature=%s study=%s label=%s",
        task.config.feature_space,
        task.study.key,
        task.config.label(),
    )
    run_knn_cli(cli_args)
    metrics_path = (
        outputs_root / task.study.key / f"opinion_knn_{task.study.key}_validation_metrics.json"
    )
    if not metrics_path.exists():
        raise FileNotFoundError(f"Opinion sweep metrics missing at {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return opinion_sweep_outcome_from_metrics(task, metrics, metrics_path)

def emit_sweep_plan(tasks: Sequence[SweepTask]) -> None:
    """
    Print a human-readable sweep plan for next-video tasks.

    :param tasks: Collection of sweep tasks scheduled for execution.

    :type tasks: Sequence[SweepTask]

    :returns: None.

    :rtype: None

    """
    print(f"TOTAL_TASKS={len(tasks)}")
    if not tasks:
        return
    print("INDEX\tSTUDY\tISSUE\tFEATURE_SPACE\tLABEL")
    for display_index, task in enumerate(tasks):
        print(
            f"{display_index}\t{task.study.key}\t{task.issue}\t"
            f"{task.config.feature_space}\t{task.config.label()}"
        )

def emit_combined_sweep_plan(
    *,
    slate_tasks: Sequence[SweepTask],
    opinion_tasks: Sequence[OpinionSweepTask],
) -> None:
    """
    Print a combined sweep plan covering next-video and opinion tasks.

    :param slate_tasks: Slate sweep tasks prepared for execution.

    :type slate_tasks: Sequence[SweepTask]

    :param opinion_tasks: Opinion sweep tasks queued for execution.

    :type opinion_tasks: Sequence[OpinionSweepTask]

    :returns: None.

    :rtype: None

    """
    total = len(slate_tasks) + len(opinion_tasks)
    print(f"TOTAL_TASKS={total}")
    if slate_tasks:
        print("### NEXT_VIDEO")
        print("INDEX\tSTUDY\tISSUE\tFEATURE_SPACE\tLABEL")
        for display_index, task in enumerate(slate_tasks):
            print(
                f"{display_index}\t{task.study.key}\t{task.issue}\t"
                f"{task.config.feature_space}\t{task.config.label()}"
            )
    if opinion_tasks:
        print("### OPINION")
        print("INDEX\tSTUDY\tFEATURE_SPACE\tLABEL")
        for display_index, task in enumerate(opinion_tasks):
            print(
                f"{display_index}\t{task.study.key}\t{task.config.feature_space}\t"
                f"{task.config.label()}"
            )

def format_sweep_task_descriptor(task: SweepTask) -> str:
    """
    Return a short descriptor for a sweep task.

    :param task: Individual sweep task describing an execution unit.

    :type task: SweepTask

    :returns: a short descriptor for a sweep task

    :rtype: str

    """
    return f"{task.config.feature_space}:{task.study.key}:{task.config.label()}"

def format_opinion_sweep_task_descriptor(task: OpinionSweepTask) -> str:
    """
    Return a short descriptor for an opinion sweep task.

    :param task: Individual sweep task describing an execution unit.

    :type task: OpinionSweepTask

    :returns: a short descriptor for an opinion sweep task

    :rtype: str

    """
    return f"{task.config.feature_space}:{task.study.key}:{task.config.label()}"

def run_sweeps(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: SweepTaskContext,
    reuse_existing: bool,
    jobs: int,
) -> List[SweepOutcome]:
    """
    Execute hyper-parameter sweeps and collect per-run metrics.

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :param configs: Iterable of sweep configurations scheduled for execution.

    :type configs: Sequence[SweepConfig]

    :param context: Shared CLI/runtime parameters reused across sweep invocations.
    :type context: SweepTaskContext

    :param reuse_existing: Whether to reuse cached results instead of recomputing them.

    :type reuse_existing: bool

    :param jobs: Maximum number of parallel workers to schedule.

    :type jobs: int

    :returns: Tuple containing slate tasks, opinion tasks, and any cached sweep outcomes.

    :rtype: List[SweepOutcome]

    """
    pending_tasks, cached_outcomes = prepare_sweep_tasks(
        studies=studies,
        configs=configs,
        context=context,
        reuse_existing=reuse_existing,
    )
    executed_outcomes = execute_sweep_tasks(pending_tasks, jobs=jobs)
    return merge_sweep_outcomes(cached_outcomes, executed_outcomes)

def select_best_configs(
    *,
    outcomes: Sequence[SweepOutcome],
    studies: Sequence[StudySpec],
    allow_incomplete: bool = False,
) -> Dict[str, Dict[str, StudySelection]]:
    """
    Select the best configuration per feature space and study.

    :param outcomes: Iterable of sweep outcomes available for aggregation.

    :type outcomes: Sequence[SweepOutcome]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :param allow_incomplete: Whether processing may continue when some sweep data is missing.

    :type allow_incomplete: bool

    :returns: Mapping of feature spaces to their selected slate configurations.

    :rtype: Dict[str, Dict[str, StudySelection]]

    """
    def _is_better(candidate: SweepOutcome, incumbent: SweepOutcome) -> bool:
        """
        Determine whether ``candidate`` should replace the incumbent outcome.

        :param candidate: Candidate sweep outcome under consideration.
        :type candidate: SweepOutcome
        :param incumbent: Currently selected sweep outcome.
        :type incumbent: SweepOutcome
        :returns: ``True`` when ``candidate`` offers a preferable accuracy/eligibility trade-off.
        :rtype: bool
        """
        if candidate.accuracy > incumbent.accuracy + 1e-9:
            return True
        if candidate.accuracy + 1e-9 < incumbent.accuracy:
            return False
        if candidate.eligible > incumbent.eligible:
            return True
        if candidate.eligible < incumbent.eligible:
            return False
        return candidate.best_k < incumbent.best_k

    selections: Dict[str, Dict[str, StudySelection]] = {}
    for outcome in outcomes:
        per_feature = selections.setdefault(outcome.feature_space, {})
        current = per_feature.get(outcome.study.key)
        if current is None or _is_better(outcome, current.outcome):
            per_feature[outcome.study.key] = StudySelection(study=outcome.study, outcome=outcome)

    expected_keys = [study.key for study in studies]
    for feature_space, per_feature in selections.items():
        missing = [key for key in expected_keys if key not in per_feature]
        if missing:
            if allow_incomplete:
                LOGGER.warning(
                    "Missing sweep selections for feature=%s: %s. Continuing because allow-incomplete mode is enabled.",
                    feature_space,
                    ", ".join(missing),
                )
            else:
                raise RuntimeError(
                    f"Missing sweep selections for feature={feature_space}: {', '.join(missing)}"
                )
    if not selections:
        raise RuntimeError("Failed to select a best configuration for any feature space.")
    return selections

def select_best_opinion_configs(
    *,
    outcomes: Sequence[OpinionSweepOutcome],
    studies: Sequence[StudySpec],
    allow_incomplete: bool = False,
) -> Dict[str, Dict[str, OpinionStudySelection]]:
    """
    Select the best configuration per feature space and study for opinion.

    :param outcomes: Iterable of sweep outcomes available for aggregation.

    :type outcomes: Sequence[OpinionSweepOutcome]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[StudySpec]

    :param allow_incomplete: Whether processing may continue when some sweep data is missing.

    :type allow_incomplete: bool

    :returns: Mapping of feature spaces to their selected opinion configurations.

    :rtype: Dict[str, Dict[str, OpinionStudySelection]]

    """
    def _is_better(candidate: OpinionSweepOutcome, incumbent: OpinionSweepOutcome) -> bool:
        """Return ``True`` when ``candidate`` should replace ``incumbent``."""

        def _compare_metric(lhs: float | None, rhs: float | None) -> int:
            if lhs is None or rhs is None:
                return 0
            delta = lhs - rhs
            if abs(delta) <= 1e-9:
                return 0
            return -1 if delta < 0 else 1

        for lhs, rhs in ((candidate.mae, incumbent.mae), (candidate.rmse, incumbent.rmse)):
            result = _compare_metric(lhs, rhs)
            if result != 0:
                return result < 0

        candidate_participants = candidate.participants or 0
        incumbent_participants = incumbent.participants or 0
        if candidate_participants != incumbent_participants:
            return candidate_participants > incumbent_participants
        return candidate.best_k < incumbent.best_k

    selections: Dict[str, Dict[str, OpinionStudySelection]] = {}
    for outcome in outcomes:
        per_feature = selections.setdefault(outcome.feature_space, {})
        current = per_feature.get(outcome.study.key)
        if current is None or _is_better(outcome, current.outcome):
            per_feature[outcome.study.key] = OpinionStudySelection(
                study=outcome.study,
                outcome=outcome,
            )

    expected_keys = [study.key for study in studies]
    for feature_space, per_feature in selections.items():
        missing = [key for key in expected_keys if key not in per_feature]
        if missing:
            if allow_incomplete:
                LOGGER.warning(
                    "Missing opinion sweep selections for feature=%s: %s. Continuing because allow-incomplete mode is enabled.",
                    feature_space,
                    ", ".join(missing),
                )
            else:
                raise RuntimeError(
                    f"Missing opinion sweep selections for feature={feature_space}: {', '.join(missing)}"
                )
    if not selections and outcomes:
        raise RuntimeError("Failed to select a best configuration for any opinion feature space.")
    return selections

__all__ = [
    "build_sweep_configs",
    "emit_combined_sweep_plan",
    "emit_sweep_plan",
    "execute_opinion_sweep_task",
    "execute_opinion_sweep_tasks",
    "execute_sweep_task",
    "execute_sweep_tasks",
    "format_opinion_sweep_task_descriptor",
    "format_sweep_task_descriptor",
    "merge_opinion_sweep_outcomes",
    "merge_sweep_outcomes",
    "opinion_sweep_outcome_from_metrics",
    "prepare_opinion_sweep_tasks",
    "prepare_sweep_tasks",
    "run_knn_cli",
    "run_sweeps",
    "select_best_configs",
    "select_best_opinion_configs",
    "sweep_outcome_from_metrics",
]
