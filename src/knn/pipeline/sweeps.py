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

"""Sweep orchestration helpers for the Grail Simulation KNN pipeline.

Builds hyper-parameter grids, prepares CLI invocations, executes sweep tasks
for next-video and opinion runs, and merges cached metrics so later stages can
select the best configurations.
"""
from __future__ import annotations

import logging
import os
import json
import re
from itertools import product
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from common.pipeline.executor import execute_indexed_tasks
from common.pipeline.types import BasePipelineSweepOutcome
from common.pipeline.utils import merge_indexed_outcomes, base_sweep_outcome_kwargs
from common.prompts.docs import merge_default_extra_fields

# Access the sibling module through the package namespace to satisfy pylint checks.
from knn.pipeline import opinion_sweeps as _opinion_sweeps
from ..cli import build_parser as build_knn_parser
from ..core.evaluate import run_eval
from ..core.opinion import run_opinion_eval
from .context import (
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
from .context_sweeps import _KnnSweepStats
from .data import issue_slug_for_study
from .io import load_metrics
from .utils import (
    ensure_dir,
    ensure_sweep_selection_coverage,
    extract_metric_summary,
    prepare_task_grid,
    TaskCacheStrategy,
)

LOGGER = logging.getLogger("knn.pipeline.sweeps")

BASE_EXTRA_FIELD_SETS: Tuple[Tuple[str, ...], ...] = (
    (),
    ("ideo1",),
    ("ideo2",),
    ("pol_interest",),
    ("religpew",),
    ("freq_youtube",),
    ("youtube_time",),
    ("newsint",),
    ("slate_source",),
    ("educ",),
    ("employ",),
    ("child18",),
    ("inputstate",),
    ("income",),
    ("participant_study",),
)

DEFAULT_TEXT_OPTION_LIMIT: int = len(BASE_EXTRA_FIELD_SETS)


def _materialise_text_options(limit: int) -> Tuple[Tuple[str, ...], ...]:
    """Return the merged text options capped at ``limit``."""

    options: List[Tuple[str, ...]] = []
    seen: set[Tuple[str, ...]] = set()
    for extra_fields in BASE_EXTRA_FIELD_SETS[:limit]:
        merged = merge_default_extra_fields(extra_fields)
        if merged not in seen:
            options.append(merged)
            seen.add(merged)
    if BASE_EXTRA_FIELD_SETS:
        all_extra_fields: List[str] = []
        for extra_fields in BASE_EXTRA_FIELD_SETS:
            for field in extra_fields:
                if field and field not in all_extra_fields:
                    all_extra_fields.append(field)
        merged_all = merge_default_extra_fields(tuple(all_extra_fields))
        if merged_all not in seen:
            options.append(merged_all)
            seen.add(merged_all)
    return tuple(options)


def _parse_metric_env(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    """Parse comma-separated metrics from ``name`` while filtering unsupported values."""

    raw = os.environ.get(name, "")
    if not raw:
        return default
    allowed = {"cosine", "l2"}
    tokens: List[str] = []
    for token in (part.strip().lower() for part in raw.split(",")):
        if not token:
            continue
        if token not in allowed:
            LOGGER.warning("Ignoring unsupported metric '%s' in %s override.", token, name)
            continue
        if token not in tokens:
            tokens.append(token)
    if not tokens:
        LOGGER.warning(
            "Ignoring %s override '%s' because no valid metrics were provided.",
            name,
            raw,
        )
        return default
    return tuple(tokens)


def _parse_limit_env(name: str, default: int) -> int:
    """Parse an integer environment override ensuring a positive value."""

    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        LOGGER.warning("Ignoring non-integer %s override '%s'.", name, raw)
        return default
    if value <= 0:
        LOGGER.warning(
            "Ignoring %s override '%s' because the limit must be positive.",
            name,
            raw,
        )
        return default
    return value


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

def _discover_existing_word2vec_workers(context: PipelineContext) -> Tuple[int, ...]:
    """Return worker counts inferred from cached Word2Vec sweep artefacts."""

    candidates: set[int] = set()
    roots = [
        context.sweep_dir / "word2vec",
        context.word2vec_model_dir / "sweeps",
    ]
    for root in roots:
        try:
            entries = list(root.iterdir())
        except FileNotFoundError:
            continue
        except NotADirectoryError:
            continue
        for study_dir in entries:
            if not study_dir.is_dir():
                continue
            try:
                config_dirs = list(study_dir.iterdir())
            except FileNotFoundError:
                continue
            except NotADirectoryError:
                continue
            for config_dir in config_dirs:
                if not config_dir.is_dir():
                    continue
                match = re.search(r"_workers(\d+)", config_dir.name)
                if not match:
                    continue
                try:
                    candidates.add(int(match.group(1)))
                except ValueError:
                    LOGGER.debug(
                        "Ignoring malformed Word2Vec worker suffix in %s.",
                        config_dir,
                    )
    return tuple(sorted(candidates))


def _word2vec_param_grid(context: PipelineContext) -> Dict[str, Tuple[int, ...]]:
    """Read Word2Vec sweep parameters from environment variables or cached artefacts."""

    def _parse_env(name: str, defaults: Sequence[int]) -> Tuple[int, ...]:
        raw = os.environ.get(name)
        tokens = raw.split(",") if raw else [str(value) for value in defaults]
        values: list[int] = []
        seen: set[int] = set()
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            try:
                value = int(token)
            except ValueError:
                LOGGER.warning("Ignoring non-integer %s override token '%s'.", name, token)
                continue
            if value not in seen:
                values.append(value)
                seen.add(value)
        if not values:
            values = list(defaults)
        return tuple(values)

    discovered_workers = _discover_existing_word2vec_workers(context)
    worker_defaults: Tuple[int, ...]
    if os.environ.get("WORD2VEC_SWEEP_WORKERS"):
        # Respect explicit overrides entirely.
        worker_defaults = discovered_workers if discovered_workers else (context.word2vec_workers,)
    else:
        worker_defaults = discovered_workers or (context.word2vec_workers,)

    # Keep defaults intentionally narrow so the sweep stays well under 100 configs.
    return {
        "sizes": _parse_env("WORD2VEC_SWEEP_SIZES", (128, 256)),
        "windows": _parse_env("WORD2VEC_SWEEP_WINDOWS", (5,)),
        "min_counts": _parse_env("WORD2VEC_SWEEP_MIN_COUNTS", (1,)),
        "epochs": _parse_env(
            "WORD2VEC_SWEEP_EPOCHS", (context.word2vec_epochs,)
        ),
        "workers": _parse_env(
            "WORD2VEC_SWEEP_WORKERS", worker_defaults
        ),
    }


def _build_tfidf_configs(metrics: Tuple[str, ...], limit: int) -> List[SweepConfig]:
    """Return TF-IDF sweep configs for the given ``metrics`` and text option ``limit``."""

    configs: List[SweepConfig] = []
    text_options = _materialise_text_options(limit)
    for metric in metrics:
        for fields in text_options:
            configs.append(
                SweepConfig(
                    feature_space="tfidf",
                    metric=metric,
                    text_fields=fields,
                )
            )
    return configs


def _build_word2vec_configs(
    context: PipelineContext, metrics: Tuple[str, ...], limit: int
) -> List[SweepConfig]:
    """Return Word2Vec sweep configs using ``context`` defaults and ``metrics``."""

    configs: List[SweepConfig] = []
    text_options = _materialise_text_options(limit)
    param_grid = _word2vec_param_grid(context)
    for metric in metrics:
        for fields in text_options:
            for size, window, min_count, epochs, workers in product(
                param_grid["sizes"],
                param_grid["windows"],
                param_grid["min_counts"],
                param_grid["epochs"],
                param_grid["workers"],
            ):
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
    return configs


def _build_sentence_transformer_configs(
    context: PipelineContext, metrics: Tuple[str, ...], limit: int
) -> List[SweepConfig]:
    """Return SentenceTransformer sweep configs from ``context`` and ``metrics``."""

    configs: List[SweepConfig] = []
    text_options = _materialise_text_options(limit)
    for metric in metrics:
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


def build_sweep_configs(context: PipelineContext) -> List[SweepConfig]:
    """
    Return the grid of configurations evaluated during sweeps.

    :param context: Pipeline context encapsulating dataset paths and configuration flags.

    :type context: ~knn.pipeline.context.PipelineContext

    :returns: the grid of configurations evaluated during sweeps

    :rtype: List[~knn.pipeline.context.SweepConfig]

    """
    feature_spaces = context.feature_spaces

    tfidf_metrics = _parse_metric_env("KNN_TFIDF_METRICS", ("cosine", "l2"))
    word2vec_metrics = _parse_metric_env("KNN_WORD2VEC_METRICS", ("cosine", "l2"))
    sentence_metrics = _parse_metric_env("KNN_SENTENCE_METRICS", ("cosine", "l2"))
    tfidf_limit = _parse_limit_env("KNN_TFIDF_TEXT_LIMIT", DEFAULT_TEXT_OPTION_LIMIT)
    word2vec_limit = _parse_limit_env("KNN_WORD2VEC_TEXT_LIMIT", DEFAULT_TEXT_OPTION_LIMIT)
    sentence_limit = _parse_limit_env("KNN_SENTENCE_TEXT_LIMIT", DEFAULT_TEXT_OPTION_LIMIT)

    configs: List[SweepConfig] = []
    if "tfidf" in feature_spaces:
        configs.extend(_build_tfidf_configs(tfidf_metrics, tfidf_limit))
    if "word2vec" in feature_spaces:
        configs.extend(_build_word2vec_configs(context, word2vec_metrics, word2vec_limit))
    if "sentence_transformer" in feature_spaces:
        configs.extend(
            _build_sentence_transformer_configs(context, sentence_metrics, sentence_limit)
        )
    return configs

@dataclass(frozen=True)
class _TaskBuildExtras:
    """Bundle of shared inputs used when constructing sweep tasks."""

    context: SweepTaskContext
    base_cli: Tuple[str, ...]
    extra_cli: Tuple[str, ...]
    train_keys_func: Callable[[str], Tuple[str, ...]]


def _build_sweep_task(
    *, index: int, config: SweepConfig, study: StudySpec, extras: _TaskBuildExtras
) -> SweepTask:
    """Construct a sweep task for the next-video pipeline."""

    issue_slug = issue_slug_for_study(study)
    run_root = (
        extras.context.sweep_dir / config.feature_space / study.study_slug / config.label()
    )
    metrics_path = run_root / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
    word2vec_model_dir = None
    if config.feature_space == "word2vec":
        word2vec_model_dir = (
            extras.context.word2vec_model_base / "sweeps" / study.study_slug / config.label()
        )
    return SweepTask(
        index=index,
        study=study,
        config=config,
        base_cli=extras.base_cli,
        extra_cli=extras.extra_cli,
        run_root=run_root,
        word2vec_model_dir=word2vec_model_dir,
        issue=study.issue,
        issue_slug=issue_slug,
        metrics_path=metrics_path,
        train_participant_studies=extras.train_keys_func(study.key),
    )


def _load_cached_outcome(task: SweepTask) -> Optional[SweepOutcome]:
    """Load cached metrics for ``task`` when available."""

    try:
        metrics, metrics_path = load_metrics(task.run_root, task.issue_slug)
    except FileNotFoundError:
        LOGGER.debug("Expected cached metrics at %s but none found.", task.metrics_path)
        return None
    LOGGER.info(
        "[SWEEP][SKIP] feature=%s study=%s label=%s (metrics cached).",
        task.config.feature_space,
        task.study.key,
        task.config.label(),
    )
    return sweep_outcome_from_metrics(task, metrics, metrics_path)


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

    :type studies: Sequence[~knn.pipeline.context.StudySpec]

    :param configs: Iterable of sweep configurations scheduled for execution.

    :type configs: Sequence[~knn.pipeline.context.SweepConfig]

    :param context: Shared CLI/runtime parameters reused across sweep invocations.
    :type context: ~knn.pipeline.context.SweepTaskContext

    :param reuse_existing: Whether to reuse cached results instead of recomputing them.

    :type reuse_existing: bool

    :returns: next-video sweep tasks requiring execution and cached outcomes

    :rtype: Tuple[List[~knn.pipeline.context.SweepTask], List[~knn.pipeline.context.SweepOutcome]]

    """
    base_cli_tuple = tuple(context.base_cli)
    extra_cli_tuple = tuple(context.extra_cli)

    # Restrict training to the evaluation study. Passing an empty tuple allows
    # the CLI to inherit the participant study filter from ``--participant-studies``
    # so no cross-study combinations are introduced during the sweep.
    def _train_keys_for(_study_key: str) -> tuple[str, ...]:
        _ = _study_key  # satisfy lint; the value is implicit via participant-studies
        return tuple()

    extras = _TaskBuildExtras(
        context=context,
        base_cli=base_cli_tuple,
        extra_cli=extra_cli_tuple,
        train_keys_func=_train_keys_for,
    )

    return prepare_task_grid(
        configs,
        studies,
        reuse_existing=reuse_existing,
        build_task=lambda task_index, config, study: _build_sweep_task(
            index=task_index, config=config, study=study, extras=extras
        ),
        cache=TaskCacheStrategy(load_cached=_load_cached_outcome),
    )


def sweep_outcome_from_metrics(
    task: SweepTask,
    metrics: Mapping[str, object],
    metrics_path: Path,
) -> SweepOutcome:
    """
    Translate cached metrics into a :class:`SweepOutcome`.

    :param task: Individual sweep task describing an execution unit.

    :type task: ~knn.pipeline.context.SweepTask

    :param metrics: Metrics dictionary captured from a previous pipeline stage.

    :type metrics: Mapping[str, object]

    :param metrics_path: Filesystem path where the metrics JSON artefact resides.

    :type metrics_path: Path

    :returns: Slate sweep outcome reconstructed from cached metrics.

    :rtype: ~knn.pipeline.context.SweepOutcome

    """
    summary = extract_metric_summary(metrics)
    eligible = (
        summary.n_eligible
        if summary.n_eligible is not None
        else int(metrics.get("n_eligible", 0))
    )
    best_k = summary.best_k if summary.best_k is not None else int(metrics.get("best_k", 0))
    accuracy = (
        summary.accuracy
        if summary.accuracy is not None
        else float(metrics.get("accuracy_overall", 0.0))
    )
    return SweepOutcome(
        base=BasePipelineSweepOutcome(
            **base_sweep_outcome_kwargs(task, metrics, metrics_path)
        ),
        knn=_KnnSweepStats(
            feature_space=task.config.feature_space,
            accuracy=accuracy,
            best_k=best_k,
            eligible=eligible,
        ),
    )

prepare_opinion_sweep_tasks = _opinion_sweeps.prepare_opinion_sweep_tasks
opinion_sweep_outcome_from_metrics = _opinion_sweeps.opinion_sweep_outcome_from_metrics
merge_opinion_sweep_outcomes = _opinion_sweeps.merge_opinion_sweep_outcomes
format_opinion_sweep_task_descriptor = _opinion_sweeps.format_opinion_sweep_task_descriptor
select_best_opinion_configs = _opinion_sweeps.select_best_opinion_configs
build_opinion_task = _opinion_sweeps.build_opinion_task

def merge_sweep_outcomes(
    cached: Sequence[SweepOutcome],
    executed: Sequence[SweepOutcome],
) -> List[SweepOutcome]:
    """
    Combine cached and freshly executed sweep outcomes preserving order.

    :param cached: Previously computed artefacts available for reuse.

    :type cached: Sequence[~knn.pipeline.context.SweepOutcome]

    :param executed: Iterable of tasks that were actually executed during the run.

    :type executed: Sequence[~knn.pipeline.context.SweepOutcome]

    :returns: Mapping of feature spaces to merged slate sweep outcomes.

    :rtype: List[~knn.pipeline.context.SweepOutcome]

    """
    return merge_indexed_outcomes(
        cached,
        executed,
        logger=LOGGER,
        message="Duplicate sweep outcome detected for index=%d (feature=%s study=%s). Overwriting.",
        args_factory=lambda _existing, incoming: (
            incoming.order_index,
            incoming.feature_space,
            incoming.study.key,
        ),
    )

def execute_opinion_sweep_tasks(
    tasks: Sequence[OpinionSweepTask],
    *,
    jobs: int,
) -> List[OpinionSweepOutcome]:
    """
    Run the supplied opinion sweep tasks via the shared CLI runner.

    :param tasks: Opinion sweep tasks scheduled for execution.
    :type tasks: Sequence[~knn.pipeline.context.OpinionSweepTask]
    :param jobs: Maximum number of parallel workers.
    :type jobs: int
    :returns: Ordered list of opinion sweep outcomes.
    :rtype: List[~knn.pipeline.context.OpinionSweepOutcome]
    """
    return _opinion_sweeps.execute_opinion_sweep_tasks(
        tasks,
        jobs=jobs,
        cli_runner=run_knn_cli,
    )


def execute_opinion_sweep_task(task: OpinionSweepTask) -> OpinionSweepOutcome:
    """
    Execute a single opinion sweep task using the shared CLI runner.

    :param task: Opinion sweep task describing a single execution unit.
    :returns: Outcome bundle produced by running the CLI for ``task``.
    """
    return _opinion_sweeps.execute_opinion_sweep_task(task, cli_runner=run_knn_cli)


def execute_sweep_tasks(
    tasks: Sequence[SweepTask],
    *,
    jobs: int,
) -> List[SweepOutcome]:
    """
    Run the supplied sweep tasks (possibly in parallel).

    :param tasks: Collection of sweep tasks scheduled for execution.

    :type tasks: Sequence[~knn.pipeline.context.SweepTask]

    :param jobs: Maximum number of parallel workers to schedule.

    :type jobs: int

    :returns: List of slate sweep outcomes generated from the provided tasks.

    :rtype: List[~knn.pipeline.context.SweepOutcome]

    """
    return execute_indexed_tasks(tasks, execute_sweep_task, jobs=jobs, logger=LOGGER, label="sweep")

def execute_sweep_task(task: SweepTask) -> SweepOutcome:
    """
    Execute a single sweep task and return the captured metrics.

    :param task: Individual sweep task describing an execution unit.

    :type task: ~knn.pipeline.context.SweepTask

    :returns: Slate sweep outcome produced by executing the given task.

    :rtype: ~knn.pipeline.context.SweepOutcome

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
    if task.train_participant_studies:
        cli_args.extend([
            "--train-participant-studies",
            ",".join(task.train_participant_studies),
        ])
    else:
        # Within-study training: do not include alternate studies.
        LOGGER.info(
            "[SWEEP] feature=%s study=%s training restricted to within-study only.",
            task.config.feature_space,
            task.study.key,
        )

    LOGGER.info(
        "[SWEEP] feature=%s study=%s issue=%s label=%s",
        task.config.feature_space,
        task.study.key,
        task.study.issue,
        task.config.label(),
    )
    run_knn_cli(cli_args)
    try:
        metrics, metrics_path = load_metrics(run_root, task.issue_slug)
    except FileNotFoundError:
        # The evaluation may be legitimately skipped (e.g. no training rows with --fit-index
        # or no eligible evaluation rows after filters). In that case, the CLI does not
        # emit a metrics file. Fall back to a zeroed-out metrics payload so the sweep can
        # proceed and downstream selection logic can operate in allow-incomplete mode.
        LOGGER.warning(
            (
                "[SWEEP][SKIP] feature=%s study=%s label=%s "
                "(no metrics written; likely skipped by filters)"
            ),
            task.config.feature_space,
            task.study.key,
            task.config.label(),
        )
        metrics_path = task.metrics_path
        metrics = {
            "model": "knn",
            "feature_space": task.config.feature_space,
            "issue": task.issue_slug,
            "split": "validation",
            # Provide minimal fields consumed by extract_metric_summary/sweep selection.
            "accuracy_overall": 0.0,
            "accuracy_overall_all_rows": 0.0,
            "best_k": 0,
            "n_total": 0,
            "n_eligible": 0,
            "skipped": True,
            "skip_reason": "No metrics written (evaluation skipped)",
        }
        # Persist placeholder metrics so finalize/report stages and reuse logic can see the skip.
        try:
            metrics_dir = metrics_path.parent
            metrics_dir.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, ensure_ascii=False, indent=2)
        except OSError as exc:  # pragma: no cover - defensive logging only
            LOGGER.warning("Failed to write placeholder metrics at %s: %s", metrics_path, exc)
    return sweep_outcome_from_metrics(task, metrics, metrics_path)

def emit_sweep_plan(tasks: Sequence[SweepTask]) -> None:
    """
    Print a human-readable sweep plan for next-video tasks.

    :param tasks: Collection of sweep tasks scheduled for execution.

    :type tasks: Sequence[~knn.pipeline.context.SweepTask]

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

    :type slate_tasks: Sequence[~knn.pipeline.context.SweepTask]

    :param opinion_tasks: Opinion sweep tasks queued for execution.

    :type opinion_tasks: Sequence[~knn.pipeline.context.OpinionSweepTask]

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

    :type task: ~knn.pipeline.context.SweepTask

    :returns: a short descriptor for a sweep task

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

    :type studies: Sequence[~knn.pipeline.context.StudySpec]

    :param configs: Iterable of sweep configurations scheduled for execution.

    :type configs: Sequence[~knn.pipeline.context.SweepConfig]

    :param context: Shared CLI/runtime parameters reused across sweep invocations.
    :type context: ~knn.pipeline.context.SweepTaskContext

    :param reuse_existing: Whether to reuse cached results instead of recomputing them.

    :type reuse_existing: bool

    :param jobs: Maximum number of parallel workers to schedule.

    :type jobs: int

    :returns: Tuple containing slate tasks, opinion tasks, and any cached sweep outcomes.

    :rtype: List[~knn.pipeline.context.SweepOutcome]

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

    :type outcomes: Sequence[~knn.pipeline.context.SweepOutcome]

    :param studies: Sequence of study specifications targeted by the workflow.

    :type studies: Sequence[~knn.pipeline.context.StudySpec]

    :param allow_incomplete: Whether processing may continue when some sweep data is missing.

    :type allow_incomplete: bool

    :returns: Mapping of feature spaces to their selected slate configurations.

    :rtype: Dict[str, Dict[str, ~knn.pipeline.context.StudySelection]]

    """
    def _is_better(candidate: SweepOutcome, incumbent: SweepOutcome) -> bool:
        """
        Determine whether ``candidate`` should replace the incumbent outcome.

        :param candidate: Candidate sweep outcome under consideration.
        :type candidate: ~knn.pipeline.context.SweepOutcome
        :param incumbent: Currently selected sweep outcome.
        :type incumbent: ~knn.pipeline.context.SweepOutcome
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

    ensure_sweep_selection_coverage(
        selections,
        studies,
        allow_incomplete=allow_incomplete,
        logger=LOGGER,
    )
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
    "build_opinion_task",
    "prepare_opinion_sweep_tasks",
    "prepare_sweep_tasks",
    "run_knn_cli",
    "run_sweeps",
    "select_best_configs",
    "select_best_opinion_configs",
    "sweep_outcome_from_metrics",
]
