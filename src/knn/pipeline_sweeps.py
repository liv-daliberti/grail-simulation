"""Sweep orchestration helpers for the modular KNN pipeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
)
from .pipeline_data import issue_slug_for_study
from .pipeline_io import load_metrics
from .pipeline_utils import ensure_dir, extract_metric_summary, extract_opinion_summary

LOGGER = logging.getLogger("knn.pipeline.sweeps")


def run_knn_cli(argv: Sequence[str]) -> None:
    """Execute the KNN CLI entry point with ``argv``."""

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
    """Return the grid of configurations evaluated during sweeps."""

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
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    sweep_dir: Path,
    word2vec_model_base: Path,
    reuse_existing: bool,
) -> Tuple[List[SweepTask], List[SweepOutcome]]:
    """Return next-video sweep tasks requiring execution and cached outcomes."""

    pending_tasks: List[SweepTask] = []
    cached_outcomes: List[SweepOutcome] = []
    base_cli_tuple = tuple(base_cli)
    extra_cli_tuple = tuple(extra_cli)

    task_index = 0
    for config in configs:
        for study in studies:
            issue_slug = issue_slug_for_study(study)
            run_root = sweep_dir / config.feature_space / study.study_slug / config.label()
            metrics_path = run_root / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
            word2vec_model_dir = None
            if config.feature_space == "word2vec":
                word2vec_model_dir = word2vec_model_base / "sweeps" / study.study_slug / config.label()
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


def prepare_opinion_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    sweep_dir: Path,
    word2vec_model_base: Path,
    reuse_existing: bool,
) -> Tuple[List[OpinionSweepTask], List[OpinionSweepOutcome]]:
    """Return opinion sweep tasks requiring execution and cached outcomes."""

    pending_tasks: List[OpinionSweepTask] = []
    cached_outcomes: List[OpinionSweepOutcome] = []
    base_cli_tuple = tuple(base_cli)
    extra_cli_tuple = tuple(extra_cli)

    task_index = 0
    for config in configs:
        for study in studies:
            run_root = sweep_dir / "opinion" / config.feature_space / study.study_slug / config.label()
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
                    word2vec_model_base / "sweeps_opinion" / study.study_slug / config.label()
                )
            task = OpinionSweepTask(
                index=task_index,
                study=study,
                config=config,
                base_cli=base_cli_tuple,
                extra_cli=extra_cli_tuple,
                run_root=run_root,
                word2vec_model_dir=word2vec_model_dir,
                metrics_path=metrics_path,
            )
            task_index += 1
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
    """Translate cached metrics into a :class:`SweepOutcome`."""

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
    """Translate cached opinion metrics into an :class:`OpinionSweepOutcome`."""

    summary = extract_opinion_summary(metrics)
    mae = summary.mae if summary.mae is not None else float(metrics.get("best_mae", float("inf")))
    rmse = summary.rmse if summary.rmse is not None else float(
        metrics.get("best_metrics", {}).get("rmse_after", 0.0)
    )
    r2 = summary.r2 if summary.r2 is not None else float(
        metrics.get("best_metrics", {}).get("r2_after", 0.0)
    )
    participants = summary.participants if summary.participants is not None else int(
        metrics.get("n_participants", 0)
    )
    best_k = summary.best_k if summary.best_k is not None else int(metrics.get("best_k", 0))
    return OpinionSweepOutcome(
        order_index=task.index,
        study=task.study,
        config=task.config,
        feature_space=task.config.feature_space,
        mae=float(mae),
        rmse=float(rmse),
        r2=float(r2),
        best_k=best_k,
        participants=participants,
        metrics_path=metrics_path,
        metrics=metrics,
    )


def merge_sweep_outcomes(
    cached: Sequence[SweepOutcome],
    executed: Sequence[SweepOutcome],
) -> List[SweepOutcome]:
    """Combine cached and freshly executed sweep outcomes preserving order."""

    def _on_replace(existing: SweepOutcome, incoming: SweepOutcome) -> None:
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
    """Combine cached and freshly executed opinion outcomes preserving order."""

    def _on_replace(existing: OpinionSweepOutcome, incoming: OpinionSweepOutcome) -> None:
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
    """Run the supplied sweep tasks (possibly in parallel)."""

    return execute_indexed_tasks(tasks, execute_sweep_task, jobs=jobs, logger=LOGGER, label="sweep")


def execute_opinion_sweep_tasks(tasks: Sequence[OpinionSweepTask]) -> List[OpinionSweepOutcome]:
    """Run the supplied opinion sweep tasks sequentially."""

    if not tasks:
        return []
    return execute_sequential_tasks(tasks, execute_opinion_sweep_task)


def execute_sweep_task(task: SweepTask) -> SweepOutcome:
    """Execute a single sweep task and return the captured metrics."""

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
    """Execute a single opinion sweep task and return the captured metrics."""

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
    """Print a human-readable sweep plan for next-video tasks."""

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
    """Print a combined sweep plan covering next-video and opinion tasks."""

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
    """Return a short descriptor for a sweep task."""

    return f"{task.config.feature_space}:{task.study.key}:{task.config.label()}"


def format_opinion_sweep_task_descriptor(task: OpinionSweepTask) -> str:
    """Return a short descriptor for an opinion sweep task."""

    return f"{task.config.feature_space}:{task.study.key}:{task.config.label()}"


def run_sweeps(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    sweep_dir: Path,
    word2vec_model_base: Path,
    reuse_existing: bool,
    jobs: int,
) -> List[SweepOutcome]:
    """Execute hyper-parameter sweeps and collect per-run metrics."""

    pending_tasks, cached_outcomes = prepare_sweep_tasks(
        studies=studies,
        configs=configs,
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=sweep_dir,
        word2vec_model_base=word2vec_model_base,
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
    """Select the best configuration per feature space and study."""

    def _is_better(candidate: SweepOutcome, incumbent: SweepOutcome) -> bool:
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
    """Select the best configuration per feature space and study for opinion."""

    def _is_better(candidate: OpinionSweepOutcome, incumbent: OpinionSweepOutcome) -> bool:
        if candidate.mae < incumbent.mae - 1e-9:
            return True
        if candidate.mae > incumbent.mae + 1e-9:
            return False
        if candidate.rmse < incumbent.rmse - 1e-9:
            return True
        if candidate.rmse > incumbent.rmse + 1e-9:
            return False
        if candidate.participants > incumbent.participants:
            return True
        if candidate.participants < incumbent.participants:
            return False
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
