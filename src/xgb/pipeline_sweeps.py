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

"""Sweep orchestration helpers for the Grail Simulation XGBoost pipeline.

Constructs hyper-parameter grids, prepares CLI invocations, executes sweep
runs for next-video and opinion tasks, and merges cached metrics so later
stages can select the best configurations.
"""

from __future__ import annotations

# pylint: disable=too-many-lines

import logging
from dataclasses import replace
import json
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from common.pipeline_executor import execute_indexed_tasks
from common.pipeline_io import load_metrics_json
from common.pipeline_utils import merge_indexed_outcomes
from common.opinion_sweep_types import AccuracySummary, MetricsArtifact

from .cli import build_parser as build_xgb_parser
from .evaluate import run_eval
from .opinion import (
    OpinionEvalRequest,
    OpinionTrainConfig,
    OpinionVectorizerConfig,
    run_opinion_eval,
)
from .pipeline_context import (
    OpinionStudySelection,
    OpinionSweepOutcome,
    OpinionSweepRunContext,
    OpinionSweepTask,
    StudySelection,
    StudySpec,
    SweepConfig,
    SweepOutcome,
    SweepRunContext,
    SweepTask,
)

try:  # pragma: no cover - optional dependency
    import xgboost  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    xgboost = None  # type: ignore[assignment]

DEFAULT_OPINION_FEATURE_SPACE = "tfidf"
LOGGER = logging.getLogger("xgb.pipeline.sweeps")


def _inject_study_metadata(metrics: Dict[str, object], spec: StudySpec) -> None:
    """Ensure study metadata fields exist in a metrics payload."""

    metrics.setdefault("issue", spec.issue)
    metrics.setdefault("issue_label", spec.issue.replace("_", " ").title())
    metrics.setdefault("study", spec.key)
    metrics.setdefault("study_label", spec.label)


def _load_metrics_with_log(
    metrics_path: Path,
    spec: StudySpec,
    *,
    log_level: int,
    message: str,
) -> Dict[str, object] | None:
    """Load metrics, logging a message when they cannot be retrieved."""

    try:
        return dict(_load_metrics(metrics_path))
    except FileNotFoundError:
        LOGGER.log(log_level, message, spec.issue, spec.key, metrics_path)
        return None


def _run_xgb_cli(args: Sequence[str]) -> None:
    """Execute the :mod:`xgb.cli` entry point with the supplied arguments."""

    parser = build_xgb_parser()
    namespace = parser.parse_args(list(args))
    run_eval(namespace)


def _load_metrics(path: Path) -> Mapping[str, object]:
    """Load the metrics JSON emitted by a sweep or evaluation task."""

    return load_metrics_json(path)

def _sweep_outcome_from_metrics(
    task: SweepTask,
    metrics: Mapping[str, object],
    metrics_path: Path,
) -> SweepOutcome:
    """Convert cached sweep metrics into an outcome instance."""

    # Prefer eligible-only accuracy when available; fall back to overall.
    acc_value = metrics.get("accuracy_eligible")
    if acc_value is None:
        acc_value = metrics.get("accuracy")
    return SweepOutcome(
        order_index=task.index,
        study=task.study,
        config=task.config,
        accuracy=float(acc_value or 0.0),
        coverage=float(metrics.get("coverage", 0.0)),
        evaluated=int(metrics.get("evaluated", 0)),
        metrics_path=metrics_path,
        metrics=metrics,
    )


def _iter_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: SweepRunContext,
) -> Sequence[SweepTask]:
    """Yield sweep tasks with deterministic ordering."""

    base_cli_tuple = tuple(context.base_cli)
    extra_cli_tuple = tuple(context.extra_cli)
    # Restrict training to the same study: do not include alternates.
    tasks: List[SweepTask] = []
    task_index = 0
    for config in configs:
        for study in studies:
            run_root = context.sweep_dir / study.issue_slug / study.study_slug / config.label()
            metrics_path = run_root / study.evaluation_slug / "metrics.json"
            train_studies: Tuple[str, ...] = tuple()
            tasks.append(
                SweepTask(
                    index=task_index,
                    study=study,
                    config=config,
                    base_cli=base_cli_tuple,
                    extra_cli=extra_cli_tuple,
                    run_root=run_root,
                    tree_method=context.tree_method,
                    metrics_path=metrics_path,
                    train_participant_studies=train_studies,
                )
            )
            task_index += 1
    return tasks


def _prepare_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: SweepRunContext,
    reuse_existing: bool,
) -> Tuple[List[SweepTask], List[SweepOutcome]]:
    """
    Determine pending sweep tasks and return cached outcomes.

    :param studies: Participant studies slated for evaluation.
    :type studies: Sequence[StudySpec]
    :param configs: Hyper-parameter configurations to explore.
    :type configs: Sequence[SweepConfig]
    :param context: Shared sweep execution context.
    :type context: SweepRunContext
    :param reuse_existing: Flag controlling whether cached artefacts should be reused.
    :type reuse_existing: bool
    :returns: Tuple containing pending tasks and cached outcomes.
    :rtype: Tuple[List[SweepTask], List[SweepOutcome]]
    """

    pending: List[SweepTask] = []
    cached: List[SweepOutcome] = []
    for task in _iter_sweep_tasks(studies=studies, configs=configs, context=context):
        metrics_path = task.metrics_path
        if reuse_existing and metrics_path.exists():
            LOGGER.info(
                "[SWEEP][SKIP] issue=%s study=%s config=%s (cached).",
                task.study.issue,
                task.study.key,
                task.config.label(),
            )
            cached_metrics = _load_metrics(metrics_path)
            cached.append(_sweep_outcome_from_metrics(task, cached_metrics, metrics_path))
            continue
        pending.append(task)
    return pending, cached


def _merge_sweep_outcomes(
    cached: Sequence[SweepOutcome],
    executed: Sequence[SweepOutcome],
) -> List[SweepOutcome]:
    """
    Combine cached and freshly executed sweep outcomes preserving order.

    :param cached: Previously cached sweep outcomes.
    :type cached: Sequence[SweepOutcome]
    :param executed: Outcomes produced by the current execution.
    :type executed: Sequence[SweepOutcome]
    :returns: Ordered list of merged outcomes.
    :rtype: List[SweepOutcome]
    """

    return merge_indexed_outcomes(
        cached,
        executed,
        logger=LOGGER,
        message="Duplicate sweep outcome for index=%d; replacing cached result.",
        args_factory=lambda _existing, incoming: (incoming.order_index,),
    )


def _execute_sweep_tasks(
    tasks: Sequence[SweepTask],
    *,
    jobs: int,
) -> List[SweepOutcome]:
    """
    Execute sweep tasks, optionally in parallel.

    :param tasks: Sweep tasks to execute.
    :type tasks: Sequence[SweepTask]
    :param jobs: Maximum number of parallel workers.
    :type jobs: int
    :returns: Ordered list of sweep outcomes.
    :rtype: List[SweepOutcome]
    """

    return execute_indexed_tasks(
        tasks,
        _execute_sweep_task,
        jobs=jobs,
        logger=LOGGER,
        label="sweep",
    )


def _emit_sweep_plan(tasks: Sequence[SweepTask]) -> None:
    """
    Print a concise sweep plan listing.

    :param tasks: Sweep tasks to display.
    :type tasks: Sequence[SweepTask]
    """

    print(f"TOTAL_TASKS={len(tasks)}")
    if not tasks:
        return
    print("INDEX\tSTUDY\tISSUE\tVECTORIZER\tLABEL")
    for display_index, task in enumerate(tasks):
        print(
            f"{display_index}\t{task.study.key}\t{task.study.issue}\t"
            f"{task.config.text_vectorizer}\t{task.config.label()}"
        )


def _format_sweep_task_descriptor(task: SweepTask) -> str:
    """
    Return a short descriptor for a sweep task.

    :param task: Sweep task to describe.
    :type task: SweepTask
    :returns: Concise descriptor combining study and configuration.
    :rtype: str
    """

    return f"{task.study.key}:{task.study.issue}:{task.config.label()}"


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
    :rtype: OpinionSweepOutcome
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
    """Construct vectoriser settings for a single opinion sweep."""

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


def _iter_opinion_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: OpinionSweepRunContext,
) -> Sequence[OpinionSweepTask]:
    """Yield opinion sweep tasks in a deterministic order."""

    dataset = str(context.dataset) if context.dataset else None
    cache_dir = str(context.cache_dir) if context.cache_dir else None
    extra_fields = tuple(context.extra_fields)
    tasks: List[OpinionSweepTask] = []
    task_index = 0
    for config in configs:
        for study in studies:
            run_root = context.sweep_dir / study.issue_slug / study.study_slug / config.label()
            vectorizer = _build_opinion_vectorizer_config(
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
            request_args: Dict[str, object] = {
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
            tasks.append(
                OpinionSweepTask(
                    index=task_index,
                    study=study,
                    config=config,
                    feature_space=feature_space,
                    request_args=request_args,
                    metrics_path=metrics_path,
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
            metrics = _load_metrics(metrics_path)
            cached.append(_opinion_sweep_outcome_from_metrics(task, metrics, metrics_path))
            continue
        pending.append(task)
    return pending, cached


def _merge_opinion_sweep_outcomes(
    cached: Sequence[OpinionSweepOutcome],
    executed: Sequence[OpinionSweepOutcome],
) -> List[OpinionSweepOutcome]:
    """Merge cached and fresh opinion sweep outcomes preserving order."""

    return merge_indexed_outcomes(
        cached,
        executed,
        logger=LOGGER,
        message="Duplicate opinion sweep outcome for index=%d; replacing cached result.",
        args_factory=lambda _existing, incoming: (incoming.order_index,),
    )


def _execute_opinion_sweep_task(task: OpinionSweepTask) -> OpinionSweepOutcome:
    """Execute a single opinion sweep task and return the resulting metrics."""

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
    metrics = _load_metrics_with_log(
        task.metrics_path,
        task.study,
        log_level=logging.WARNING,
        message=(
            "[OPINION][SWEEP][MISS] issue=%s study=%s expected metrics at %s; "
            "recording placeholder outcome."
        ),
    )
    if metrics is None:
        metrics = {
            "model": "xgb_opinion",
            "feature_space": task.feature_space,
            "study": task.study.key,
            "issue": task.study.issue,
            "split": "validation",
            "metrics": {},
            "baseline": {},
            "skipped": True,
        }
        # Persist a small breadcrumb so reuse/caching can detect the skip later.
        task.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(task.metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
        except OSError:  # pragma: no cover - best-effort breadcrumb
            LOGGER.debug(
                "[OPINION][SWEEP][MISS] Unable to write placeholder metrics at %s",
                task.metrics_path,
            )
    return _opinion_sweep_outcome_from_metrics(task, metrics, task.metrics_path)


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


def _emit_combined_sweep_plan(
    *,
    slate_tasks: Sequence[SweepTask],
    opinion_tasks: Sequence[OpinionSweepTask],
) -> None:
    """Print a combined sweep plan covering next-video and opinion tasks."""

    total = len(slate_tasks) + len(opinion_tasks)
    sections: List[str] = []
    if slate_tasks:
        sections.append("next_video")
    if opinion_tasks:
        sections.append("opinion")
    if sections:
        print(",".join(sections))
    print(f"TOTAL_TASKS={total}")
    if slate_tasks:
        print("### NEXT_VIDEO")
        print("INDEX\tSTUDY\tISSUE\tVECTORIZER\tLABEL")
        for display_index, task in enumerate(slate_tasks):
            print(
                f"{display_index}\t{task.study.key}\t{task.study.issue}\t"
                f"{task.config.text_vectorizer}\t{task.config.label()}"
            )
    if opinion_tasks:
        print("### OPINION")
        print("INDEX\tSTUDY\tISSUE\tVECTORIZER\tLABEL")
        for display_index, task in enumerate(opinion_tasks):
            vectorizer = getattr(
                task,
                "feature_space",
                getattr(task.config, "text_vectorizer", "opinion"),
            )
            print(
                f"{display_index}\t{task.study.key}\t{task.study.issue}\t"
                f"{vectorizer}\t{task.config.label()}"
            )


def _format_opinion_sweep_task_descriptor(task: OpinionSweepTask) -> str:
    """
    Return a short descriptor for an opinion sweep task.

    :param task: Opinion sweep task to describe.
    :type task: OpinionSweepTask
    :returns: Concise descriptor combining study and configuration.
    :rtype: str
    """

    return f"{task.study.key}:{task.study.issue}:{task.feature_space}:{task.config.label()}"


def _gpu_tree_method_supported() -> bool:
    """
    Determine whether the installed XGBoost build supports GPU boosters.

    :returns: ``True`` if GPU boosters appear to be available.
    :rtype: bool
    """

    if xgboost is None:
        return False
    core = xgboost.core  # type: ignore[attr-defined]

    # Prefer the helper exposed in newer releases.
    maybe_has_cuda = getattr(core, "_has_cuda_support", None)
    if callable(maybe_has_cuda):
        has_cuda_callable: Callable[[], object] = maybe_has_cuda
        try:
            return bool(has_cuda_callable())  # pylint: disable=not-callable
        except (TypeError, ValueError, RuntimeError, AttributeError):
            LOGGER.debug("Failed to query XGBoost CUDA support.", exc_info=True)
            return False

    # Fallback: inspect the shared library for a device-specific symbol.
    lib = getattr(core, "_LIB", None)
    return hasattr(lib, "XGBoosterPredictFromDeviceDMatrix")


def _load_final_metrics_from_disk(
    *,
    next_video_dir: Path,
    studies: Sequence[StudySpec],
) -> Dict[str, Mapping[str, object]]:
    """
    Load persisted final evaluation metrics per study.

    :param next_video_dir: Directory containing final next-video artefacts.
    :type next_video_dir: Path
    :param studies: Participant studies to load.
    :type studies: Sequence[StudySpec]
    :returns: Mapping from study key to metrics payloads.
    :rtype: Dict[str, Mapping[str, object]]
    """

    metrics_by_study: Dict[str, Mapping[str, object]] = {}
    for spec in studies:
        metrics_path = next_video_dir / spec.evaluation_slug / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = dict(_load_metrics(metrics_path))
        _inject_study_metadata(metrics, spec)
        metrics_by_study[spec.key] = metrics
    return metrics_by_study


def _load_loso_metrics_from_disk(
    *,
    next_video_dir: Path,
    studies: Sequence[StudySpec],
) -> Dict[str, Mapping[str, object]]:
    """
    Load leave-one-study-out evaluation metrics for each study.

    :param next_video_dir: Directory containing next-video evaluation artefacts.
    :type next_video_dir: Path
    :param studies: Participant studies to load.
    :type studies: Sequence[StudySpec]
    :returns: Mapping from study key to LOSO metrics payloads.
    :rtype: Dict[str, Mapping[str, object]]
    """

    metrics_by_study: Dict[str, Mapping[str, object]] = {}
    loso_root = next_video_dir / "loso"
    if not loso_root.exists():
        return metrics_by_study
    for spec in studies:
        metrics_path = loso_root / spec.evaluation_slug / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _load_metrics_with_log(
            metrics_path,
            spec,
            log_level=logging.DEBUG,
            message="[LOSO][MISS] issue=%s study=%s expected metrics at %s but none found.",
        )
        if metrics is None:
            continue
        _inject_study_metadata(metrics, spec)
        metrics_by_study[spec.key] = metrics
    return metrics_by_study


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
    :type studies: Sequence[StudySpec]
    :returns: Mapping from study key to opinion metrics payload.
    :rtype: Dict[str, Dict[str, object]]
    """

    results: Dict[str, Dict[str, object]] = {}
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
        metrics = dict(_load_metrics(metrics_path))
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
    :type studies: Sequence[StudySpec]
    :returns: Mapping from study key to opinion-from-next metrics payload.
    :rtype: Dict[str, Dict[str, object]]
    """

    results: Dict[str, Dict[str, object]] = {}
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
            metrics = dict(_load_metrics(metrics_path))
        except FileNotFoundError:
            LOGGER.debug(
                "[OPINION-NEXT][MISS] study=%s expected metrics at %s but none found.",
                spec.key,
                metrics_path,
            )
            continue
        results[spec.key] = metrics
    return results


def _run_sweeps(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: SweepRunContext,
) -> List[SweepOutcome]:
    """
    Execute hyper-parameter sweeps and collect outcome metadata.

    :param studies: Participant studies slated for evaluation.
    :type studies: Sequence[StudySpec]
    :param configs: Hyper-parameter configurations to explore.
    :type configs: Sequence[SweepConfig]
    :param context: Shared sweep execution context.
    :type context: SweepRunContext
    :returns: Ordered list of combined cached and executed outcomes.
    :rtype: List[SweepOutcome]
    """

    pending_tasks, cached_outcomes = _prepare_sweep_tasks(
        studies=studies,
        configs=configs,
        context=context,
        reuse_existing=False,
    )
    executed_outcomes = _execute_sweep_tasks(pending_tasks, jobs=context.jobs)
    return _merge_sweep_outcomes(cached_outcomes, executed_outcomes)


def _execute_sweep_task(task: SweepTask) -> SweepOutcome:
    """
    Execute a single XGBoost sweep task and return the resulting metrics.

    :param task: Sweep task to execute.
    :type task: SweepTask
    :returns: Sweep outcome populated with metrics.
    :rtype: SweepOutcome
    """

    run_root = task.run_root
    run_root.mkdir(parents=True, exist_ok=True)

    cli_args: List[str] = list(task.base_cli)
    cli_args.extend(task.config.cli_args(task.tree_method))
    cli_args.extend(["--issues", task.study.issue])
    cli_args.extend(["--participant_studies", task.study.key])
    cli_args.extend(["--out_dir", str(run_root)])
    cli_args.extend(task.extra_cli)
    if task.train_participant_studies:
        cli_args.extend([
            "--train_participant_studies",
            ",".join(task.train_participant_studies),
        ])
    else:
        # Within-study training: do not include alternate studies.
        LOGGER.info(
            "[SWEEP] issue=%s study=%s training restricted to within-study only.",
            task.study.issue,
            task.study.key,
        )

    evaluation_dir = task.metrics_path.parent
    has_existing_outputs = evaluation_dir.exists()
    missing_metrics = not task.metrics_path.exists()
    if has_existing_outputs and "--overwrite" not in cli_args:
        if missing_metrics:
            LOGGER.warning(
                "[SWEEP][RECOVER] issue=%s study=%s config=%s detected partial outputs at %s; "
                "automatically enabling overwrite for rerun.",
                task.study.issue,
                task.study.key,
                task.config.label(),
                evaluation_dir,
            )
        else:
            LOGGER.info(
                "[SWEEP][OVERWRITE] issue=%s study=%s config=%s existing outputs at %s; "
                "enabling overwrite to refresh metrics.",
                task.study.issue,
                task.study.key,
                task.config.label(),
                evaluation_dir,
            )
        cli_args.append("--overwrite")

    LOGGER.info(
        "[SWEEP] issue=%s study=%s config=%s",
        task.study.issue,
        task.study.key,
        task.config.label(),
    )
    _run_xgb_cli(cli_args)

    # Handle runs that were skipped by the evaluator (e.g., no train/eval rows)
    # by logging and producing a placeholder outcome instead of raising.
    metrics = _load_metrics_with_log(
        task.metrics_path,
        task.study,
        log_level=logging.WARNING,
        message=(
            "[SWEEP][MISS] issue=%s study=%s missing metrics at %s; "
            "recording placeholder outcome."
        ),
    )
    if metrics is None:
        from common.pipeline_utils import make_placeholder_metrics
        metrics = make_placeholder_metrics(
            task.study.evaluation_slug,
            [task.study.key],
            extra_fields=[],
        )
        # Persist a small breadcrumb so reuse/caching can detect the skip later.
        task.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(task.metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
        except OSError:  # pragma: no cover - best-effort breadcrumb
            LOGGER.debug(
                "[SWEEP][MISS] Unable to write placeholder metrics at %s",
                task.metrics_path,
            )
    return SweepOutcome(
        order_index=task.index,
        study=task.study,
        config=task.config,
        accuracy=float(metrics.get("accuracy", 0.0)),
        coverage=float(metrics.get("coverage", 0.0)),
        evaluated=int(metrics.get("evaluated", 0)),
        metrics_path=task.metrics_path,
        metrics=metrics,
    )


def _select_best_configs(outcomes: Sequence[SweepOutcome]) -> Dict[str, StudySelection]:
    """
    Pick the best configuration per study using accuracy, coverage, and support.

    :param outcomes: Sweep outcomes covering all studies and configurations.
    :type outcomes: Sequence[SweepOutcome]
    :returns: Mapping from study key to the chosen configuration.
    :rtype: Dict[str, StudySelection]
    """

    selections: Dict[str, StudySelection] = {}

    for outcome in outcomes:
        current = selections.get(outcome.study.key)
        if current is None:
            selections[outcome.study.key] = StudySelection(study=outcome.study, outcome=outcome)
            continue
        incumbent = current.outcome
        if outcome.accuracy > incumbent.accuracy + 1e-9:
            selections[outcome.study.key] = StudySelection(study=outcome.study, outcome=outcome)
            continue
        if incumbent.accuracy - outcome.accuracy <= 1e-9:
            if outcome.coverage > incumbent.coverage + 1e-9:
                selections[outcome.study.key] = StudySelection(
                    study=outcome.study,
                    outcome=outcome,
                )
            elif (
                abs(outcome.coverage - incumbent.coverage) <= 1e-9
                and outcome.evaluated > incumbent.evaluated
            ):
                selections[outcome.study.key] = StudySelection(
                    study=outcome.study,
                    outcome=outcome,
                )
    return selections


def _select_best_opinion_configs(
    outcomes: Sequence[OpinionSweepOutcome],
) -> Dict[str, OpinionStudySelection]:
    """
    Pick the best opinion configuration per study prioritising MAE, RMSE, then R².

    :param outcomes: Opinion sweep outcomes spanning studies and configurations.
    :type outcomes: Sequence[OpinionSweepOutcome]
    :returns: Mapping from study key to the chosen opinion configuration.
    :rtype: Dict[str, OpinionStudySelection]
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
    "DEFAULT_OPINION_FEATURE_SPACE",
    "_run_xgb_cli",
    "_load_metrics",
    "_sweep_outcome_from_metrics",
    "_opinion_sweep_outcome_from_metrics",
    "_prepare_sweep_tasks",
    "_prepare_opinion_sweep_tasks",
    "_merge_sweep_outcomes",
    "_merge_opinion_sweep_outcomes",
    "_execute_sweep_tasks",
    "_execute_opinion_sweep_tasks",
    "execute_opinion_sweep_tasks",
    "_execute_sweep_task",
    "_execute_opinion_sweep_task",
    "_emit_sweep_plan",
    "_emit_combined_sweep_plan",
    "_format_sweep_task_descriptor",
    "_format_opinion_sweep_task_descriptor",
    "_gpu_tree_method_supported",
    "_load_final_metrics_from_disk",
    "_load_opinion_metrics_from_disk",
    "_run_sweeps",
    "_select_best_configs",
    "_select_best_opinion_configs",
]
