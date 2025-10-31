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

"""Utility helpers shared by the Grail Simulation KNN pipeline stages.

Collects small formatting utilities, filesystem helpers, and conversions
from raw metric payloads into structured summaries used by the reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from common.pipeline.formatters import safe_float, safe_int

from .context import MetricSummary, OpinionSummary, StudySpec, OpinionSummaryInputs
from common.opinion import OpinionCalibrationMetrics

TaskT = TypeVar("TaskT")
OutcomeT = TypeVar("OutcomeT")


@dataclass
class TaskCachePartition(Generic[TaskT, OutcomeT]):
    """Accumulators representing pending and cached tasks."""

    pending: List[TaskT]
    cached: List[OutcomeT]


@dataclass(frozen=True)
class TaskCacheStrategy(Generic[TaskT, OutcomeT]):
    """Bundle describing how cached tasks are discovered."""

    load_cached: Callable[[TaskT], Optional[OutcomeT]]
    cache_path: Callable[[TaskT], Path] | None = None


def handle_cached_task(
    task: TaskT,
    *,
    reuse_existing: bool,
    cache_path: Path,
    load_cached: Callable[[TaskT], Optional[OutcomeT]],
    partition: TaskCachePartition[TaskT, OutcomeT],
) -> bool:
    """
    Append ``task`` to the ``partition`` unless a cached outcome can be reused.

    :returns: ``True`` when a cached outcome was appended and the caller should skip execution.
    """

    if reuse_existing and cache_path.exists():
        cached_result = load_cached(task)
        if cached_result is not None:
            partition.cached.append(cached_result)
            return True
    partition.pending.append(task)
    return False


def partition_cached_tasks(
    tasks: Iterable[TaskT],
    *,
    reuse_existing: bool,
    cache_path: Callable[[TaskT], Path],
    load_cached: Callable[[TaskT], Optional[OutcomeT]],
) -> Tuple[List[TaskT], List[OutcomeT]]:
    """
    Split ``tasks`` into pending work and cached results.

    :returns: Tuple containing pending task list and cached outcomes respectively.
    """

    partition = TaskCachePartition(pending=[], cached=[])
    for task in tasks:
        was_cached = handle_cached_task(
            task,
            reuse_existing=reuse_existing,
            cache_path=cache_path(task),
            load_cached=load_cached,
            partition=partition,
        )
        if was_cached:
            continue
    return partition.pending, partition.cached


def prepare_task_grid(
    configs: Sequence[Any],
    studies: Sequence[Any],
    *,
    reuse_existing: bool,
    build_task: Callable[[int, Any, Any], TaskT],
    cache: TaskCacheStrategy[TaskT, OutcomeT],
) -> Tuple[List[TaskT], List[OutcomeT]]:
    """
    Construct sweep task grid and partition cached outcomes.

    :param configs: Hyper-parameter configurations included in the sweep.
    :type configs: Sequence[Any]
    :param studies: Study specifications evaluated by the sweep.
    :type studies: Sequence[Any]
    :param reuse_existing: Whether cached metrics should be reused.
    :type reuse_existing: bool
    :param build_task: Callable that constructs a task for the given index, config, and study.
    :type build_task: Callable[[int, Any, Any], TaskT]
    :param cache: Strategy describing how cached outcomes are located and loaded.
    :type cache: TaskCacheStrategy[TaskT, OutcomeT]
    :returns: Pending tasks and cached outcomes separated according to ``reuse_existing``.
    :rtype: Tuple[List[TaskT], List[OutcomeT]]
    """

    tasks = [
        build_task(task_index, config, study)
        for task_index, (config, study) in enumerate(product(configs, studies))
    ]
    cache_path_fn = cache.cache_path
    if cache_path_fn is None:

        def default_cache_path(task: TaskT) -> Path:
            """
            Resolve the filesystem location backing a cached task outcome.

            :param task: Sweep task that provides a ``metrics_path`` attribute.
            :type task: TaskT
            :returns: ``Path`` pointing to the cached metrics artefact.
            :rtype: Path
            """
            try:
                path = getattr(task, "metrics_path")
            except AttributeError as exc:  # pragma: no cover - defensive only
                raise AttributeError(
                    "Task is missing 'metrics_path' required for caching."
                ) from exc
            return Path(path)
        cache_path_fn = default_cache_path

    return partition_cached_tasks(
        tasks,
        reuse_existing=reuse_existing,
        cache_path=cache_path_fn,
        load_cached=cache.load_cached,
    )


def ensure_dir(path: Path) -> Path:
    """
    Ensure the given directory exists and return it.

    :param path: Directory path that should be created if missing.
    :type path: Path
    :returns: The original ``Path`` instance for convenient chaining.
    :rtype: Path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

def snake_to_title(value: str) -> str:
    """
    Convert a ``snake_case`` string into Title Case.

    :param value: String containing underscore-separated tokens.
    :type value: str
    :returns: Title-cased string with spaces instead of underscores.
    :rtype: str
    """
    return value.replace("_", " ").title()

def format_float(value: float) -> str:
    """
    Format a floating-point metric with three decimal places.

    :param value: Numeric metric to render.
    :type value: float
    :returns: Formatted string (``{value:.3f}``).
    :rtype: str
    """
    return f"{value:.3f}"

def format_optional_float(value: Optional[float]) -> str:
    """
    Format an optional floating-point metric.

    :param value: Metric value or ``None`` when unavailable.
    :type value: Optional[float]
    :returns: Formatted float or an em dash when ``None``.
    :rtype: str
    """
    return format_float(value) if value is not None else "—"

def format_delta(delta: Optional[float]) -> str:
    """
    Format a signed improvement metric.

    :param delta: Change relative to a baseline.
    :type delta: Optional[float]
    :returns: Signed string (``+0.000`` style) or an em dash when ``None``.
    :rtype: str
    """
    return f"{delta:+.3f}" if delta is not None else "—"

def format_count(value: Optional[int]) -> str:
    """
    Format integer counts with thousands separators.

    :param value: Count to render.
    :type value: Optional[int]
    :returns: Formatted count or an em dash when ``None``.
    :rtype: str
    """
    if value is None:
        return "—"
    return f"{value:,}"

def format_k(value: Optional[int]) -> str:
    """
    Format the selected ``k`` hyper-parameter.

    :param value: Neighbourhood size under consideration.
    :type value: Optional[int]
    :returns: String representation of ``k`` or an em dash when unset.
    :rtype: str
    """
    if value is None or value <= 0:
        return "—"
    return str(value)

def format_uncertainty_details(uncertainty: Mapping[str, object]) -> str:
    """
    Format auxiliary uncertainty metadata for reporting.

    :param uncertainty: Mapping containing extra uncertainty fields.
    :type uncertainty: Mapping[str, object]
    :returns: Parenthesised detail string or an empty string.
    :rtype: str
    """
    if not isinstance(uncertainty, Mapping):
        return ""
    detail_bits: List[str] = []
    for key in ("n_bootstrap", "n_groups", "n_rows", "seed"):
        value = uncertainty.get(key)
        if value is None:
            continue
        detail_bits.append(f"{key}={value}")
    return f" ({', '.join(detail_bits)})" if detail_bits else ""

def parse_ci(ci_value: object) -> Optional[Tuple[float, float]]:
    """
    Convert confidence-interval payloads into numeric tuples.

    :param ci_value: Mapping or sequence describing a confidence interval.
    :type ci_value: object
    :returns: Tuple containing ``(low, high)`` bounds when available.
    :rtype: Optional[Tuple[float, float]]
    """
    if isinstance(ci_value, Mapping):
        low = safe_float(ci_value.get("low"))
        high = safe_float(ci_value.get("high"))
        if low is not None and high is not None:
            return (low, high)
        return None
    if isinstance(ci_value, (tuple, list)) and len(ci_value) == 2:
        low = safe_float(ci_value[0])
        high = safe_float(ci_value[1])
        if low is not None and high is not None:
            return (low, high)
    return None

def extract_metric_summary(data: Mapping[str, object]) -> MetricSummary:
    """
    Collect reusable next-video metric fields from ``data``.

    :param data: Raw metrics dictionary emitted by the evaluation stage.
    :type data: Mapping[str, object]
    :returns: Normalised metric summary for downstream reports.
    :rtype: MetricSummary
    """
    accuracy = safe_float(data.get("accuracy_overall"))
    accuracy_all_rows = safe_float(data.get("accuracy_overall_all_rows"))
    best_k = safe_int(data.get("best_k"))
    n_total = safe_int(data.get("n_total"))
    n_eligible = safe_int(data.get("n_eligible"))
    accuracy_ci = parse_ci(
        data.get("accuracy_ci_95") or data.get("accuracy_uncertainty", {}).get("ci95")
    )

    baseline_ci = parse_ci(
        data.get("baseline_ci_95") or data.get("baseline_uncertainty", {}).get("ci95")
    )
    baseline_data = data.get("baseline_most_frequent_gold_index", {})
    baseline = None
    if isinstance(baseline_data, Mapping):
        baseline = safe_float(baseline_data.get("accuracy"))

    random_baseline = safe_float(data.get("random_baseline_expected_accuracy"))

    return MetricSummary(
        inputs={
            "accuracy": accuracy,
            "accuracy_ci": accuracy_ci,
            "baseline": baseline,
            "baseline_ci": baseline_ci,
            "random_baseline": random_baseline,
            "best_k": best_k,
            "n_total": n_total,
            "n_eligible": n_eligible,
            "accuracy_all_rows": accuracy_all_rows,
        }
    )

def extract_opinion_summary(data: Mapping[str, object]) -> OpinionSummary:
    """
    Collect opinion regression metrics into a normalised structure.

    :param data: Raw metrics dictionary emitted by the opinion pipeline.
    :type data: Mapping[str, object]
    :returns: Normalised opinion summary for reporting.
    :rtype: ~knn.pipeline.context.OpinionSummary
    """
    best_metrics = data.get("best_metrics", {})
    baseline_metrics = data.get("baseline", {})
    if not isinstance(best_metrics, Mapping):
        best_metrics = {}
    if not isinstance(baseline_metrics, Mapping):
        baseline_metrics = {}

    def metric(source: Mapping[str, Any], key: str) -> Optional[float]:
        """
        Extract a floating-point metric from the given mapping.

        :param source: Metric container to query.
        :type source: Mapping[str, Any]
        :param key: Metric key expected inside ``source``.
        :type key: str
        :returns: Normalised metric value or ``None`` when unavailable.
        :rtype: Optional[float]
        """
        return safe_float(source.get(key))

    def difference(lhs: Optional[float], rhs: Optional[float]) -> Optional[float]:
        """
        Compute the difference between two optional floating-point metrics.

        :param lhs: Left-hand operand.
        :type lhs: Optional[float]
        :param rhs: Right-hand operand.
        :type rhs: Optional[float]
        :returns: ``lhs - rhs`` when both operands are present; otherwise ``None``.
        :rtype: Optional[float]
        """
        if lhs is None or rhs is None:
            return None
        return lhs - rhs

    participants = safe_int(data.get("n_participants"))
    eligible = next(
        (
            value
            for value in (
                safe_int(best_metrics.get("eligible")),
                safe_int(data.get("eligible")),
                participants,
            )
            if value is not None
        ),
        None,
    )

    accuracy = metric(best_metrics, "direction_accuracy")
    baseline_accuracy = metric(baseline_metrics, "direction_accuracy")
    mae_value = metric(best_metrics, "mae_after")
    baseline_mae_value = metric(baseline_metrics, "mae_using_before")

    mae_delta = (
        baseline_mae_value - mae_value
        if mae_value is not None and baseline_mae_value is not None
        else None
    )

    calibration = OpinionCalibrationMetrics(
        baseline_accuracy=baseline_accuracy,
        accuracy_delta=difference(accuracy, baseline_accuracy),
        calibration_slope=metric(best_metrics, "calibration_slope"),
        baseline_calibration_slope=metric(
            baseline_metrics, "calibration_slope_change_zero"
        ),
        calibration_intercept=metric(best_metrics, "calibration_intercept"),
        baseline_calibration_intercept=metric(
            baseline_metrics, "calibration_intercept_change_zero"
        ),
        calibration_ece=metric(best_metrics, "calibration_ece"),
        baseline_calibration_ece=metric(
            baseline_metrics, "calibration_ece_change_zero"
        ),
        kl_divergence_change=metric(best_metrics, "kl_divergence_change"),
        baseline_kl_divergence_change=metric(
            baseline_metrics, "kl_divergence_change_zero"
        ),
        participants=participants,
        eligible=eligible,
        dataset=str(data.get("dataset")) if data.get("dataset") else None,
        split=str(data.get("split")) if data.get("split") else None,
    )

    regression = OpinionSummary.Regression(
        primary=OpinionSummary.PrimaryRegression(
            mae=mae_value,
            rmse=metric(best_metrics, "rmse_after"),
            r2_score=metric(best_metrics, "r2_after"),
        ),
        change=OpinionSummary.ChangeStats(
            mae_change=metric(best_metrics, "mae_change"),
            rmse_change=metric(best_metrics, "rmse_change"),
            baseline_rmse_change=metric(baseline_metrics, "rmse_change_zero"),
        ),
        baseline=OpinionSummary.BaselineStats(
            baseline_mae=baseline_mae_value,
            mae_delta=mae_delta,
        ),
        best_k=safe_int(data.get("best_k")),
    )

    inputs = OpinionSummaryInputs(
        calibration=calibration,
        regression=regression,
        accuracy=accuracy,
    )
    return OpinionSummary(inputs=inputs)


@dataclass(frozen=True)
class SelectionValidationOptions:
    """Reusable configuration for selection validation helpers."""

    allow_incomplete: bool
    logger: Any
    missing_descriptor: str
    empty_descriptor: str
    require_selected: bool


def ensure_feature_selections(
    selections: Mapping[str, Mapping[str, Any]],
    expected_keys: Sequence[str],
    *,
    options: SelectionValidationOptions,
) -> None:
    """
    Validate that ``selections`` covers all ``expected_keys`` for each feature space.

    Emits a warning when ``allow_incomplete`` is set and some studies are missing.
    Raises ``RuntimeError`` when data is missing and incomplete results are not allowed.
    If ``require_selected`` is true and no selections were recorded, a ``RuntimeError``
    is raised to signal the failure.
    """

    for feature_space, per_feature in selections.items():
        missing = [key for key in expected_keys if key not in per_feature]
        if not missing:
            continue
        message = (
            f"Missing {options.missing_descriptor} for feature={feature_space}: "
            f"{', '.join(missing)}"
        )
        if options.allow_incomplete:
            options.logger.warning(
                "%s. Continuing because allow-incomplete mode is enabled.",
                message,
            )
        else:
            raise RuntimeError(message)

    if not selections and options.require_selected:
        raise RuntimeError(
            f"Failed to select a best configuration for any {options.empty_descriptor}."
        )


def ensure_selection_coverage(
    selections: Mapping[str, Mapping[str, Any]],
    studies: Sequence[StudySpec],
    *,
    options: SelectionValidationOptions,
) -> None:
    """
    Validate selection coverage across feature spaces for the given studies.

    Wrapper around :func:`ensure_feature_selections` that derives expected study
    keys from ``studies`` to avoid duplicating boilerplate.

    :param selections: Mapping from feature space to per-study selections.
    :param studies: Study specifications providing the expected keys.
    :param options: Validation options controlling logging and error behaviour.
    :returns: ``None`` (raises on validation failure when configured).
    """

    expected_keys = [study.key for study in studies]
    ensure_feature_selections(
        selections,
        expected_keys,
        options=options,
    )


def ensure_sweep_selection_coverage(
    selections: Mapping[str, Mapping[str, Any]],
    studies: Sequence[StudySpec],
    *,
    allow_incomplete: bool,
    logger: Any,
    require_selected: bool = True,
) -> None:
    """
    Validate coverage for the core sweep selections produced by the pipeline.

    :param selections: Mapping from feature space to per-study selections.
    :param studies: Study specifications providing expected keys.
    :param allow_incomplete: When ``True``, emit warnings instead of raising.
    :param logger: Logger used for warnings/info messages.
    :param require_selected: When ``True``, require at least one selection globally.
    :returns: ``None``.
    """

    ensure_selection_coverage(
        selections,
        studies,
        options=SelectionValidationOptions(
            allow_incomplete=allow_incomplete,
            logger=logger,
            missing_descriptor="sweep selections",
            empty_descriptor="feature space",
            require_selected=require_selected,
        ),
    )


def ensure_opinion_selection_coverage(
    selections: Mapping[str, Mapping[str, Any]],
    studies: Sequence[StudySpec],
    *,
    allow_incomplete: bool,
    logger: Any,
    require_selected: bool,
) -> None:
    """
    Validate coverage for opinion sweep selections with custom descriptors.

    :param selections: Mapping from feature space to per-study opinion selections.
    :param studies: Study specifications providing expected keys.
    :param allow_incomplete: When ``True``, emit warnings instead of raising.
    :param logger: Logger used for warnings/info messages.
    :param require_selected: When ``True``, require at least one selection globally.
    :returns: ``None``.
    """

    ensure_selection_coverage(
        selections,
        studies,
        options=SelectionValidationOptions(
            allow_incomplete=allow_incomplete,
            logger=logger,
            missing_descriptor="opinion sweep selections",
            empty_descriptor="opinion feature space",
            require_selected=require_selected,
        ),
    )

__all__ = [
    "ensure_dir",
    "ensure_feature_selections",
    "ensure_selection_coverage",
    "ensure_sweep_selection_coverage",
    "ensure_opinion_selection_coverage",
    "extract_metric_summary",
    "extract_opinion_summary",
    "format_count",
    "format_delta",
    "format_float",
    "format_k",
    "format_optional_float",
    "format_uncertainty_details",
    "handle_cached_task",
    "prepare_task_grid",
    "partition_cached_tasks",
    "parse_ci",
    "SelectionValidationOptions",
    "TaskCacheStrategy",
    "safe_float",
    "safe_int",
    "snake_to_title",
]
