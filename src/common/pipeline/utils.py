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

"""Shared utilities for merging sweep outcomes and representing selections."""

from __future__ import annotations

import logging
from operator import attrgetter
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Sequence, TypeVar, Optional, Tuple


T = TypeVar("T")
StudyT = TypeVar("StudyT")
OutcomeT = TypeVar("OutcomeT")


def merge_ordered(
    cached: Sequence[T],
    executed: Sequence[T],
    *,
    order_key: Callable[[T], int],
    on_replace: Callable[[T, T], None] | None = None,
) -> List[T]:
    """Merge cached and freshly executed results while preserving order indices.

    :param cached: Previously materialised results (e.g., metrics read from disk).
    :type cached: Sequence[T]
    :param executed: Newly computed results from the current run.
    :type executed: Sequence[T]
    :param order_key: Callable returning a deterministic integer position for each result.
    :type order_key: Callable[[T], int]
    :param on_replace: Optional callback invoked when ``executed`` replaces ``cached`` at an index.
    :type on_replace: Callable[[T, T], None] | None
    :returns: Combined results ordered by ``order_key``.
    :rtype: List[T]
    """

    by_index: Dict[int, T] = {order_key(item): item for item in cached}
    for item in executed:
        index = order_key(item)
        if index in by_index and on_replace is not None:
            on_replace(by_index[index], item)
        by_index[index] = item
    return [by_index[index] for index in sorted(by_index)]


def make_duplicate_warning(
    logger: logging.Logger,
    message: str,
    *,
    args_factory: Callable[[T, T], Sequence[Any]] | None = None,
) -> Callable[[T, T], None]:
    """Return an ``on_replace`` callback that logs duplicate outcomes consistently.

    :param logger: Logger used to emit the warning.
    :type logger: logging.Logger
    :param message: Logging format string passed to :py:meth:`logging.Logger.warning`.
    :type message: str
    :param args_factory: Optional callable returning the arguments supplied alongside the
        format string. When omitted, the message is emitted without additional arguments.
    :type args_factory: Callable[[T, T], Sequence[Any]] | None
    :returns: Callback suitable for the ``on_replace`` parameter of :func:`merge_ordered`.
    :rtype: Callable[[T, T], None]
    """

    def _on_replace(existing: T, incoming: T) -> None:
        """
        Log a warning when ``incoming`` replaces ``existing`` in a result slot.

        :param existing: Cached outcome being replaced.
        :param incoming: Newly executed outcome replacing ``existing``.
        :returns: ``None``.
        """
        if args_factory is None:
            logger.warning(message)
        else:
            logger.warning(message, *args_factory(existing, incoming))

    return _on_replace


def merge_ordered_with_warning(
    cached: Sequence[T],
    executed: Sequence[T],
    *,
    order_key: Callable[[T], int],
    duplicate_warning: Callable[[T, T], None] | None = None,
) -> List[T]:
    """
    Convenience wrapper around :func:`merge_ordered` that logs duplicate replacements.

    :param cached: Previously materialised results (e.g., metrics read from disk).
    :type cached: Sequence[T]
    :param executed: Newly computed results from the current run.
    :type executed: Sequence[T]
    :param order_key: Callable returning a deterministic integer position for each result.
    :type order_key: Callable[[T], int]
    :param duplicate_warning: Optional callback invoked when cached results are replaced.
    :type duplicate_warning: Callable[[T, T], None] | None
    :returns: Combined results ordered by ``order_key``.
    :rtype: List[T]
    """

    return merge_ordered(
        cached,
        executed,
        order_key=order_key,
        on_replace=duplicate_warning,
    )


def merge_indexed_outcomes(
    cached: Sequence[T],
    executed: Sequence[T],
    *,
    logger: logging.Logger,
    message: str,
    args_factory: Callable[[T, T], Sequence[Any]] | None = None,
) -> List[T]:
    """
    Merge outcomes ordered by their ``order_index`` attribute.

    This wraps :func:`merge_ordered_with_warning` with a fixed ``order_key`` based on
    ``order_index`` to cut down on repeated boilerplate across pipeline modules.

    :param cached: Previously materialised results (e.g., metrics read from disk).
    :param executed: Newly computed results from the current run.
    :param logger: Logger used when emitting duplicate warnings.
    :param message: Logging format string passed to :py:meth:`logging.Logger.warning`.
    :param args_factory: Optional callable producing format string arguments.
    :returns: Combined results ordered by ``order_index``.
    """

    return merge_ordered_with_warning(
        cached,
        executed,
        order_key=attrgetter("order_index"),
        duplicate_warning=make_duplicate_warning(
            logger,
            message,
            args_factory=args_factory,
        ),
    )


@dataclass
class OpinionStudySelection(Generic[StudyT, OutcomeT]):
    """Pair a study descriptor with the selected outcome for that study."""
    study: StudyT
    outcome: OutcomeT

    @property
    def config(self):
        """Return the configuration object associated with the selection."""
        return self.outcome.config


__all__ = [
    "StageOverwriteContext",
    "merge_ordered",
    "make_duplicate_warning",
    "merge_ordered_with_warning",
    "merge_indexed_outcomes",
    "OpinionStudySelection",
    "compose_cli_args",
    "make_placeholder_metrics",
    "ensure_overwrite_flag",
    "ensure_stage_overwrite_flag",
    "ensure_final_stage_overwrite",
    "ensure_final_stage_overwrite_with_context",
]


@dataclass(frozen=True)
class StageOverwriteContext:
    """Configuration bundle for stage-specific overwrite logging."""

    stage: str
    labels: Sequence[Tuple[str, object]] = ()
    noun: str = "outputs"


def compose_cli_args(*segments: Sequence[str]) -> List[str]:
    """
    Return a flattened list of CLI arguments from multiple segments.

    :param segments: One or more argument sequences (or single strings) to be
        concatenated into a single list suitable for ``subprocess`` or CLIs.
    :returns: Combined argument vector preserving the order of inputs.
    """

    cli_args: List[str] = []
    for segment in segments:
        if isinstance(segment, str):
            cli_args.append(segment)
        else:
            cli_args.extend(segment)
    return cli_args


def make_placeholder_metrics(
    issue: str,
    participant_studies: Sequence[str],
    *,
    extra_fields: Optional[Sequence[str]] = None,
    skip_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a standard placeholder metrics payload for skipped runs.

    Consolidates duplicated inline dictionaries across pipeline modules when a run is
    skipped (e.g., due to missing train/eval rows). Zero-initialises known fields and
    includes optional ``extra_fields`` and ``skip_reason`` when provided.

    :param issue: Issue identifier associated with the skipped run.
    :param participant_studies: Studies associated with the skipped run.
    :param extra_fields: Optional list of additional fields noted in the payload.
    :param skip_reason: Optional human-readable explanation for the skip.
    :returns: Dictionary containing placeholder metrics and metadata.
    """

    payload: Dict[str, Any] = {
        "issue": issue,
        "participant_studies": list(participant_studies),
        "evaluated": 0,
        "correct": 0,
        "accuracy": 0.0,
        "known_candidate_hits": 0,
        "known_candidate_total": 0,
        "coverage": 0.0,
        "eligible": 0,
        "skipped": True,
    }
    if extra_fields is not None:
        payload["extra_fields"] = list(extra_fields)
    if skip_reason is not None:
        payload["skip_reason"] = skip_reason
    return payload


ArgsFactory = Callable[[Path], Sequence[Any]]


def ensure_overwrite_flag(
    cli_args: List[str],
    metrics_path: Path,
    *,
    logger: logging.Logger,
    recover_log: Optional[Tuple[str, ArgsFactory]] = None,
    overwrite_log: Optional[Tuple[str, ArgsFactory]] = None,
) -> bool:
    """
    Append ``--overwrite`` to ``cli_args`` when prior evaluation artefacts exist.

    Many pipelines share the same pattern of re-running evaluators when a metrics
    directory already exists on disk. This helper centralises the guard so that
    callers only need to provide logging metadata.

    :param cli_args: Mutable list of CLI arguments that may require ``--overwrite``.
    :param metrics_path: Expected metrics file within the evaluation directory.
    :param logger: Logger used when emitting informational or recovery messages.
    :param recover_log: Optional tuple of ``(message, args_factory)`` used when the
        metrics file is missing but other outputs are present.
    :param overwrite_log: Optional tuple of ``(message, args_factory)`` used when both
        the directory and metrics file already exist.
    :returns: ``True`` when ``--overwrite`` was appended to ``cli_args``.
    """

    evaluation_dir = metrics_path.parent
    has_existing_outputs = evaluation_dir.exists()
    missing_metrics = not metrics_path.exists()

    if not has_existing_outputs or "--overwrite" in cli_args:
        return False

    if missing_metrics and recover_log is not None:
        message, args_factory = recover_log
        logger.warning(message, *args_factory(evaluation_dir))
    elif not missing_metrics and overwrite_log is not None:
        message, args_factory = overwrite_log
        logger.info(message, *args_factory(evaluation_dir))

    cli_args.append("--overwrite")
    return True


def ensure_stage_overwrite_flag(
    cli_args: List[str],
    metrics_path: Path,
    *,
    logger: logging.Logger,
    context: StageOverwriteContext,
) -> bool:
    """
    Apply :func:`ensure_overwrite_flag` using standardised stage-aware log messages.

    :param cli_args: Mutable list of CLI arguments that may require ``--overwrite``.
    :param metrics_path: Expected metrics file within the evaluation directory.
    :param logger: Logger used when emitting informational or recovery messages.
    :param context: Bundled stage metadata (token, labels, noun).
    :returns: ``True`` when ``--overwrite`` was appended to ``cli_args``.
    """

    stage_token = context.stage.upper()
    label_fmt = " ".join(f"{label}=%s" for label, _ in context.labels)
    context_values = tuple(value for _, value in context.labels)

    if label_fmt:
        label_fmt = f"{label_fmt} "

    def _args_factory(evaluation_dir: Path) -> Tuple[object, ...]:
        if context_values:
            return (*context_values, evaluation_dir)
        return (evaluation_dir,)

    recover_message = (
        f"[{stage_token}][RECOVER] {label_fmt}detected partial {context.noun} at %s; "
        "automatically enabling overwrite for rerun."
    )
    overwrite_message = (
        f"[{stage_token}][OVERWRITE] {label_fmt}existing {context.noun} at %s; "
        "enabling overwrite to refresh metrics."
    )

    return ensure_overwrite_flag(
        cli_args,
        metrics_path,
        logger=logger,
        recover_log=(recover_message, _args_factory),
        overwrite_log=(overwrite_message, _args_factory),
    )


def ensure_final_stage_overwrite(
    cli_args: List[str],
    metrics_path: Path,
    *,
    logger: logging.Logger,
    context_labels: Sequence[Tuple[str, object]] | Tuple[()],
) -> bool:
    """
    Convenience wrapper for :func:`ensure_stage_overwrite_flag` when stage is ``FINAL``.

    :param cli_args: Mutable list of CLI arguments that may require ``--overwrite``.
    :param metrics_path: Expected metrics file within the evaluation directory.
    :param logger: Logger used when emitting informational messages.
    :param context_labels: Ordered ``(label, value)`` pairs included in log messages.
    :returns: ``True`` when ``--overwrite`` was appended to ``cli_args``.
    """

    return ensure_stage_overwrite_flag(
        cli_args,
        metrics_path,
        logger=logger,
        context=StageOverwriteContext(stage="FINAL", labels=context_labels),
    )


def ensure_final_stage_overwrite_with_context(
    cli_args: List[str],
    metrics_path: Path,
    *,
    logger: logging.Logger,
    **context_labels: object,
) -> bool:
    """Call :func:`ensure_final_stage_overwrite` with keyword context labels."""

    labels: Tuple[Tuple[str, object], ...] = tuple(context_labels.items())
    return ensure_final_stage_overwrite(
        cli_args,
        metrics_path,
        logger=logger,
        context_labels=labels,
    )
