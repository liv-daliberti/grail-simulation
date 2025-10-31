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

"""Shared utilities for XGBoost sweep orchestration."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, TypeVar

from common.pipeline.io import load_metrics_json
from common.pipeline.utils import merge_indexed_outcomes

from ...cli import build_parser as build_xgb_parser
from ...core.evaluate import run_eval
from ..context import StudySpec

try:  # pragma: no cover - optional dependency
    import xgboost  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    xgboost = None  # type: ignore[assignment]

DEFAULT_OPINION_FEATURE_SPACE = "tfidf"
LOGGER = logging.getLogger("xgb.pipeline.sweeps")
OutcomeT = TypeVar("OutcomeT")
MetricsFactory = Callable[[], Dict[str, object]]


def get_sweeps_attr(name: str) -> Any:
    """
    Retrieve an attribute from the public ``xgb.pipeline.sweeps`` namespace.

    The helper keeps monkeypatched call sites working while allowing internal
    helpers to obtain underscored functions without tripping pylint's protected
    access checks.

    :param name: Attribute name to resolve (for example ``\"_load_metrics\"``).
    :type name: str
    :returns: Attribute fetched from the ``xgb.pipeline.sweeps`` module.
    :rtype: Any
    :raises AttributeError: If the sweeps package is not loaded or the attribute
        cannot be found.
    """

    sweeps_module = sys.modules.get("xgb.pipeline.sweeps")
    if sweeps_module is None or not hasattr(sweeps_module, name):
        raise AttributeError(f"sweeps attribute {name!r} is unavailable")
    return getattr(sweeps_module, name)


def _inject_study_metadata(metrics: Dict[str, object], spec: StudySpec) -> None:
    """Ensure study metadata fields exist in a metrics payload."""

    metrics.setdefault("issue", spec.issue)
    metrics.setdefault("issue_label", spec.issue.replace("_", " ").title())
    metrics.setdefault("study", spec.key)
    metrics.setdefault("study_label", spec.label)


def _load_metrics(path: Path) -> Mapping[str, object]:
    """Load the metrics JSON emitted by a sweep or evaluation task."""

    return load_metrics_json(path)


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


def merge_sweep_outcomes(
    cached: Sequence[OutcomeT],
    executed: Sequence[OutcomeT],
    *,
    duplicate_message: str,
) -> List[OutcomeT]:
    """
    Merge cached and executed outcomes while logging duplicate replacements.

    :param cached: Outcomes retrieved from previous runs.
    :type cached: Sequence[OutcomeT]
    :param executed: Outcomes produced by the current execution.
    :type executed: Sequence[OutcomeT]
    :param duplicate_message: Warning message emitted when cached outcomes are replaced.
    :type duplicate_message: str
    :returns: Combined outcomes ordered by their ``order_index`` attribute.
    :rtype: List[OutcomeT]
    """

    return merge_indexed_outcomes(
        cached,
        executed,
        logger=LOGGER,
        message=duplicate_message,
        args_factory=lambda _existing, incoming: (incoming.order_index,),
    )


def persist_metrics_payload(
    metrics_path: Path,
    metrics: Dict[str, object],
    *,
    debug_message: str,
) -> None:
    """
    Persist metrics to disk, tolerating missing directories and IO errors.

    :param metrics_path: Destination path for the metrics payload.
    :type metrics_path: Path
    :param metrics: Metrics dictionary to serialise.
    :type metrics: Dict[str, object]
    :param debug_message: Debug log message emitted when persistence fails.
    :type debug_message: str
    :returns: ``None``.
    :rtype: None
    """

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    except OSError:  # pragma: no cover - best-effort breadcrumb
        LOGGER.debug(debug_message, metrics_path)


def ensure_metrics_with_placeholder(
    loader: Callable[[], Dict[str, object] | None],
    *,
    placeholder_factory: MetricsFactory,
    metrics_path: Path,
    debug_message: str,
) -> Dict[str, object]:
    """
    Return metrics produced by ``loader`` or materialise a placeholder payload.

    :param loader: Callable returning the metrics payload or ``None`` when unavailable.
    :type loader: Callable[[], Dict[str, object] | None]
    :param placeholder_factory: Callable producing a placeholder metrics payload.
    :type placeholder_factory: MetricsFactory
    :param metrics_path: Filesystem path where metrics should be written.
    :type metrics_path: Path
    :param debug_message: Debug log message used when persisting placeholders fails.
    :type debug_message: str
    :returns: Metrics payload retrieved or generated by the placeholder.
    :rtype: Dict[str, object]
    """

    metrics = loader()
    if metrics is not None:
        return metrics

    placeholder = placeholder_factory()
    persist_metrics_payload(
        metrics_path,
        placeholder,
        debug_message=debug_message,
    )
    return placeholder


def _gpu_tree_method_supported() -> bool:
    """
    Determine whether the installed XGBoost build supports GPU boosters.

    :returns: ``True`` if GPU boosters appear to be available.
    :rtype: bool
    """

    sweeps_module = sys.modules.get("xgb.pipeline.sweeps")
    xgb_module = getattr(sweeps_module, "xgboost", xgboost) if sweeps_module else xgboost

    if xgb_module is None:
        return False
    core = xgb_module.core  # type: ignore[attr-defined]

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


__all__ = [
    "DEFAULT_OPINION_FEATURE_SPACE",
    "LOGGER",
    "get_sweeps_attr",
    "merge_sweep_outcomes",
    "persist_metrics_payload",
    "xgboost",
    "_gpu_tree_method_supported",
    "_inject_study_metadata",
    "_load_metrics",
    "_load_metrics_with_log",
    "ensure_metrics_with_placeholder",
    "_run_xgb_cli",
]
