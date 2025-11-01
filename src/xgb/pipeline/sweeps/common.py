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

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypeVar, cast

from common.pipeline.io import load_metrics_json
from common.pipeline.metrics import (
    ensure_metrics_with_placeholder,
    persist_metrics_payload,
)
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


@dataclass(frozen=True)
class MissingMetricsLogConfig:
    """
    Configuration for logging and persistence of placeholder metrics.

    :param message: Logging format string used when metrics cannot be loaded.
    :type message: str
    :param debug_message: Debug message emitted when placeholder persistence fails.
    :type debug_message: str
    :param log_level: Logging level used when ``message`` is emitted.
    :type log_level: int
    :param logger: Optional logger used for debug persistence failures.
    :type logger: logging.Logger | None
    """

    message: str
    debug_message: str
    log_level: int = logging.WARNING
    logger: logging.Logger | None = None

    def level_name(self) -> str:
        """Return the textual name for ``log_level`` (e.g., WARNING)."""
        return logging.getLevelName(self.log_level)

    def as_kwargs(self) -> Dict[str, object]:
        """Expose a serialisable view suitable for debugging or tests."""
        return {
            "message": self.message,
            "debug_message": self.debug_message,
            "log_level": int(self.log_level),
            "logger": getattr(self.logger, "name", None),
        }


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


def build_merge_sweep_outcomes(
    *,
    duplicate_message: str,
    docstring: Optional[str] = None,
) -> Callable[[Sequence[OutcomeT], Sequence[OutcomeT]], List[OutcomeT]]:
    """
    Return a callable that merges cached and executed outcomes.

    :param duplicate_message: Warning message emitted when cached outcomes are replaced.
    :type duplicate_message: str
    :param docstring: Custom docstring attached to the generated callable.
    :type docstring: str | None
    :returns: Callable combining cached and executed outcomes ordered by ``order_index``.
    :rtype: Callable[[Sequence[OutcomeT], Sequence[OutcomeT]], List[OutcomeT]]
    """

    def _merge(
        cached: Sequence[OutcomeT],
        executed: Sequence[OutcomeT],
    ) -> List[OutcomeT]:
        return merge_sweep_outcomes(
            cached,
            executed,
            duplicate_message=duplicate_message,
        )

    _merge.__doc__ = (
        docstring
        if docstring is not None
        else "Merge cached and freshly executed sweep outcomes while preserving order indices."
    )
    return _merge


def load_metrics_with_placeholder(
    *,
    metrics_path: Path,
    study: StudySpec,
    loader: Callable[[Path, StudySpec, int, str], Dict[str, object] | None],
    placeholder_factory: Callable[[], Dict[str, object]],
    log_config: MissingMetricsLogConfig,
) -> Dict[str, object]:
    """
    Load metrics via ``loader`` or materialise a placeholder payload on failure.

    :param metrics_path: Filesystem path where metrics are stored.
    :type metrics_path: Path
    :param study: Study descriptor used when emitting log messages.
    :type study: ~common.pipeline.types.StudySpec
    :param loader: Callable mirroring :func:`_load_metrics_with_log` semantics.
    :type loader: Callable[[Path, StudySpec, int, str], Dict[str, object] | None]
    :param placeholder_factory: Callable producing a fallback metrics payload.
    :type placeholder_factory: Callable[[], Dict[str, object]]
    :param log_config: Logging configuration applied when metrics are missing.
    :type log_config: MissingMetricsLogConfig
    :returns: Metrics payload retrieved from disk or generated by the placeholder.
    :rtype: Dict[str, object]
    """

    return ensure_metrics_with_placeholder(
        lambda: loader(
            metrics_path,
            study,
            log_level=log_config.log_level,
            message=log_config.message,
        ),
        placeholder_factory=placeholder_factory,
        metrics_path=metrics_path,
        debug_message=log_config.debug_message,
        logger=log_config.logger or LOGGER,
    )

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
        try:
            return bool(cast(Callable[[], object], maybe_has_cuda)())
        except (TypeError, ValueError, RuntimeError, AttributeError):
            LOGGER.debug("Failed to query XGBoost CUDA support.", exc_info=True)
            return False

    # Fallback: inspect the shared library for a device-specific symbol.
    lib = getattr(core, "_LIB", None)
    return hasattr(lib, "XGBoosterPredictFromDeviceDMatrix")


__all__ = [
    "DEFAULT_OPINION_FEATURE_SPACE",
    "LOGGER",
    "MissingMetricsLogConfig",
    "get_sweeps_attr",
    "build_merge_sweep_outcomes",
    "load_metrics_with_placeholder",
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
