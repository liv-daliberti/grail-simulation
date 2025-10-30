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

"""
Shared XGBoost training callbacks used across classification and regression.

All imports are performed lazily via importlib to avoid hard dependencies in
lint environments where xgboost may be missing.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, List, Optional


def _load_callback_classes():
    """Return (TrainingCallback, EarlyStopping | None) if available, else (None, None)."""

    try:
        # Use importlib.import_module so tests can monkeypatch it reliably.
        mod = importlib.import_module("xgboost.callback")
        training_callback_cls = getattr(mod, "TrainingCallback")  # type: ignore[attr-defined]
        early_stopping_cls = getattr(mod, "EarlyStopping", None)  # type: ignore[attr-defined]
        return training_callback_cls, early_stopping_cls
    except (ImportError, AttributeError):  # pragma: no cover - optional dependency
        return None, None


def _metric_text_fn(objective: str) -> Callable[[dict], Optional[str]]:
    """
    Return a function that extracts a concise metric string from eval logs.

    :param objective: Objective label used to select classification or regression metrics.
    :returns: Callable that accepts an evaluation log mapping and returns a metric string.
    """

    def _cls_metrics(block: dict) -> Optional[str]:
        """
        Derive a classification metric string from ``block``.

        :param block: Evaluation log mapping emitted by XGBoost.
        :returns: Metric string such as ``\"acc=0.9876\"`` or ``None`` when unavailable.
        """
        if not isinstance(block, dict):
            return None
        if "merror" in block and block["merror"]:
            try:
                err = float(block["merror"][-1])
                return f"acc={1.0 - err:.4f}"
            except (ValueError, TypeError, KeyError, IndexError):  # pragma: no cover - defensive
                return None
        if "mlogloss" in block and block["mlogloss"]:
            try:
                return f"mlogloss={float(block['mlogloss'][-1]):.4f}"
            except (ValueError, TypeError, KeyError, IndexError):  # pragma: no cover - defensive
                return None
        return None

    def _reg_metrics(block: dict) -> Optional[str]:
        """
        Derive a regression metric string from ``block``.

        :param block: Evaluation log mapping emitted by XGBoost.
        :returns: Metric string such as ``\"mae=0.1234\"`` or ``None`` when unavailable.
        """
        if not isinstance(block, dict):
            return None
        if "mae" in block and block["mae"]:
            try:
                return f"mae={float(block['mae'][-1]):.4f}"
            except (ValueError, TypeError, KeyError, IndexError):  # pragma: no cover - defensive
                return None
        if "rmse" in block and block["rmse"]:
            try:
                return f"rmse={float(block['rmse'][-1]):.4f}"
            except (ValueError, TypeError, KeyError, IndexError):  # pragma: no cover - defensive
                return None
        return None

    return _cls_metrics if str(objective).lower().startswith("cls") else _reg_metrics


def make_progress_logger(
    *,
    objective: str,
    logger,
    prefix: str,
    interval: int = 25,
):
    """
    Return an XGBoost TrainingCallback instance that logs progress, or None.

    :param objective: "classification" or "regression" to control metric selection.
    :param logger: Logger exposing .info for progress messages.
    :param prefix: Message prefix tag (e.g., "[XGB][Train]").
    :param interval: Logging period measured in boosting rounds.
    """

    training_callback_cls, _ = _load_callback_classes()
    if training_callback_cls is None:
        return None

    metric_text = _metric_text_fn(objective)

    class _ProgressLogger(training_callback_cls):  # type: ignore
        # pylint: disable=too-few-public-methods
        def __init__(self, interval: int = 25) -> None:
            """
            Configure the logging interval for the progress callback.

            :param interval: Number of boosting rounds between log entries.
            """
            self._interval = max(1, int(interval))

        def after_iteration(self, _model, epoch: int, evals_log):  # type: ignore[override]
            """
            Log metrics every ``interval`` boosting rounds.

            :param _model: Booster instance (unused).
            :param epoch: Zero-based boosting round index.
            :param evals_log: Mapping of evaluation metrics keyed by dataset name.
            :returns: ``False`` to continue training.
            """
            round_idx = int(epoch) + 1
            if round_idx % self._interval != 0:
                return False
            train_block = evals_log.get("validation_0", {}) if hasattr(evals_log, "get") else {}
            eval_block = evals_log.get("validation_1", {}) if hasattr(evals_log, "get") else {}
            train_text = metric_text(train_block)
            eval_text = metric_text(eval_block)
            parts: list[str] = []
            if train_text:
                parts.append(f"train {train_text}")
            if eval_text:
                parts.append(f"eval {eval_text}")
            if parts:
                logger.info("%s round=%d  %s", prefix, round_idx, "  ".join(parts))
            return False

    return _ProgressLogger(interval=interval)


def build_fit_callbacks(  # pylint: disable=too-many-arguments
    *,
    objective: str,
    logger,
    prefix: str,
    has_eval: bool,
    interval: int = 25,
    early_stopping_metric: Optional[str] = None,
    early_stopping_data_name: str = "validation_1",
    early_stopping_rounds: int = 50,
) -> List[Any]:
    """
    Construct a list of XGBoost callbacks for training progress and early stopping.

    Returns an empty list when callbacks are unavailable.

    :param objective: Objective label forwarded to :func:`make_progress_logger`.
    :param logger: Logger used for progress output.
    :param prefix: Prefix string prepended to progress messages.
    :param has_eval: Indicates whether evaluation data is provided.
    :param interval: Number of boosting rounds between progress logs.
    :param early_stopping_metric: Optional metric name used by early stopping.
    :param early_stopping_data_name: Evaluation dataset monitored for early stopping.
    :param early_stopping_rounds: Patience in rounds before early stopping triggers.
    :returns: List of configured callback instances.
    """

    training_callback_cls, early_stopping_cls = _load_callback_classes()
    if training_callback_cls is None:
        return []

    callbacks: list[Any] = []
    progress = make_progress_logger(
        objective=objective, logger=logger, prefix=prefix, interval=interval
    )
    if progress is not None:
        callbacks.append(progress)

    if has_eval and early_stopping_cls is not None:
        try:
            if early_stopping_metric:
                callbacks.append(
                    early_stopping_cls(
                        rounds=early_stopping_rounds,
                        metric_name=early_stopping_metric,
                        data_name=early_stopping_data_name,
                    )
                )
            else:
                callbacks.append(
                    early_stopping_cls(rounds=early_stopping_rounds)  # type: ignore[misc]
                )
        except TypeError:
            # Older callback signature
            callbacks.append(early_stopping_cls(rounds=early_stopping_rounds))  # type: ignore[misc]
    return callbacks


__all__ = [
    "build_fit_callbacks",
    "make_progress_logger",
]
