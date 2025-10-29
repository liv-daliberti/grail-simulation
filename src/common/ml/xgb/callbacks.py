#!/usr/bin/env python
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
    """Return a function that extracts a concise metric string from eval logs."""

    def _cls_metrics(block: dict) -> Optional[str]:
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
            self._interval = max(1, int(interval))

        def after_iteration(self, _model, epoch: int, evals_log):  # type: ignore[override]
            """Log metrics every ``interval`` boosting rounds."""
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
