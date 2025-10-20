"""Evaluation helpers for KNN baselines."""

from __future__ import annotations

from typing import Any, Dict


def evaluate_predictions(predictions: Any, gold: Any) -> Dict[str, float]:
    """Return metrics comparing predictions to gold labels."""

    raise NotImplementedError("Evaluation metrics will be implemented during refactor")


__all__ = ["evaluate_predictions"]
