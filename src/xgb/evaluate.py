#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.

"""Compatibility shim exposing the legacy ``xgb.evaluate`` API.

The evaluation implementation lives in :mod:`xgb.core.evaluate`. This module
re-exports the helpers referenced by tests and downstream scripts, including
private names used for monkeypatching.
"""

from __future__ import annotations

# Import the implementation module and selectively re-export attributes,
# including private names referenced by tests.
from .core import evaluate as _core
from .core.evaluate_helpers import split_tokens as _public_split_tokens

# Public dataclasses and entrypoints
PredictionOutcome = _core.PredictionOutcome
run_eval = _core.run_eval

# Private helpers used in tests (underscore-prefixed)
# Expose them by aliasing to public symbols to avoid protected access.
_accuracy_curve_from_records = _core.accuracy_curve_from_records
_candidate_probabilities = _core.candidate_probabilities
_curve_metrics_from_training_history = _core.curve_metrics_from_training_history
_probability_context = _core.probability_context
_records_to_predictions = _core.records_to_predictions
_split_tokens = _public_split_tokens
_summarise_outcomes = _core.summarise_outcomes
_evaluate_issue = _core.evaluate_issue

# Dataset resolver symbol that tests monkeypatch; delegates to core wrapper.
prepare_dataset = _core.prepare_dataset

__all__ = [
    "PredictionOutcome",
    "run_eval",
    # Explicitly export private helpers for test imports
    "_accuracy_curve_from_records",
    "_candidate_probabilities",
    "_curve_metrics_from_training_history",
    "_probability_context",
    "_records_to_predictions",
    "_split_tokens",
    "_summarise_outcomes",
    "_evaluate_issue",
    # Monkeypatch target
    "prepare_dataset",
]
