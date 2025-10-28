"""Unit tests for helper utilities in ``xgb.evaluate``."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import pytest

from xgb.evaluate import (
    PredictionOutcome,
    _accuracy_curve_from_records,
    _candidate_probabilities,
    _curve_metrics_from_training_history,
    _probability_context,
    _records_to_predictions,
    _split_tokens,
    _summarise_outcomes,
)

pytestmark = pytest.mark.xgb


def test_split_tokens_trims_and_filters() -> None:
    assert _split_tokens(" alpha , beta,, gamma , ") == ["alpha", "beta", "gamma"]
    assert _split_tokens("") == []
    assert _split_tokens(None) == []


def test_candidate_probabilities_canonicalises_ids() -> None:
    slate: Sequence[Tuple[str, str]] = (
        ("Primary", "https://youtu.be/AbC12345678"),
        ("Secondary", "video_beta"),
    )
    probability_map: Dict[str, float] = {"AbC12345678": 0.75}

    candidate_probs, known_candidates = _candidate_probabilities(
        slate, probability_map
    )

    assert candidate_probs == {1: 0.75, 2: 0.0}
    assert known_candidates == {1: "AbC12345678"}


def test_probability_context_handles_known_and_unknown_candidates() -> None:
    candidate_probs = {1: 0.8, 2: 0.1}
    known_candidates = {1: "abc12345678", 2: "zzz98765432"}

    ctx = _probability_context(
        prediction_idx=1,
        candidate_probs=candidate_probs,
        known_candidates=known_candidates,
        gold_id_canon="abc12345678",
    )
    assert ctx.best_probability == pytest.approx(0.8)
    assert ctx.record_probability is True
    assert ctx.known_candidate_hit is True

    ctx = _probability_context(
        prediction_idx=2,
        candidate_probs=candidate_probs,
        known_candidates=known_candidates,
        gold_id_canon="abc12345678",
    )
    assert ctx.best_probability == pytest.approx(0.1)
    assert ctx.record_probability is True
    assert ctx.known_candidate_hit is False

    ctx = _probability_context(
        prediction_idx=None,
        candidate_probs=candidate_probs,
        known_candidates=known_candidates,
        gold_id_canon="abc12345678",
    )
    assert ctx.best_probability == 0.0
    assert ctx.record_probability is False
    assert ctx.known_candidate_hit is False

    ctx = _probability_context(
        prediction_idx=3,
        candidate_probs=candidate_probs,
        known_candidates=known_candidates,
        gold_id_canon="abc12345678",
    )
    assert ctx.best_probability == 0.0
    assert ctx.record_probability is False
    assert ctx.known_candidate_hit is False


def _prediction_outcome(
    *,
    prediction_index: Optional[int],
    predicted_id: str,
    gold_video_id: str,
    candidate_probs: Dict[int, float],
    best_probability: float,
    known_candidate_seen: bool,
    known_candidate_hit: bool,
    record_probability: bool,
    correct: bool,
    option_count: int = 0,
    gold_index: Optional[int] = None,
    eligible: bool = False,
) -> PredictionOutcome:
    return PredictionOutcome(
        prediction_index=prediction_index,
        predicted_id=predicted_id,
        gold_video_id=gold_video_id,
        candidate_probs=candidate_probs,
        best_probability=best_probability,
        known_candidate_seen=known_candidate_seen,
        known_candidate_hit=known_candidate_hit,
        record_probability=record_probability,
        correct=correct,
        option_count=option_count,
        gold_index=gold_index,
        eligible=eligible,
    )


def test_summarise_outcomes_counts_metrics() -> None:
    records = [
        (
            1,
            _prediction_outcome(
                prediction_index=1,
                predicted_id="vid_alpha",
                gold_video_id="vid_alpha",
                candidate_probs={1: 0.9, 2: 0.1},
                best_probability=0.9,
                known_candidate_seen=True,
                known_candidate_hit=True,
                record_probability=True,
                correct=True,
                option_count=2,
                gold_index=1,
                eligible=True,
            ),
        ),
        (
            2,
            _prediction_outcome(
                prediction_index=2,
                predicted_id="vid_beta",
                gold_video_id="vid_gamma",
                candidate_probs={1: 0.2, 2: 0.3},
                best_probability=0.3,
                known_candidate_seen=False,
                known_candidate_hit=False,
                record_probability=False,
                correct=False,
                option_count=3,
                gold_index=None,
                eligible=False,
            ),
        ),
        (
            3,
            _prediction_outcome(
                prediction_index=1,
                predicted_id="vid_delta",
                gold_video_id="vid_epsilon",
                candidate_probs={1: 0.5},
                best_probability=0.5,
                known_candidate_seen=True,
                known_candidate_hit=False,
                record_probability=True,
                correct=False,
                option_count=1,
                gold_index=1,
                eligible=True,
            ),
        ),
    ]

    summary = _summarise_outcomes(records)

    assert summary.evaluated == 3
    assert summary.correct == 1
    assert summary.known_hits == 1
    assert summary.known_total == 2
    assert summary.avg_probability == pytest.approx(0.7)
    assert summary.eligible == 2
    assert summary.gold_hist == {1: 2}
    assert summary.random_inverse_sum == pytest.approx(1.5)
    assert summary.random_inverse_count == 2


def test_records_to_predictions_serialises_expected_fields() -> None:
    record = (
        7,
        _prediction_outcome(
            prediction_index=2,
            predicted_id="vid_beta",
            gold_video_id="vid_gamma",
            candidate_probs={1: 0.1, 2: 0.9},
            best_probability=0.9,
            known_candidate_seen=True,
            known_candidate_hit=False,
            record_probability=True,
            correct=False,
        ),
    )

    payload = _records_to_predictions([record], issue_slug="issue-42")

    assert payload == [
        {
            "issue": "issue-42",
            "index": 7,
            "prediction_index": 2,
            "predicted_video_id": "vid_beta",
            "gold_video_id": "vid_gamma",
            "correct": False,
            "probabilities": {1: 0.1, 2: 0.9},
        }
    ]


def test_accuracy_curve_from_records_tracks_progression() -> None:
    outcomes = [
        _prediction_outcome(
            prediction_index=1,
            predicted_id=f"vid_{idx}",
            gold_video_id=f"gold_{idx}",
            candidate_probs={1: 0.5},
            best_probability=0.5,
            known_candidate_seen=False,
            known_candidate_hit=False,
            record_probability=False,
            correct=correct,
        )
        for idx, correct in enumerate([True, False, True, True, False], start=1)
    ]
    records = list(enumerate(outcomes, start=1))

    curve = _accuracy_curve_from_records(records, target_points=2)

    assert curve["n_examples"] == 5
    assert curve["stride"] == 2
    assert curve["accuracy_by_step"]["2"] == pytest.approx(0.5)
    assert curve["accuracy_by_step"]["5"] == pytest.approx(0.6)


def test_curve_metrics_from_training_history_builds_series() -> None:
    history = {
        "validation_0": {"merror": [0.4, 0.1]},
        "validation_1": {"merror": [0.5, 0.25]},
    }

    metrics = _curve_metrics_from_training_history(history)

    assert metrics is not None
    assert metrics["metric"] == "merror"
    assert metrics["train"]["n_rounds"] == 2
    assert metrics["eval"]["n_rounds"] == 2
    assert metrics["train"]["accuracy_by_round"] == {"1": 0.6, "2": 0.9}
    assert metrics["eval"]["accuracy_by_round"] == {"1": 0.5, "2": 0.75}


def test_curve_metrics_from_training_history_handles_invalid_payload() -> None:
    assert _curve_metrics_from_training_history(None) is None
    assert _curve_metrics_from_training_history({}) is None
    assert (
        _curve_metrics_from_training_history({"validation_0": {"merror": []}}) is None
    )
