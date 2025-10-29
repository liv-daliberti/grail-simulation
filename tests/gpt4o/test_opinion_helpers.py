"""Unit tests for helper utilities in :mod:`gpt4o.opinion`."""

from __future__ import annotations

import math

import pytest

from gpt4o import opinion
from tests.helpers.datasets_stub import ensure_datasets_stub

pytestmark = pytest.mark.gpt4o

ensure_datasets_stub()


def test_parse_tokens_normalises_entries_and_handles_all_keyword() -> None:
    """The helper should keep ordering while clearing lowercase set on ``all``."""

    tokens, lowered = opinion._parse_tokens("alpha, Bravo , all")  # pylint: disable=protected-access
    assert tokens == ["alpha", "Bravo", "all"]
    assert lowered == set()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3.5", 3.5),
        (None, None),
        ("not-a-number", None),
        (float("nan"), None),
    ],
)
def test_float_or_none_handles_invalid_inputs(raw: object, expected: float | None) -> None:
    """Numeric parsing should discard invalid values while preserving floats."""

    result = opinion._float_or_none(raw)  # pylint: disable=protected-access
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_document_from_example_combines_profile_context_and_titles() -> None:
    """Constructed document should include all present sections with spacing."""

    example = {
        "viewer_profile": "Profile text",
        "state_text": "Context line",
        "current_video_title": "Current Title",
        "next_video_title": "Next Title",
    }
    document = opinion._document_from_example(example)  # pylint: disable=protected-access
    assert "Viewer profile:\nProfile text" in document
    assert "Context:\nContext line" in document
    assert "Currently watching: Current Title" in document
    assert document.endswith("Next video shown: Next Title")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.2, 1.0),
        (3.5, 3.5),
        (9.1, 7.0),
    ],
)
def test_clip_prediction_enforces_bounds(value: float, expected: float) -> None:
    """Predictions should be clipped to the inclusive [1, 7] interval."""

    assert opinion._clip_prediction(value) == pytest.approx(expected)  # pylint: disable=protected-access


def test_baseline_metrics_matches_expected_statistics() -> None:
    """Baseline metrics should mirror the behaviour documented for KNN/XGB."""

    truth_before = [2.0, 3.0, 4.0]
    truth_after = [4.0, 6.0, 5.0]
    metrics = opinion._baseline_metrics(truth_before, truth_after)  # pylint: disable=protected-access

    assert metrics["global_mean_after"] == pytest.approx(sum(truth_after) / len(truth_after))
    assert metrics["mae_global_mean_after"] == pytest.approx(2.0 / 3.0)
    assert metrics["rmse_global_mean_after"] == pytest.approx(math.sqrt(2.0 / 3.0))


def test_resolve_spec_keys_defaults_to_known_specs() -> None:
    """Omitting overrides should fall back to the default opinion study list."""

    expected = [spec.key for spec in opinion.DEFAULT_SPECS]
    assert opinion._resolve_spec_keys(None) == expected  # pylint: disable=protected-access

    override = opinion._resolve_spec_keys(" first , second ")  # pylint: disable=protected-access
    assert override == ["first", "second"]
