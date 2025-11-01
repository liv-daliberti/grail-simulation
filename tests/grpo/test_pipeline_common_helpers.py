#!/usr/bin/env python
# pylint: disable=missing-function-docstring
from __future__ import annotations

import types

import pytest

from grpo.pipeline_common import (
    _comma_separated,
    _extract_next_video_metrics,
    _fmt_count,
    _fmt_rate,
    _log_next_video_summary,
    _log_opinion_summary,
    _safe_float,
    _safe_int,
)


def test_comma_separated_handles_empty_and_whitespace() -> None:
    assert _comma_separated("") == ()
    assert _comma_separated(" , , ") == ()
    assert _comma_separated("a, b ,c") == ("a", "b", "c")


def test_safe_float_and_int_cover_invalid_and_nan() -> None:
    assert _safe_float("1.5") == 1.5
    assert _safe_float(None) is None
    assert _safe_float("nan") is None

    assert _safe_int("3") == 3
    assert _safe_int(7.9) == 7
    assert _safe_int(None) is None


def test_fmt_helpers_render_em_dash_on_missing() -> None:
    assert _fmt_rate(0.1234, digits=3) == "0.123"
    assert _fmt_rate(None) == "—"
    assert _fmt_count(12345) == "12,345"
    assert _fmt_count(None) == "—"


def test_extract_next_video_metrics_unwraps_nested() -> None:
    m = {"metrics": {"accuracy_overall": 0.5, "n_total": 10}}
    assert _extract_next_video_metrics(m) == m["metrics"]
    assert _extract_next_video_metrics(m["metrics"]) == m["metrics"]
    assert _extract_next_video_metrics(None) == {}


def test_log_next_video_summary_handles_missing_payloads(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger="grpo.pipeline"):
        _log_next_video_summary(None)
    assert any("unavailable; skipping" in msg for msg in caplog.messages)


def test_weighted_baseline_direction_and_opinion_summary(caplog: pytest.LogCaptureFixture) -> None:
    # Study baselines: 0.5 @ 10 eligible, 0.7 @ 20 eligible
    study1 = types.SimpleNamespace(baseline={"direction_accuracy": 0.5}, eligible=10, participants=10)
    study2 = types.SimpleNamespace(baseline={"direction_accuracy": 0.7}, eligible=20, participants=20)
    result = types.SimpleNamespace(
        studies=[study1, study2],
        combined_metrics={
            "direction_accuracy": 0.7,
            "mae_after": 0.1,
            "rmse_after": 0.2,
            "mae_change": 0.05,
            "rmse_change": 0.15,
            "eligible": 30,
        },
    )
    caplog.clear()
    with caplog.at_level("INFO", logger="grpo.pipeline"):
        _log_opinion_summary(result)
    # Expect 3-digit rounded direction and baseline
    summary_lines = "\n".join(caplog.messages)
    assert "direction=0.700" in summary_lines
    # Weighted baseline: (0.5*10 + 0.7*20) / 30 = 0.6333 -> 0.633
    assert "baseline=0.633" in summary_lines

