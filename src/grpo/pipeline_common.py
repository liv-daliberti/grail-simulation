#!/usr/bin/env python
"""Common helpers for the GRPO evaluation pipeline.

This module centralizes light-weight utilities used across the pipeline
implementation to keep the main entrypoint small and focused.
"""

from __future__ import annotations

import logging
import math
from typing import Mapping
import faulthandler


LOGGER = logging.getLogger("grpo.pipeline")


def _status(message: str, *args) -> None:
    """Log ``message`` at INFO level and mirror it to stdout immediately."""

    text = message % args if args else message
    LOGGER.info(text)
    print(f"[grpo.pipeline] {text}", flush=True)


# Public message added to report READMEs to help regeneration.
DEFAULT_REGENERATE_HINT = (
    "Regenerate via `python -m grpo.pipeline --stage full` after producing "
    "updated evaluation artifacts under `models/grpo/`."
)


def configure_logging(log_level: str | int) -> None:
    """Configure structured logging and enable faulthandler early."""

    if isinstance(log_level, str):
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        numeric_level = int(log_level)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    try:  # pragma: no cover - faulthandler may already be active
        faulthandler.enable()
    except RuntimeError:
        LOGGER.debug("faulthandler already enabled")
    LOGGER.debug(
        "Logging configured at level %s", logging.getLevelName(numeric_level)
    )


def _comma_separated(raw: str) -> tuple[str, ...]:
    """Split comma-separated CLI tokens into a tuple."""

    if not raw:
        return ()
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def _safe_float(value: object) -> float | None:
    """Return ``value`` coerced to ``float`` when possible."""

    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _safe_int(value: object) -> int | None:
    """Return ``value`` coerced to ``int`` when possible."""

    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _fmt_rate(value: float | None, digits: int = 3) -> str:
    """Return a formatted rate or an em dash when ``value`` is missing."""

    return f"{value:.{digits}f}" if value is not None else "—"


def _fmt_count(value: int | None) -> str:
    """Return an integer with thousands separators or an em dash."""

    return f"{value:,}" if value is not None else "—"


def _extract_next_video_metrics(payload: Mapping[str, object] | None) -> Mapping[str, object]:
    """Normalise next-video metrics payloads that may wrap values under ``metrics``."""

    if not isinstance(payload, Mapping):
        return {}
    if "accuracy_overall" in payload:
        return payload
    nested = payload.get("metrics")
    if isinstance(nested, Mapping):
        return nested
    return payload


def _log_next_video_summary(result) -> None:  # lazy-typed to avoid import cycle
    """Emit INFO-level summary metrics for a NextVideoEvaluationResult."""

    if result is None:
        LOGGER.info("Next-video metrics unavailable; skipping summary log.")
        return

    metrics = _extract_next_video_metrics(getattr(result, "metrics", None))
    if not metrics:
        LOGGER.info("Next-video metrics payload empty; skipping summary log.")
        return

    accuracy = _safe_float(metrics.get("accuracy_overall"))
    parsed_rate = _safe_float(metrics.get("parsed_rate"))
    format_rate = _safe_float(metrics.get("format_rate"))
    eligible = _safe_int(metrics.get("n_eligible"))
    total = _safe_int(metrics.get("n_total"))
    baseline_block = metrics.get("baseline_most_frequent_gold_index")
    baseline_accuracy = _safe_float(baseline_block.get("accuracy")) if isinstance(
        baseline_block, Mapping
    ) else None
    random_accuracy = _safe_float(metrics.get("random_baseline_expected_accuracy"))
    delta_accuracy = (
        accuracy - baseline_accuracy
        if accuracy is not None and baseline_accuracy is not None
        else None
    )

    LOGGER.info(
        "Next-video metrics | accuracy=%s | baseline=%s | Δ=%s | "
        "random=%s | parsed=%s | format=%s | eligible=%s/%s",
        _fmt_rate(accuracy),
        _fmt_rate(baseline_accuracy),
        _fmt_rate(delta_accuracy),
        _fmt_rate(random_accuracy),
        _fmt_rate(parsed_rate),
        _fmt_rate(format_rate),
        _fmt_count(eligible),
        _fmt_count(total),
    )

    if isinstance(baseline_block, Mapping):
        top_index = baseline_block.get("top_index")
        baseline_count = _safe_int(baseline_block.get("count"))
        LOGGER.debug(
            "Baseline (most frequent gold index): index=%s count=%s",
            top_index if top_index is not None else "—",
            _fmt_count(baseline_count),
        )


def _weighted_baseline_direction(result) -> float | None:  # lazy-typed to avoid import cycle
    """Return an eligible-weighted baseline directional accuracy for ``result``."""

    if result is None:
        return None
    numerator = 0.0
    denominator = 0
    for study in getattr(result, "studies", []) or []:
        baseline = getattr(study, "baseline", None) or getattr(study, "summary", None)
        eligible = getattr(study, "eligible", None) or getattr(study, "summary", {}).get(
            "eligible"
        )
        baseline_direction = (
            _safe_float(baseline.get("direction_accuracy"))
            if isinstance(baseline, Mapping)
            else None
        )
        if baseline_direction is None or not eligible:
            continue
        denominator += int(eligible)
        numerator += baseline_direction * int(eligible)
    if denominator:
        return numerator / denominator
    return None


def _log_opinion_summary(result) -> None:  # lazy-typed to avoid import cycle
    """Emit INFO-level summary metrics for aggregated opinion results."""

    if result is None:
        LOGGER.info("Opinion metrics unavailable; skipping summary log.")
        return

    combined = result.combined_metrics if isinstance(result.combined_metrics, Mapping) else {}
    if not combined:
        LOGGER.info("Opinion combined metrics payload empty; skipping summary log.")
        return

    direction_accuracy = _safe_float(combined.get("direction_accuracy"))
    mae_after = _safe_float(combined.get("mae_after"))
    rmse_after = _safe_float(combined.get("rmse_after"))
    mae_change = _safe_float(combined.get("mae_change"))
    rmse_change = _safe_float(combined.get("rmse_change"))
    eligible = _safe_int(combined.get("eligible"))
    baseline_direction = _weighted_baseline_direction(result)
    delta_direction = (
        direction_accuracy - baseline_direction
        if direction_accuracy is not None and baseline_direction is not None
        else None
    )
    participants = sum(int(getattr(study, "participants", 0)) for study in result.studies)

    LOGGER.info(
        "Opinion metrics | direction=%s | baseline=%s | Δ=%s | "
        "mae_after=%s | rmse_after=%s | mae_change=%s | rmse_change=%s | "
        "eligible=%s | participants=%s",
        _fmt_rate(direction_accuracy),
        _fmt_rate(baseline_direction),
        _fmt_rate(delta_direction),
        _fmt_rate(mae_after),
        _fmt_rate(rmse_after),
        _fmt_rate(mae_change),
        _fmt_rate(rmse_change),
        _fmt_count(eligible),
        _fmt_count(participants),
    )
