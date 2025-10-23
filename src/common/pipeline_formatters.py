"""Shared formatting helpers for pipeline reports and evaluations."""

from __future__ import annotations

from typing import Optional


def format_float(value: float) -> str:
    """Format a floating-point metric with three decimal places."""

    return f"{value:.3f}"


def format_optional_float(value: Optional[float]) -> str:
    """Format optional floating-point metrics."""

    return format_float(value) if value is not None else "—"


def format_delta(value: Optional[float]) -> str:
    """Return a signed delta with three decimal places."""

    return f"{value:+.3f}" if value is not None else "—"


def format_count(value: Optional[int]) -> str:
    """Render optional integer counts with thousands separators."""

    return f"{value:,}" if value is not None else "—"


def format_ratio(numerator: Optional[int], denominator: Optional[int]) -> str:
    """Format ratios as 'hit/total' when both sides are known."""

    if numerator is None or denominator is None:
        return "—"
    return f"{numerator:,}/{denominator:,}"


def safe_float(value: object) -> Optional[float]:
    """Best-effort conversion to float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: object) -> Optional[int]:
    """Best-effort conversion to int."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None

