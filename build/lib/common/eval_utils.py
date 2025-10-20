"""Shared evaluation utilities reused across baseline implementations."""

from __future__ import annotations


def safe_div(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    """
    Return the division result guarding against a zero denominator.

    Parameters
    ----------
    numerator:
        Value forming the numerator.
    denominator:
        Value forming the denominator.
    default:
        Fallback value returned when ``denominator`` is zero.
    """

    if not denominator:
        return default
    return numerator / denominator


__all__ = ["safe_div"]
