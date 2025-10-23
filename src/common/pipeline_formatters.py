"""Shared formatting helpers for pipeline reports and evaluations."""

from __future__ import annotations

from typing import Optional


def format_float(value: float) -> str:
    """

    Format a floating-point metric with three decimal places.



    :param value: Value provided for ``value``.

    :type value: float

    :returns: Result produced by ``format_float``.

    :rtype: str

    """


    return f"{value:.3f}"


def format_optional_float(value: Optional[float]) -> str:
    """

    Format optional floating-point metrics.



    :param value: Value provided for ``value``.

    :type value: Optional[float]

    :returns: Result produced by ``format_optional_float``.

    :rtype: str

    """


    return format_float(value) if value is not None else "—"


def format_delta(value: Optional[float]) -> str:
    """

    Return a signed delta with three decimal places.



    :param value: Value provided for ``value``.

    :type value: Optional[float]

    :returns: Result produced by ``format_delta``.

    :rtype: str

    """


    return f"{value:+.3f}" if value is not None else "—"


def format_count(value: Optional[int]) -> str:
    """

    Render optional integer counts with thousands separators.



    :param value: Value provided for ``value``.

    :type value: Optional[int]

    :returns: Result produced by ``format_count``.

    :rtype: str

    """


    return f"{value:,}" if value is not None else "—"


def format_ratio(numerator: Optional[int], denominator: Optional[int]) -> str:
    """

    Format ratios as 'hit/total' when both sides are known.



    :param numerator: Value provided for ``numerator``.

    :type numerator: Optional[int]

    :param denominator: Value provided for ``denominator``.

    :type denominator: Optional[int]

    :returns: Result produced by ``format_ratio``.

    :rtype: str

    """


    if numerator is None or denominator is None:
        return "—"
    return f"{numerator:,}/{denominator:,}"


def safe_float(value: object) -> Optional[float]:
    """

    Best-effort conversion to float.



    :param value: Value provided for ``value``.

    :type value: object

    :returns: Result produced by ``safe_float``.

    :rtype: Optional[float]

    """


    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: object) -> Optional[int]:
    """

    Best-effort conversion to int.



    :param value: Value provided for ``value``.

    :type value: object

    :returns: Result produced by ``safe_int``.

    :rtype: Optional[int]

    """


    try:
        return int(value)
    except (TypeError, ValueError):
        return None
