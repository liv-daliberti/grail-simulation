"""Parsing and normalisation helpers for prompt construction."""

from __future__ import annotations

import json
import math
from typing import Any, List, Optional

from .constants import FALSE_STRINGS, TRUE_STRINGS

try:  # pragma: no cover - optional dependency
    import pyarrow as pa  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pa = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore


def as_list_json(x: Any, default: str = "[]") -> List[Any]:
    """
    Convert ``x`` into a Python list, accepting JSON strings and Arrow arrays.

    :param x: Source value that may already be a list or serialised representation.
    :type x: Any
    :param default: Fallback JSON array used when ``x`` is empty.
    :type default: str
    :returns: Parsed list representation, or an empty list when conversion fails.
    :rtype: List[Any]
    """

    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            value = json.loads(x or default)
        except (TypeError, ValueError, json.JSONDecodeError):  # pragma: no cover - malformed JSON
            return []
        return value if isinstance(value, list) else []
    if pa is not None and isinstance(x, pa.Array):  # pragma: no cover
        return x.to_pylist()
    return []


def secs(x: Any) -> str:
    """
    Format ``x`` as an integer number of seconds.

    :param x: Duration-like value (numeric or string).
    :type x: Any
    :returns: Seconds expressed as ``"<int>s"`` or ``"?"`` when parsing fails.
    :rtype: str
    """

    try:
        return f"{int(round(float(x)))}s"
    except (TypeError, ValueError):
        return "?"


def _is_nanlike(x: Any) -> bool:
    """
    Evaluate whether ``x`` should be treated as a NaN-equivalent sentinel.

    :param x: Value of arbitrary type to test for NaN-like semantics.
    :type x: Any
    :returns: ``True`` when ``x`` is ``None``, a floating NaN, a pandas NA, or a
        canonical missing-text token.
    :rtype: bool
    """
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if pd is not None:
        try:
            if pd.isna(x):  # type: ignore[union-attr]
                return True
        except (TypeError, ValueError):  # pd.isna on complex objects may raise
            pass
    s = str(x).strip().lower()
    return s in {"", "nan", "none", "null", "n/a", "na"}


def is_nanlike(value: Any) -> bool:
    """
    Determine whether ``value`` should be treated as a missing entry.

    :param value: Candidate value that may represent ``NaN`` or null-ish text.
    :type value: Any
    :returns: ``True`` when the value is considered missing, ``False`` otherwise.
    :rtype: bool
    """

    return _is_nanlike(value)


def _truthy_from_number(value: float) -> Optional[bool]:
    """
    Interpret numeric values that conventionally encode boolean states.

    :param value: Numeric marker expected to be ``1``/``0`` or close equivalents.
    :type value: float
    :returns: ``True`` for one-like values, ``False`` for zero-like values, otherwise ``None``.
    :rtype: Optional[bool]
    """

    if value == 1:
        return True
    if value == 0:
        return False
    return None


def _truthy_from_text(value: Any) -> Optional[bool]:
    """
    Interpret textual encodings of boolean information.

    :param value: Raw value that may contain a boolean label or numeric text.
    :type value: Any
    :returns: ``True`` or ``False`` when a known token or numeric literal is
        detected; ``None`` otherwise.
    :rtype: Optional[bool]
    """

    text = str(value).strip().lower()
    if not text:
        return None
    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False

    result: Optional[bool] = None
    try:
        number = float(text)
    except ValueError:
        number = None

    if number is not None:
        if math.isclose(number, 1.0, rel_tol=0.0, abs_tol=1e-9):
            result = True
        elif math.isclose(number, 0.0, rel_tol=0.0, abs_tol=1e-9):
            result = False
    return result


def truthy(value: Any) -> Optional[bool]:
    """
    Interpret common boolean representations.

    :param value: Input that may encode boolean information.
    :type value: Any
    :returns: ``True`` or ``False`` when recognised, otherwise ``None``.
    :rtype: Optional[bool]
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = _truthy_from_number(float(value))
        if numeric is not None:
            return numeric
    return _truthy_from_text(value)


def format_yes_no(value: Any, *, yes: str = "yes", no: str = "no") -> Optional[str]:
    """
    Render ``value`` as a ``yes``/``no`` string when a boolean marker is present.

    :param value: Input that may encode a boolean concept.
    :type value: Any
    :param yes: Text to return when ``value`` evaluates to ``True``.
    :type yes: str
    :param no: Text to return when ``value`` evaluates to ``False``.
    :type no: str
    :returns: ``yes`` or ``no`` strings, or ``None`` when the value is indeterminate.
    :rtype: Optional[str]
    """

    verdict = truthy(value)
    if verdict is True:
        return yes
    if verdict is False:
        return no
    return None


def format_count(value: Any) -> Optional[str]:
    """
    Present ``value`` as a human-readable count with thousands separators.

    :param value: Numeric or textual count.
    :type value: Any
    :returns: Formatted count string, or ``None`` for missing/invalid input.
    :rtype: Optional[str]
    """

    if _is_nanlike(value):
        return None
    try:
        if isinstance(value, str):
            text = value.replace(",", "").strip()
            if not text:
                return None
            num = float(text)
        else:
            num = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None
    if math.isnan(num):
        return None
    if abs(num - int(round(num))) < 1e-6:
        return f"{int(round(num)):,}"
    return f"{num:,.2f}"


def format_age(value: Any) -> Optional[str]:
    """
    Normalise age-related text into a canonical string representation.

    :param value: Age value supplied as text or numeric.
    :type value: Any
    :returns: Cleaned age string, or ``None`` when the age cannot be inferred.
    :rtype: Optional[str]
    """

    if value is None:
        return None
    try:
        age = int(float(str(value).strip()))
    except (TypeError, ValueError):
        age = None
    if isinstance(age, int) and age > 0:
        return str(age)
    text = str(value).strip()
    return text or None


# Backward-compatibility aliases for the original module-level helpers.
_format_yes_no = format_yes_no
_format_count = format_count
_format_age = format_age

__all__ = [
    "as_list_json",
    "format_age",
    "format_count",
    "format_yes_no",
    "is_nanlike",
    "secs",
    "truthy",
]
