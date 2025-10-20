"""Parsing and normalisation helpers for prompt construction."""

from __future__ import annotations

import json
import math
from typing import Any, List, Optional

from .constants import FALSE_STRINGS, TRUE_STRINGS


def as_list_json(x: Any, default: str = "[]") -> List[Any]:
    """Return ``x`` as a Python list, accepting JSON strings and Arrow arrays."""

    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            value = json.loads(x or default)
            return value if isinstance(value, list) else []
        except Exception:  # pragma: no cover - defensive for malformed JSON
            return []
    try:  # pragma: no cover - pyarrow may be unavailable in some environments
        import pyarrow as pa  # type: ignore

        if isinstance(x, pa.Array):
            return x.to_pylist()
    except Exception:
        pass
    return []


def secs(x: Any) -> str:
    """Format a duration in seconds, returning ``"?"`` when parsing fails."""

    try:
        return f"{int(round(float(x)))}s"
    except Exception:
        return "?"


def _is_nanlike(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float):
        if math.isnan(x):
            return True
    try:
        import pandas as pd  # type: ignore

        if isinstance(x, pd._libs.missing.NAType):  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    s = str(x).strip().lower()
    return s in {"", "nan", "none", "null", "n/a", "na"}


def is_nanlike(value: Any) -> bool:
    """Public wrapper for :func:`_is_nanlike`."""

    return _is_nanlike(value)


def truthy(value: Any) -> Optional[bool]:
    """Interpret common boolean representations, returning ``None`` when unknown."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False
    return None


def format_yes_no(value: Any, *, yes: str = "yes", no: str = "no") -> Optional[str]:
    """Return ``yes``/``no`` strings when ``value`` contains a truthy marker."""

    verdict = truthy(value)
    if verdict is True:
        return yes
    if verdict is False:
        return no
    return None


def format_count(value: Any) -> Optional[str]:
    """Format counts using thousands separators, falling back to raw text."""

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
        if math.isnan(num):
            return None
        if abs(num - int(round(num))) < 1e-6:
            return f"{int(round(num)):,}"
        return f"{num:,.2f}"
    except Exception:
        text = str(value).strip()
        return text or None


def format_age(value: Any) -> Optional[str]:
    """Return a normalised age string when ``value`` contains a valid number."""

    if value is None:
        return None
    try:
        age = int(float(str(value).strip()))
        if age > 0:
            return str(age)
    except Exception:
        pass
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
