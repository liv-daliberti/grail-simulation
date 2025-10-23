"""Shared helpers used across prompt_builder modules."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .parsers import is_nanlike


def first_non_nan_value(
    example: Dict[str, Any],
    selected: Dict[str, Any],
    *keys: str,
) -> Optional[Any]:
    """Return the first non-null/non-NaN-like value from ``example`` or ``selected``."""

    for key in keys:
        if key in example:
            value = example[key]
            if value is not None and not is_nanlike(value):
                return value
        if key in selected:
            value = selected.get(key)
            if value is not None and not is_nanlike(value):
                return value
    return None
