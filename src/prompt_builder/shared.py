"""Shared helpers used across prompt_builder modules."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .parsers import is_nanlike


def first_non_nan_value(
    example: Dict[str, Any],
    selected: Dict[str, Any],
    *keys: str,
) -> Optional[Any]:
    """
    Return the first value from ``example`` or ``selected`` that is not NaN-like.

    :param example: Primary dataset row containing viewer or survey metadata.
    :type example: Dict[str, Any]
    :param selected: Secondary mapping representing ``selected_survey_row`` fields.
    :type selected: Dict[str, Any]
    :param keys: Ordered candidate field names to inspect in both mappings.
    :type keys: str
    :returns: The first value that is neither ``None`` nor :func:`prompt_builder.parsers.is_nanlike`.
    :rtype: Optional[Any]
    """

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
