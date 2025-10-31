"""Extra field metadata and helpers for prompt document assembly."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Sequence, Tuple

gun_field_labels: Dict[str, str]
min_wage_field_labels: Dict[str, str]

try:
    _constants_module = import_module("prompt_builder.constants")
    gun_field_labels = dict(getattr(_constants_module, "GUN_FIELD_LABELS", {}))
    min_wage_field_labels = dict(getattr(_constants_module, "MIN_WAGE_FIELD_LABELS", {}))
except ImportError:  # pragma: no cover - optional dependency for lint/docs builds
    gun_field_labels = {}
    min_wage_field_labels = {}

try:
    value_maps = import_module("prompt_builder.value_maps")
except ImportError:  # pragma: no cover - optional dependency for lint/docs builds

    class _ValueMapsStub:  # pylint: disable=too-few-public-methods
        """Fallback shim used when ``prompt_builder`` is unavailable."""

        @staticmethod
        def format_field_value(_field: str, value: Any) -> str:
            """Return a string representation compatible with prompt formatting."""
            if value is None:
                return ""
            return str(value)

    value_maps = _ValueMapsStub()

DEFAULT_EXTRA_TEXT_FIELDS: Tuple[str, ...] = ("viewer_profile", "state_text")

EXTRA_FIELD_LABELS: Dict[str, str] = {
    "pid1": "Party identification",
    "pid2": "Party lean",
    "ideo1": "Political ideology",
    "ideo2": "Ideology intensity",
    "pol_interest": "Political interest",
    "religpew": "Religion",
    "freq_youtube": "YouTube frequency",
    "youtube_time": "YouTube time",
    "newsint": "News attention",
    "participant_study": "Participant study",
    "slate_source": "Slate source",
    "educ": "Education level",
    "employ": "Employment status",
    "child18": "Children in household",
    "inputstate": "State",
    "q31": "Household income",
    "income": "Household income",
}
EXTRA_FIELD_LABELS.update(min_wage_field_labels)
EXTRA_FIELD_LABELS.update(gun_field_labels)


def merge_default_extra_fields(extra_fields: Sequence[str] | None) -> Tuple[str, ...]:
    """
    Ensure the default extra text fields are always present.

    :param extra_fields: Caller-provided sequence of extra field names.
    :type extra_fields: Sequence[str] | None
    :returns: Tuple containing the default field list plus any additional ones.
    :rtype: Tuple[str, ...]
    """
    ordered: list[str] = []
    seen: set[str] = set()

    for default_field in DEFAULT_EXTRA_TEXT_FIELDS:
        token = default_field.strip()
        if token and token not in seen:
            ordered.append(token)
            seen.add(token)

    if extra_fields:
        for extra_field_name in extra_fields:
            token = str(extra_field_name or "").strip()
            if token and token not in seen:
                ordered.append(token)
                seen.add(token)

    return tuple(ordered)


def format_extra_field(example: dict, field_name: str) -> str:
    """
    Return a labelled, human-readable representation of an extra field.

    :param example: Dataset row providing the raw field value.
    :type example: dict
    :param field_name: Name of the field to format.
    :type field_name: str
    :returns: Sphinx-friendly ``"Label: value"`` string or ``""`` when empty.
    :rtype: str
    """
    value = example.get(field_name)
    formatted = value_maps.format_field_value(field_name, value)
    if not formatted:
        return ""
    label = EXTRA_FIELD_LABELS.get(field_name)
    if not label:
        label = field_name.replace("_", " ").strip().capitalize()
    if field_name == "child18":
        lowered = formatted.lower()
        if lowered.startswith("no"):
            formatted = "no"
        elif "children" in lowered:
            formatted = "yes"
    return f"{label}: {formatted}"


__all__ = [
    "DEFAULT_EXTRA_TEXT_FIELDS",
    "EXTRA_FIELD_LABELS",
    "format_extra_field",
    "merge_default_extra_fields",
]
