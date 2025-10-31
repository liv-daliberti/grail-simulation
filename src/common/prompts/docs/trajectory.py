"""Helpers for parsing trajectory metadata embedded in prompt records."""

from __future__ import annotations

import json
from typing import Mapping, Sequence


def load_trajectory_entries(payload: object) -> list[Mapping[str, object]]:
    """
    Return sanitized trajectory entries extracted from ``payload``.

    :param payload: Raw trajectory JSON blob or mapping supplied by datasets.
    :type payload: object
    :returns: List containing mapping objects for each trajectory event.
    :rtype: list[Mapping[str, object]]
    """
    if isinstance(payload, str) and payload.strip():
        try:
            data = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):  # pragma: no cover - defensive
            return []
    elif isinstance(payload, Mapping):
        data = payload
    else:
        return []

    if not isinstance(data, Mapping):
        return []
    rows = data.get("order") or data.get("videos") or data.get("history") or []
    if not isinstance(rows, Sequence):
        return []

    entries: list[Mapping[str, object]] = []
    for entry in rows:
        if isinstance(entry, Mapping):
            entries.append(entry)
    return entries


__all__ = ["load_trajectory_entries"]
