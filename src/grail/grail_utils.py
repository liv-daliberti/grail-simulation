"""Utility helpers shared across the GRAIL reward shaping modules."""

from __future__ import annotations

import re
from typing import Any, List, Optional

ANS_RE = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
IDX_ONLY = re.compile(r"^\s*(?:option\s*)?(\d+)\s*$", re.I)


def _completion_text(payload: Any) -> str:
    """Extract the assistant payload from chat-style or raw completion objects.

    :param payload: Chat completion response, message list, or raw string emitted by the model.
    :returns: Assistant message content with surrounding whitespace removed.
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        return str(payload.get("content", "")).strip()
    if isinstance(payload, list) and payload:
        for message in reversed(payload):
            if isinstance(message, dict) and "content" in message:
                content = str(message.get("content", "")).strip()
                if content:
                    return content
        joined_contents = [
            str(msg.get("content", "")).strip()
            for msg in payload
            if isinstance(msg, dict) and str(msg.get("content", "")).strip()
        ]
        if joined_contents:
            return " ".join(joined_contents)
    return str(payload)


def _parse_index_from_answer_block(text: str) -> Optional[int]:
    """Parse the integer embedded inside ``<answer>`` tags.

    :param text: Completion body that may contain an ``<answer>`` block with a numeric choice.
    :returns: Parsed integer index or ``None`` when the format is invalid.
    """
    match = ANS_RE.search(text or "")
    payload = (match.group(1).strip() if match else (text or "").strip())
    match_idx = IDX_ONLY.match(payload)
    if not match_idx:
        return None
    try:
        return int(match_idx.group(1))
    except (TypeError, ValueError):
        return None


def _ensure_list(value: Any, count: int) -> List[Any]:
    """Return ``value`` as a list, repeating scalars ``count`` times.

    :param value: Existing list or scalar to broadcast.
    :param count: Expected number of elements in the output list.
    :returns: List of length ``count`` with either the original list or repeated scalar.
    """
    if isinstance(value, list):
        return value
    return [value] * count


def _safe_int(value: Any, default: int = -1) -> int:
    """Cast ``value`` to ``int`` and fall back to ``default`` on failure.

    :param value: Value that should represent an integer.
    :param default: Fallback value returned when casting fails.
    :returns: Parsed integer or ``default`` when conversion raises.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "_completion_text",
    "_parse_index_from_answer_block",
    "_ensure_list",
    "_safe_int",
]
