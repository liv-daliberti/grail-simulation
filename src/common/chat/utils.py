#!/usr/bin/env python
"""Lightweight chat message helpers shared across modules.

These utilities avoid re-implementing simple message-scanning patterns in
multiple places (e.g., extracting the latest user prompt for logging).
"""

from __future__ import annotations

from typing import Mapping, Sequence


def latest_user_content(messages: Sequence[Mapping[str, str]]) -> str:
    """Return the most recent user message content from ``messages``.

    :param messages: Ordered chat message payloads.
    :returns: Stripped user content or an empty string when missing.
    """

    for message in reversed(messages):
        if (
            isinstance(message, Mapping)
            and message.get("role") == "user"
            and message.get("content")
        ):
            return str(message["content"]).strip()
    return ""


def first_system_content(messages: Sequence[Mapping[str, str]]) -> str:
    """Return the first system prompt content from ``messages`` when present.

    :param messages: Ordered chat message payloads.
    :returns: Stripped system prompt text or an empty string when absent.
    """

    for message in messages:
        if (
            isinstance(message, Mapping)
            and message.get("role") == "system"
            and message.get("content")
        ):
            return str(message["content"]).strip()
    return ""


__all__ = [
    "first_system_content",
    "latest_user_content",
]
