#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for rendering Markdown report tables."""

from __future__ import annotations

from typing import Iterable, Sequence


def normalise_field_values(
    values: Iterable[object] | None,
    *,
    default: Sequence[str],
) -> tuple[str, ...]:
    """
    Canonicalise an iterable of field names.

    Whitespace is stripped, duplicates are removed, and the ``default`` set is used when
    the iterable is empty or ``None``.
    """

    if values is None:
        return tuple(default)

    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw or "").strip()
        if not token or token in seen:
            continue
        ordered.append(token)
        seen.add(token)

    return tuple(ordered) if ordered else tuple(default)


def format_field_list(fields: Sequence[str]) -> str:
    """Render a comma-separated field list with inline code formatting."""

    if not fields:
        return "—"
    return ", ".join(f"`{field}`" for field in fields)


def render_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> list[str]:
    """Render a GitHub-flavoured Markdown table."""

    if not rows:
        return ["No entries recorded.", ""]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def append_markdown_table(
    lines: list[str],
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    empty_message: str,
) -> None:
    """Append a titled markdown table (or fallback message) to ``lines``."""

    lines.append(title)
    lines.append("")
    if rows:
        lines.extend(render_markdown_table(headers, rows))
    else:
        lines.append(empty_message)
        lines.append("")
