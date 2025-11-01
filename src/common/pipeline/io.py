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

"""Small JSON/Markdown helpers reused across pipeline report builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def load_metrics_json(path: Path) -> Mapping[str, object]:
    """
    Load a metrics dictionary from ``path``.

    :param path: JSON file containing previously computed metrics.
    :returns: Parsed metrics mapping.
    :raises FileNotFoundError: If the path does not exist.
    :raises json.JSONDecodeError: If the file cannot be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metrics_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """
    Serialise a metrics dictionary to ``path`` using newline-terminated JSON.

    :param path: Destination metrics file.
    :param payload: Dictionary of metrics to persist.
    :param indent: JSON indentation level applied when serialising.
    :param sort_keys: Whether JSON keys should be sorted alphabetically.
    :returns: ``None``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=sort_keys)
        handle.write("\n")


def write_segmented_markdown_log(
    path: Path,
    *,
    title: str,
    entries: Sequence[str],
    separator: str = "\n---\n\n",
) -> None:
    """
    Write Markdown log entries separated by ``separator`` under a shared heading.

    :param path: Destination Markdown log file.
    :param title: Heading written at the top of the file.
    :param entries: Iterable of entry payloads appended to the log.
    :param separator: Separator inserted between entries. Defaults to ``\"\\n---\\n\\n\"``.
    :returns: ``None``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        for entry in entries:
            handle.write(entry)
            if not entry.endswith("\n"):
                handle.write("\n")
            handle.write(separator)


def write_markdown_lines(path: Path, lines: Sequence[str]) -> None:
    """
    Write ``lines`` to ``path`` ensuring the file ends with a newline.

    :param path: Destination Markdown file.
    :param lines: Iterable of Markdown lines to write.
    :returns: ``None``.
    """
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def iter_jsonl_rows(path: Path, *, ignore_errors: bool = True) -> Iterable[Mapping[str, object]]:
    """
    Yield JSON objects from a JSONL file one by one.

    - Skips blank lines.
    - When ``ignore_errors`` is ``True`` (default), skips malformed lines.

    :param path: Path to the JSONL file.
    :param ignore_errors: Whether to skip lines that fail JSON decoding.
    :yields: Mapping rows parsed from the file.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    if ignore_errors:
                        continue
                    raise
                if isinstance(obj, Mapping):
                    yield obj
    except FileNotFoundError:
        # Nothing to yield when the file is missing.
        return


def write_jsonl_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    ensure_ascii: bool = False,
) -> None:
    """
    Write a sequence of mappings to ``path`` as JSON Lines.

    :param path: Destination file.
    :param rows: Sequence of mapping objects to serialise.
    :param ensure_ascii: Forwarded to :func:`json.dumps` (defaults to ``False``).
    :returns: ``None``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in rows:
            handle.write(json.dumps(entry, ensure_ascii=ensure_ascii))
            handle.write("\n")


__all__ = [
    "load_metrics_json",
    "write_markdown_lines",
    "write_metrics_json",
    "write_segmented_markdown_log",
    "iter_jsonl_rows",
    "write_jsonl_rows",
]
