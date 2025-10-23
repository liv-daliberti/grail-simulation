"""Small IO helpers shared across pipeline implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


def load_metrics_json(path: Path) -> Mapping[str, object]:
    """Load a JSON metrics file and return its contents."""

    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_markdown_lines(path: Path, lines: Sequence[str]) -> None:
    """Persist ``lines`` as a Markdown document with a trailing newline."""

    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


__all__ = ["load_metrics_json", "write_markdown_lines"]
