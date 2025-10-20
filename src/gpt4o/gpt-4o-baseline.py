#!/usr/bin/env python3
"""Backwards-compatible entrypoint for the GPT-4o slate baseline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from gpt4o.cli import main


if __name__ == "__main__":
    main()
