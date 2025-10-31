#!/usr/bin/env python
"""Module entry point for ``python -m gpt4o.pipeline``."""

from . import main


def _run() -> None:
    """Invoke the package main to satisfy ``python -m`` expectations."""

    main()


if __name__ == "__main__":
    _run()
