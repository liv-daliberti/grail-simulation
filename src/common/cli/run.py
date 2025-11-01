#!/usr/bin/env python
"""Tiny helpers to standardise CLI entrypoints across projects."""

from __future__ import annotations

import argparse
import logging
from typing import Callable


def configure_logging(level_name: str | None) -> None:
    """Initialise basic logging with a consistent format.

    :param level_name: Desired log level name (e.g. "INFO", "DEBUG").
    """
    level = getattr(logging, (level_name or "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")


def run_main(
    build_parser: Callable[[], argparse.ArgumentParser],
    runner: Callable[[argparse.Namespace], None],
    argv: list[str] | None = None,
) -> None:
    """Parse args, set up logging, and run the provided ``runner``.

    :param build_parser: Factory returning a configured ``ArgumentParser``.
    :param runner: Callable invoked with the parsed arguments.
    :param argv: Optional argument vector override for testing.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "log_level", None))
    runner(args)
