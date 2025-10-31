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

"""Command-line interface for evaluating the GPT-4o baseline model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from importlib import import_module

from ..core.config import DEFAULT_CACHE_DIR
from ..core.evaluate import run_eval

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[3] / "models" / "gpt-4o"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the GPT-4o baseline.

    :returns: Fully configured argument parser for the GPT-4o CLI.
    :rtype: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description="Evaluate GPT-4o on the GRAIL slate dataset.")
    add_opts = import_module("common.cli.options")
    add_opts.add_eval_arguments(
        parser,
        default_out_dir=str(DEFAULT_OUT_DIR),
        default_cache_dir=str(DEFAULT_CACHE_DIR),
        include_llm_args=True,
        include_opinion_args=True,
        include_studies_filter=True,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and execute the evaluation routine.

    :param argv: Optional argument list to parse instead of ``sys.argv``.
    :type argv: list[str] | None
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    run_eval(args)


if __name__ == "__main__":  # pragma: no cover
    main()
