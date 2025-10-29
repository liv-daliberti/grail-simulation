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

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common.cli.args import add_comma_separated_argument

from .config import DEFAULT_CACHE_DIR
from .evaluate import run_eval


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the GPT-4o baseline.

    :returns: Fully configured argument parser for the GPT-4o CLI.
    :rtype: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description="Evaluate GPT-4o on the GRAIL slate dataset.")
    parser.add_argument(
        "--eval_max",
        type=int,
        default=0,
        help="Limit evaluation examples (0 means evaluate every row).",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset path or Hugging Face dataset id (defaults to config.DATASET_NAME).",
    )
    parser.add_argument(
        "--issues",
        default="",
        help="Comma-separated list of issue labels to evaluate (defaults to all issues).",
    )
    add_comma_separated_argument(
        parser,
        flags="--studies",
        dest="studies",
        help_text=(
            "Comma-separated participant study identifiers to filter (defaults to all studies)."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=str(Path("models") / "gpt4o"),
        help="Directory for predictions and metrics.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files inside --out_dir.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to GPT-4o.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32,
        help="Maximum number of tokens in the model response.",
    )
    parser.add_argument(
        "--cache_dir",
        default=str(DEFAULT_CACHE_DIR),
        help="HF datasets cache directory.",
    )
    parser.add_argument(
        "--deployment",
        default="",
        help="Optional Azure deployment identifier (defaults to the config setting).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
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
