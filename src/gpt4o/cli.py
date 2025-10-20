"""Command-line interface for the GPT-4o slate baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import DEFAULT_CACHE_DIR
from .evaluate import run_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GPT-4o on the GRAIL slate dataset.")
    parser.add_argument(
        "--eval_max",
        type=int,
        default=0,
        help="Limit evaluation examples (0 means evaluate every row).",
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
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    run_eval(args)


if __name__ == "__main__":  # pragma: no cover
    main()

