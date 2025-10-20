"""Command-line interface for the refactored KNN baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .evaluate import run_eval


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the KNN baseline CLI."""

    parser = argparse.ArgumentParser(description="TF-IDF KNN baseline for GRAIL")
    parser.add_argument(
        "--fit_index",
        action="store_true",
        help="Build KNN index from the train split before evaluation.",
    )
    parser.add_argument(
        "--save_index",
        default="",
        help="Directory to save the fitted TF-IDF index (npz + joblib).",
    )
    parser.add_argument(
        "--load_index",
        default="",
        help="Directory containing a previously saved TF-IDF index to reuse.",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=25,
        help="Primary number of neighbours to score per slate option.",
    )
    parser.add_argument(
        "--knn_k_sweep",
        default="1,2,3,4,5,10,25,50",
        help="Comma-separated list of additional k values to evaluate.",
    )
    parser.add_argument(
        "--knn_metric",
        default="l2",
        choices=["l2", "cosine"],
        help="Distance metric used to weight neighbour votes.",
    )
    parser.add_argument(
        "--knn_max_train",
        type=int,
        default=200_000,
        help="Maximum number of training rows to keep when building the index.",
    )
    parser.add_argument(
        "--knn_seed",
        type=int,
        default=42,
        help="Random seed applied when subsampling the train split.",
    )
    parser.add_argument(
        "--eval_max",
        type=int,
        default=0,
        help="Limit evaluation examples (0 means evaluate every row).",
    )
    parser.add_argument(
        "--out_dir",
        default=str(Path("models") / "knn"),
        help="Directory for predictions, metrics, and saved indices.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting metrics/prediction files in --out_dir.",
    )
    parser.add_argument(
        "--cache_dir",
        default=str(Path.cwd() / "hf_cache"),
        help="HF datasets cache directory.",
    )
    parser.add_argument(
        "--knn_text_fields",
        default="",
        help="Comma-separated extra columns to append to TF-IDF queries.",
    )
    parser.add_argument(
        "--dataset",
        default="data/cleaned_grail",
        help="Local dataset path (load_from_disk) or Hugging Face dataset id.",
    )
    parser.add_argument(
        "--issues",
        default="",
        help="Comma-separated list of issues to evaluate (defaults to all).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and execute the evaluation routine."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    run_eval(args)


if __name__ == "__main__":  # pragma: no cover
    main()
