"""Command-line interface for the refactored KNN baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .evaluate import run_eval


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the KNN baseline CLI."""

    parser = argparse.ArgumentParser(description="KNN baselines for GRAIL (TF-IDF / Word2Vec)")
    parser.add_argument(
        "--feature-space",
        "--feature_space",
        default="tfidf",
        choices=["tfidf", "word2vec"],
        dest="feature_space",
        help="Feature space used to build the KNN index.",
    )
    parser.add_argument(
        "--fit-index",
        "--fit_index",
        action="store_true",
        dest="fit_index",
        help="Build KNN index from the train split before evaluation.",
    )
    parser.add_argument(
        "--save-index",
        "--save_index",
        default="",
        dest="save_index",
        help="Directory to save the fitted index (per-issue subdirectories).",
    )
    parser.add_argument(
        "--load-index",
        "--load_index",
        default="",
        dest="load_index",
        help="Directory containing a previously saved index to reuse.",
    )
    parser.add_argument(
        "--knn-k",
        "--knn_k",
        type=int,
        default=25,
        dest="knn_k",
        help="Primary number of neighbours to score per slate option.",
    )
    parser.add_argument(
        "--knn-k-sweep",
        "--knn_k_sweep",
        default="1,2,3,4,5,10,25,50",
        dest="knn_k_sweep",
        help="Comma-separated list of additional k values to evaluate.",
    )
    parser.add_argument(
        "--knn-metric",
        "--knn_metric",
        default="l2",
        choices=["l2", "cosine"],
        dest="knn_metric",
        help="Distance metric used to weight neighbour votes.",
    )
    parser.add_argument(
        "--knn-max-train",
        "--knn_max_train",
        type=int,
        default=200_000,
        dest="knn_max_train",
        help="Maximum number of training rows to keep when building the index.",
    )
    parser.add_argument(
        "--knn-seed",
        "--knn_seed",
        type=int,
        default=42,
        dest="knn_seed",
        help="Random seed applied when subsampling the train split.",
    )
    parser.add_argument(
        "--eval-max",
        "--eval_max",
        type=int,
        default=0,
        dest="eval_max",
        help="Limit evaluation examples (0 means evaluate every row).",
    )
    parser.add_argument(
        "--word2vec-size",
        "--word2vec_size",
        type=int,
        default=256,
        dest="word2vec_size",
        help="Embedding size when using the Word2Vec feature space.",
    )
    parser.add_argument(
        "--word2vec-model-dir",
        "--word2vec_model_dir",
        default="",
        dest="word2vec_model_dir",
        help="Optional directory to store/load Word2Vec models (defaults to models/knn_word2vec).",
    )
    parser.add_argument(
        "--train-curve-max",
        "--train_curve_max",
        type=int,
        default=0,
        dest="train_curve_max",
        help="Limit the number of training examples evaluated for curve metrics (0 means use all).",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        default=str(Path("models") / "knn"),
        dest="out_dir",
        help="Directory for predictions, metrics, and saved indices.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting metrics/prediction files in --out_dir.",
    )
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        default=str(Path.cwd() / "hf_cache"),
        dest="cache_dir",
        help="HF datasets cache directory.",
    )
    parser.add_argument(
        "--knn-text-fields",
        "--knn_text_fields",
        default="",
        dest="knn_text_fields",
        help="Comma-separated extra columns to append to the viewer prompt text.",
    )
    parser.add_argument(
        "--dataset",
        default="data/cleaned_grail",
        help="Local dataset path (load_from_disk) or Hugging Face dataset id.",
    )
    parser.add_argument(
        "--issues",
        "--issue",
        default="",
        dest="issues",
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
