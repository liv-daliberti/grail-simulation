"""Command-line interface for the refactored KNN baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class CLIArgs:
    dataset: str
    cache_dir: str
    feature_space: str
    vector_size: int
    window: int
    min_count: int
    run_mode: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refactored KNN baseline")
    parser.add_argument("--dataset", default="data/cleaned_grail", help="Dataset path or HF repo")
    parser.add_argument("--cache-dir", default=".cache", help="Where to cache dataset downloads")
    parser.add_argument(
        "--feature-space",
        choices=["tfidf", "word2vec"],
        default="word2vec",
        help="Feature space to use for the KNN index",
    )
    parser.add_argument("--vector-size", type=int, default=256, help="Word2Vec vector size")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec window size")
    parser.add_argument("--min-count", type=int, default=2, help="Word2Vec minimum token frequency")
    parser.add_argument(
        "--run-mode",
        choices=["fit", "predict"],
        default="fit",
        help="Whether to fit a new index or run predictions with an existing one",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> CLIArgs:
    parser = build_parser()
    args = parser.parse_args(argv)
    return CLIArgs(
        dataset=args.dataset,
        cache_dir=args.cache_dir,
        feature_space=args.feature_space,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        run_mode=args.run_mode,
    )


def main() -> None:  # pragma: no cover - placeholder
    raise NotImplementedError("CLI entry point will be implemented during refactor")


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()
