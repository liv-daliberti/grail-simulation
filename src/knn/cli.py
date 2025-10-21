"""Command-line interface for the refactored KNN baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .evaluate import run_eval
from .features import Word2VecConfig

DEFAULT_KNN_TEXT_FIELDS = (
    "pid1,pid2,ideo1,ideo2,pol_interest,religpew,educ,employ,child18,inputstate,"
    "freq_youtube,youtube_time,newsint,q31,participant_study,slate_source,"
    "minwage_text_w2,minwage_text_w1,mw_support_w2,mw_support_w1,minwage15_w2,"
    "minwage15_w1,mw_index_w2,mw_index_w1,gun_importance,gun_index,gun_enthusiasm,"
    "gun_identity"
)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the KNN baseline CLI.

    :returns: Configured :class:`argparse.ArgumentParser` instance with all KNN options.
    """

    parser = argparse.ArgumentParser(description="KNN baselines for GRAIL (TF-IDF / Word2Vec)")
    parser.add_argument(
        "--task",
        choices=["slate", "opinion"],
        default="slate",
        help="Select evaluation target: 'slate' for next-video recommendation or 'opinion' for post-study indices.",
    )
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
        "--word2vec-window",
        "--word2vec_window",
        type=int,
        default=Word2VecConfig().window,
        dest="word2vec_window",
        help="Context window size for Word2Vec training.",
    )
    parser.add_argument(
        "--word2vec-min-count",
        "--word2vec_min_count",
        type=int,
        default=Word2VecConfig().min_count,
        dest="word2vec_min_count",
        help="Minimum token frequency retained in the Word2Vec vocabulary.",
    )
    parser.add_argument(
        "--word2vec-epochs",
        "--word2vec_epochs",
        type=int,
        default=Word2VecConfig().epochs,
        dest="word2vec_epochs",
        help="Number of training epochs for Word2Vec.",
    )
    parser.add_argument(
        "--word2vec-workers",
        "--word2vec_workers",
        type=int,
        default=Word2VecConfig().workers,
        dest="word2vec_workers",
        help="Worker threads for Word2Vec training (set >1 for faster but non-deterministic runs).",
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
        default=DEFAULT_KNN_TEXT_FIELDS,
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
    parser.add_argument(
        "--opinion-studies",
        "--opinion_studies",
        default="",
        dest="opinion_studies",
        help="Comma-separated list of opinion study keys (study1,study2,study3). Defaults to all.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and execute the evaluation routine.

    :param argv: Optional list of command-line arguments. When ``None`` the values
        are read from :data:`sys.argv`.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if args.task == "slate":
        run_eval(args)
        return

    if args.task == "opinion":
        from .opinion import run_opinion_eval

        run_opinion_eval(args)
        return

    raise ValueError(f"Unsupported task '{args.task}'.")


if __name__ == "__main__":  # pragma: no cover
    main()
