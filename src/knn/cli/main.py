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

"""Command-line interface for the Grail Simulation KNN baselines."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common.cli.args import add_comma_separated_argument
from common.cli.options import add_overwrite_argument, add_eval_arguments

from ..core.evaluate import run_eval
from ..core.features import Word2VecConfig
from .utils import add_sentence_transformer_normalize_flags

DEFAULT_KNN_TEXT_FIELDS = (
    "pid1,pid2,ideo1,ideo2,pol_interest,religpew,educ,employ,child18,inputstate,"
    "freq_youtube,youtube_time,newsint,q31,participant_study,slate_source,"
    "minwage_text_w2,minwage_text_w1,mw_support_w2,mw_support_w1,minwage15_w2,"
    "minwage15_w1,mw_index_w2,mw_index_w1,gun_importance,gun_index,gun_enthusiasm,"
    "gun_identity"
)

def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser that drives the refactored KNN workflows.

    The parser exposes switches for both slate (next-video) and opinion
    prediction flows, including feature-space selection, pre-computed index
    reuse, bootstrap configuration, and Word2Vec/SentenceTransformer tuning.

    :returns: Parser pre-populated with all supported CLI arguments for the KNN baseline.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="KNN baselines for GRAIL (TF-IDF / Word2Vec)")
    parser.add_argument(
        "--task",
        choices=["slate", "opinion"],
        default="slate",
        help=(
            "Select evaluation target: 'slate' for next-video recommendation or "
            "'opinion' for post-study indices."
        ),
    )
    parser.add_argument(
        "--feature-space",
        "--feature_space",
        default="tfidf",
        choices=["tfidf", "word2vec", "sentence_transformer"],
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
        "--knn-k-select",
        "--knn_k_select",
        "--k-select-method",
        "--k_select_method",
        choices=["max", "elbow"],
        default="max",
        dest="k_select_method",
        help=(
            "Method for selecting best k: 'max' chooses the accuracy-maximising k "
            "(eligible-only accuracy), 'elbow' applies a diminishing-returns heuristic."
        ),
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
    # Shared eval args (dataset, issues, eval_max, out_dir, cache_dir, overwrite)
    add_eval_arguments(
        parser,
        default_out_dir=str(Path("models") / "knn"),
        default_cache_dir=str(Path.cwd() / "hf_cache"),
        include_llm_args=False,
        include_opinion_args=False,
        include_studies_filter=False,
        dataset_default="data/cleaned_grail",
        issues_default="",
        include_legacy_aliases=True,
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
        help=(
            "Optional directory to store/load Word2Vec models "
            "(defaults to models/knn/next_video/word2vec_models)."
        ),
    )
    parser.add_argument(
        "--sentence-transformer-model",
        "--sentence_transformer_model",
        default="sentence-transformers/all-mpnet-base-v2",
        dest="sentence_transformer_model",
        help=(
            "SentenceTransformer model identifier when using the "
            "sentence_transformer feature space."
        ),
    )
    parser.add_argument(
        "--sentence-transformer-device",
        "--sentence_transformer_device",
        default="",
        dest="sentence_transformer_device",
        help="Optional device string (cpu, cuda) forwarded to SentenceTransformer.",
    )
    parser.add_argument(
        "--sentence-transformer-batch-size",
        "--sentence_transformer_batch_size",
        type=int,
        default=32,
        dest="sentence_transformer_batch_size",
        help="Batch size applied during sentence-transformer encoding.",
    )
    add_sentence_transformer_normalize_flags(parser, legacy_aliases=True)
    parser.add_argument(
        "--train-curve-max",
        "--train_curve_max",
        type=int,
        default=0,
        dest="train_curve_max",
        help="Limit the number of training examples evaluated for curve metrics (0 means use all).",
    )
    # out_dir/cache_dir/overwrite added via add_eval_arguments
    parser.add_argument(
        "--knn-text-fields",
        "--knn_text_fields",
        default=DEFAULT_KNN_TEXT_FIELDS,
        dest="knn_text_fields",
        help="Comma-separated extra columns to append to the viewer prompt text.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        "--bootstrap_replicates",
        type=int,
        default=500,
        dest="bootstrap_replicates",
        help=(
            "Number of bootstrap replicates used to compute uncertainty intervals "
            "(0 disables bootstrap)."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        "--bootstrap_seed",
        type=int,
        default=2024,
        dest="bootstrap_seed",
        help="Random seed for bootstrap resampling.",
    )
    # 'dataset' and 'issues' added via add_eval_arguments
    parser.add_argument(
        "--train-issues",
        "--train_issues",
        default="",
        dest="train_issues",
        help=(
            "Optional comma-separated list of issues used to TRAIN the KNN index. "
            "Defaults to --issues when unset."
        ),
    )
    parser.add_argument(
        "--eval-issues",
        "--eval_issues",
        default="",
        dest="eval_issues",
        help=(
            "Optional comma-separated list of issues used for EVALUATION. "
            "Defaults to --issues when unset."
        ),
    )
    add_comma_separated_argument(
        parser,
        flags=("--participant-studies", "--participant_studies"),
        dest="participant_studies",
        help_text=(
            "Comma-separated list of participant study keys (e.g. study1,study2). "
            "When supplied, next-video evaluation filters to those studies."
        ),
    )
    parser.add_argument(
        "--train-participant-studies",
        "--train_participant_studies",
        default="",
        dest="train_participant_studies",
        help=(
            "Optional comma-separated list of participant study keys used when "
            "FITTING the index. Defaults to --participant-studies when unset."
        ),
    )
    parser.add_argument(
        "--eval-participant-studies",
        "--eval_participant_studies",
        default="",
        dest="eval_participant_studies",
        help=(
            "Optional comma-separated list of participant study keys used for "
            "EVALUATION. Defaults to --participant-studies when unset."
        ),
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
    """
    Dispatch execution for the CLI by parsing arguments and invoking the correct runner.

    :param argv: Optional argument vector supplied for testing. When ``None``,
        :data:`sys.argv` is used verbatim.
    :type argv: list[str] | None
    :returns: ``None``. The function delegates work to the slate or opinion pipeline.
    :rtype: None
    :raises ValueError: If ``--task`` is neither ``slate`` nor ``opinion``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if args.task == "slate":
        run_eval(args)
        return

    if args.task == "opinion":
        from ..core.opinion import run_opinion_eval  # pylint: disable=import-outside-toplevel

        run_opinion_eval(args)
        return

    raise ValueError(f"Unsupported task '{args.task}'.")

if __name__ == "__main__":  # pragma: no cover
    main()
