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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common.cli_args import add_comma_separated_argument, add_sentence_transformer_normalise_flags
from common.cli_options import add_log_level_argument, add_overwrite_argument

from .evaluate import run_eval


def build_parser() -> argparse.ArgumentParser:
    """
    Return the argument parser for the XGBoost baseline CLI.

    :returns: Configured argument parser exposing training and evaluation options.
    :rtype: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description="XGBoost slate baseline for GRAIL")
    parser.add_argument(
        "--fit_model",
        action="store_true",
        help="Train an XGBoost model on the train split before evaluation.",
    )
    parser.add_argument(
        "--save_model",
        default="",
        help="Directory to save the fitted XGBoost model bundle.",
    )
    parser.add_argument(
        "--load_model",
        default="",
        help="Directory containing a previously saved XGBoost model bundle.",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=200_000,
        help="Maximum number of training rows to keep when fitting the model (0 means all).",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=200_000,
        help="Maximum number of TF-IDF features (0 means unlimited).",
    )
    parser.add_argument(
        "--seed",
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
        "--extra_text_fields",
        default="",
        help="Comma-separated additional columns appended to the prompt document.",
    )
    parser.add_argument(
        "--text_vectorizer",
        default="tfidf",
        choices=["tfidf", "word2vec", "sentence_transformer"],
        help="Feature extraction strategy for prompt documents.",
    )
    parser.add_argument(
        "--word2vec_size",
        type=int,
        default=256,
        help="Word2Vec vector dimensionality when using the word2vec feature space.",
    )
    parser.add_argument(
        "--word2vec_window",
        type=int,
        default=5,
        help="Window size used during Word2Vec training.",
    )
    parser.add_argument(
        "--word2vec_min_count",
        type=int,
        default=2,
        help="Minimum token frequency retained in the Word2Vec vocabulary.",
    )
    parser.add_argument(
        "--word2vec_epochs",
        type=int,
        default=10,
        help="Number of Word2Vec training epochs.",
    )
    parser.add_argument(
        "--word2vec_workers",
        type=int,
        default=1,
        help="Worker threads allocated to Word2Vec training.",
    )
    parser.add_argument(
        "--word2vec_model_dir",
        default="",
        help="Directory to persist trained Word2Vec models (optional).",
    )
    parser.add_argument(
        "--sentence_transformer_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help=(
            "SentenceTransformer model identifier used when "
            "--text_vectorizer=sentence_transformer."
        ),
    )
    parser.add_argument(
        "--sentence_transformer_device",
        default=None,
        help="PyTorch device string passed to SentenceTransformer (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--sentence_transformer_batch_size",
        type=int,
        default=32,
        help="Batch size applied during sentence-transformer encoding.",
    )
    add_sentence_transformer_normalise_flags(
        parser,
        dest="sentence_transformer_normalize",
        enable_flags=("--sentence_transformer_normalize",),
        disable_flags=("--sentence_transformer_no_normalize",),
        enable_help="L2-normalise sentence-transformer embeddings (default).",
        disable_help="Disable L2-normalisation for sentence-transformer embeddings.",
    )
    parser.add_argument(
        "--xgb_learning_rate",
        type=float,
        default=0.1,
        help="XGBoost learning rate (eta).",
    )
    parser.add_argument(
        "--xgb_max_depth",
        type=int,
        default=6,
        help="Maximum tree depth.",
    )
    parser.add_argument(
        "--xgb_n_estimators",
        type=int,
        default=300,
        help="Number of boosting rounds.",
    )
    parser.add_argument(
        "--xgb_subsample",
        type=float,
        default=0.8,
        help="Subsample ratio of the training instances.",
    )
    parser.add_argument(
        "--xgb_colsample_bytree",
        type=float,
        default=0.8,
        help="Subsample ratio of columns when constructing each tree.",
    )
    parser.add_argument(
        "--xgb_tree_method",
        default="gpu_hist",
        help="Tree construction algorithm used by XGBoost (default: gpu_hist).",
    )
    parser.add_argument(
        "--xgb_reg_lambda",
        type=float,
        default=1.0,
        help="L2 regularisation term on weights.",
    )
    parser.add_argument(
        "--xgb_reg_alpha",
        type=float,
        default=0.0,
        help="L1 regularisation term on weights.",
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
    add_comma_separated_argument(
        parser,
        flags=("--participant-studies", "--participant_studies"),
        dest="participant_studies",
        help_text="Comma-separated participant study keys to evaluate (defaults to all).",
    )
    parser.add_argument(
        "--cache_dir",
        default=str(Path.cwd() / "hf_cache"),
        help="HF datasets cache directory.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(Path("models") / "xgb"),
        help="Directory for predictions, metrics, and saved models.",
    )
    add_overwrite_argument(parser)
    add_log_level_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Parse CLI arguments and execute the evaluation routine.

    :param argv: Optional override for command-line arguments.
    :type argv: list[str], optional
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
