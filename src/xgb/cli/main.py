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

"""Command-line interface for training and evaluating the XGBoost baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common.cli.args import add_comma_separated_argument
from common.cli.options import add_eval_arguments

from ..core.evaluate import run_eval
from .args_shared import add_sentence_transformer_args, add_word2vec_args

# Align default extra text fields with KNN to reduce signal disparity.
# These fields augment the base viewer_profile/state_text from prompt_docs.
DEFAULT_XGB_TEXT_FIELDS = (
    "pid1,pid2,ideo1,ideo2,pol_interest,religpew,educ,employ,child18,inputstate,"
    "freq_youtube,youtube_time,newsint,q31,participant_study,slate_source,"
    "minwage_text_w2,minwage_text_w1,mw_support_w2,mw_support_w1,minwage15_w2,"
    "minwage15_w1,mw_index_w2,mw_index_w1,gun_importance,gun_index,gun_enthusiasm,"
    "gun_identity"
)


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
    # Shared eval args (dataset, issues, eval_max, out_dir, cache_dir, overwrite, log-level)
    add_eval_arguments(
        parser,
        default_out_dir=str(Path("models") / "xgb"),
        default_cache_dir=str(Path.cwd() / "hf_cache"),
        include_llm_args=False,
        include_opinion_args=False,
        include_studies_filter=False,
        dataset_default="data/cleaned_grail",
        issues_default="",
        include_legacy_aliases=True,
    )
    parser.add_argument(
        "--extra_text_fields",
        default=DEFAULT_XGB_TEXT_FIELDS,
        help=(
            "Comma-separated additional columns appended to the prompt document. "
            "Defaults mirror KNN's extended text fields."
        ),
    )
    parser.add_argument(
        "--text_vectorizer",
        default="tfidf",
        choices=["tfidf", "word2vec", "sentence_transformer"],
        help="Feature extraction strategy for prompt documents.",
    )
    add_word2vec_args(parser, style="underscore")
    add_sentence_transformer_args(parser, style="underscore", normalize_default=True)
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
    # participant_studies flags retained below
    add_comma_separated_argument(
        parser,
        flags=("--participant-studies", "--participant_studies"),
        dest="participant_studies",
        help_text="Comma-separated participant study keys to evaluate (defaults to all).",
    )
    add_comma_separated_argument(
        parser,
        flags=("--train-participant-studies", "--train_participant_studies"),
        dest="train_participant_studies",
        help_text=(
            "Comma-separated participant study keys used for TRAINING. "
            "Defaults to --participant-studies when unset."
        ),
    )
    add_comma_separated_argument(
        parser,
        flags=("--eval-participant-studies", "--eval_participant_studies"),
        dest="eval_participant_studies",
        help_text=(
            "Comma-separated participant study keys used for EVALUATION. "
            "Defaults to --participant-studies when unset."
        ),
    )
    # out_dir/cache_dir/overwrite/log-level added via add_eval_arguments
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
