"""Command-line interface for the XGBoost slate baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .evaluate import run_eval


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the XGBoost baseline CLI."""

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
        default="hist",
        help="Tree construction algorithm used by XGBoost.",
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
    parser.add_argument(
        "--cache_dir",
        default=str(Path.cwd() / "hf_cache"),
        help="HF datasets cache directory.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(Path("models") / "xgboost"),
        help="Directory for predictions, metrics, and saved models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting metrics/prediction files in --out_dir.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and execute the evaluation routine."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    run_eval(args)


if __name__ == "__main__":  # pragma: no cover
    main()
