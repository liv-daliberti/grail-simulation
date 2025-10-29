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

"""Command-line helpers for the Grail Simulation XGBoost pipeline.

Provides argument parsing, default-directory resolution, and sweep grid
builders consumed by ``xgb.pipeline`` and its orchestration helpers.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import logging
import os
from itertools import product
from pathlib import Path
from typing import List, Sequence, Tuple

from common.cli.args import add_comma_separated_argument, add_sentence_transformer_normalise_flags
from common.cli.options import (
    add_jobs_argument,
    add_log_level_argument,
    add_overwrite_argument,
    add_stage_arguments,
)

from ..core.data import issues_in_dataset, load_dataset_source
from ..cli import DEFAULT_XGB_TEXT_FIELDS
from ..core.opinion import DEFAULT_SPECS
from .context import StudySpec, SweepConfig

LOGGER = logging.getLogger("xgb.pipeline.cli")


def _parse_args(argv: Sequence[str] | None) -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse pipeline command-line arguments and collect passthrough flags.

    :param argv: Optional override for CLI arguments (defaults to ``sys.argv[1:]``).
    :type argv: Sequence[str] | None
    :returns: Tuple of ``(namespace, extra_args)`` where ``extra_args`` are forwarded
        to downstream commands.
    :rtype: Tuple[argparse.Namespace, List[str]]
    :raises ValueError: If an unknown pipeline task is requested.
    """
    # pylint: disable=too-many-statements

    parser = argparse.ArgumentParser(
        description="Full XGBoost baseline pipeline (sweeps, selection, reports)."
    )
    parser.add_argument("--dataset", default=None, help="Dataset path or HF dataset id.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HF datasets cache directory (default: <repo>/.cache/huggingface/xgb).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for final next-video and opinion artefacts (default: <repo>/models/xgb).",
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help="Directory for hyper-parameter sweep outputs (default: <out-dir>/next_video/sweeps).",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directory receiving Markdown reports (default: <repo>/reports/xgb).",
    )
    parser.add_argument(
        "--issues",
        default="",
        help="Comma-separated list of issues to evaluate (defaults to all issues).",
    )
    add_comma_separated_argument(
        parser,
        flags="--studies",
        dest="studies",
        help_text="Comma-separated opinion study keys (defaults to all studies).",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated pipeline tasks to execute (next_video,opinion).",
    )
    parser.add_argument(
        "--extra-text-fields",
        default=DEFAULT_XGB_TEXT_FIELDS,
        help=(
            "Comma-separated extra text fields appended to prompt documents. "
            "Defaults mirror the xgb CLI extended text fields."
        ),
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=200_000,
        help="Maximum training rows sampled during slate model fitting (0 keeps all).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=200_000,
        help="Maximum TF-IDF features (0 keeps all).",
    )
    parser.add_argument(
        "--eval-max",
        type=int,
        default=0,
        help="Limit evaluation rows (0 processes all).",
    )
    parser.add_argument(
        "--opinion-max-participants",
        type=int,
        default=0,
        help="Optional cap on participants per opinion study (0 keeps all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed shared across training stages.",
    )
    parser.add_argument(
        "--tree-method",
        default="gpu_hist",
        help="Tree construction algorithm passed to XGBoost (default: gpu_hist).",
    )
    add_jobs_argument(parser)
    parser.add_argument(
        "--learning-rate-grid",
        default="0.03,0.05,0.1",
        help="Comma-separated learning rates explored during sweeps.",
    )
    parser.add_argument(
        "--max-depth-grid",
        default="3,4",
        help="Comma-separated integer depths explored during sweeps.",
    )
    parser.add_argument(
        "--n-estimators-grid",
        default="200,300,400",
        help="Comma-separated boosting round counts explored during sweeps.",
    )
    parser.add_argument(
        "--subsample-grid",
        default="0.75,0.9",
        help="Comma-separated subsample ratios explored during sweeps.",
    )
    parser.add_argument(
        "--colsample-grid",
        default="0.8",
        help="Comma-separated column subsample ratios explored during sweeps.",
    )
    parser.add_argument(
        "--reg-lambda-grid",
        default="0.5,1.0",
        help="Comma-separated L2 regularisation weights explored during sweeps.",
    )
    parser.add_argument(
        "--reg-alpha-grid",
        default="0.0",
        help="Comma-separated L1 regularisation weights explored during sweeps.",
    )
    parser.add_argument(
        "--text-vectorizer-grid",
        default="tfidf",
        help="Comma-separated list of text vectorisers explored during sweeps.",
    )
    parser.add_argument(
        "--word2vec-size",
        type=int,
        default=256,
        help="Word2Vec vector size applied when evaluating the word2vec feature space.",
    )
    parser.add_argument(
        "--word2vec-window",
        type=int,
        default=5,
        help="Word2Vec context window size during training.",
    )
    parser.add_argument(
        "--word2vec_min_count",
        type=int,
        default=2,
        help="Minimum token frequency retained in the Word2Vec vocabulary.",
    )
    parser.add_argument(
        "--word2vec-epochs",
        type=int,
        default=10,
        help="Number of epochs used when training Word2Vec embeddings.",
    )
    parser.add_argument(
        "--word2vec-workers",
        type=int,
        default=1,
        help="Worker threads allocated to Word2Vec training.",
    )
    parser.add_argument(
        "--word2vec-model-dir",
        default="",
        help="Optional directory where Word2Vec models should be stored.",
    )
    parser.add_argument(
        "--sentence-transformer-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help=(
            "SentenceTransformer model evaluated when using the "
            "sentence_transformer feature space."
        ),
    )
    parser.add_argument(
        "--sentence-transformer-device",
        default="",
        help="Optional device string (cpu/cuda) forwarded to SentenceTransformer.",
    )
    parser.add_argument(
        "--sentence-transformer-batch-size",
        type=int,
        default=32,
        help="Encoding batch size for sentence-transformer embeddings.",
    )
    add_sentence_transformer_normalise_flags(parser)
    parser.add_argument(
        "--save-model-dir",
        default=None,
        help="Optional directory used to persist the final slate models.",
    )
    add_log_level_argument(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the planned actions without executing sweeps or evaluations.",
    )
    add_overwrite_argument(parser)
    parser.add_argument(
        "--reuse-sweeps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse cached sweep artefacts when available "
            "(disabled by default; pass --reuse-sweeps to enable)."
        ),
    )
    parser.add_argument(
        "--reuse-final",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Reuse cached finalize-stage artefacts when available "
            "(use --no-reuse-final to force recomputation)."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow finalize/report stages to proceed with partial sweeps or metrics "
            "(use --no-allow-incomplete to require completeness)."
        ),
    )
    add_stage_arguments(parser)

    parsed, extra = parser.parse_known_args(argv)
    tasks_raw = parsed.tasks or os.environ.get("XGB_PIPELINE_TASKS", "")
    raw_tokens = [token.lower() for token in _split_tokens(tasks_raw)]
    normalized_tasks: set[str] = set()
    invalid_tokens: List[str] = []
    for token in raw_tokens:
        if token in {"next_video", "next-video", "next", "slate"}:
            normalized_tasks.add("next_video")
        elif token in {"opinion", "opinion_stage"}:
            normalized_tasks.add("opinion")
        else:
            invalid_tokens.append(token)
    if invalid_tokens:
        raise ValueError(
            "Unknown pipeline task(s): "
            + ", ".join(sorted(invalid_tokens))
            + ". Expected next_video and/or opinion."
        )
    if not normalized_tasks:
        normalized_tasks.update({"next_video", "opinion"})
    parsed.tasks_tokens = sorted(normalized_tasks)
    parsed.run_next_video = "next_video" in normalized_tasks
    parsed.run_opinion = "opinion" in normalized_tasks
    return parsed, list(extra)


def _repo_root() -> Path:
    """
    Determine the repository root relative to this module.

    :returns: Absolute path to the repository root.
    :rtype: Path
    """

    return Path(__file__).resolve().parents[3]


def _default_out_dir(root: Path) -> Path:
    """
    Return the default directory storing XGBoost artefacts.

    :param root: Repository root path.
    :type root: Path
    :returns: Path to ``models/xgb`` beneath ``root``.
    :rtype: Path
    """

    return root / "models" / "xgb"


def _default_cache_dir(root: Path) -> Path:
    """
    Return the default HuggingFace cache directory for the pipeline.

    :param root: Repository root path.
    :type root: Path
    :returns: Cache directory path beneath ``.cache/huggingface/xgb``.
    :rtype: Path
    """

    return root / ".cache" / "huggingface" / "xgb"


def _default_reports_dir(root: Path) -> Path:
    """
    Return the directory receiving generated Markdown reports.

    :param root: Repository root path.
    :type root: Path
    :returns: Path to ``reports/xgb`` beneath ``root``.
    :rtype: Path
    """

    return root / "reports" / "xgb"


def _split_tokens(raw: str) -> List[str]:
    """
    Split comma-delimited CLI input into trimmed tokens.

    :param raw: Raw comma-separated string.
    :type raw: str
    :returns: List of trimmed, non-empty tokens.
    :rtype: List[str]
    """

    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _sanitize_token(value: str) -> str:
    """
    Sanitise a token for filesystem-safe usage.

    :param value: Original token value.
    :type value: str
    :returns: Sanitised token suitable for directory names.
    :rtype: str
    """

    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(".", "p")
        .replace(" ", "_")
    )


def _load_dataset_source_or_none(
    *,
    dataset: str,
    cache_dir: str,
    allow_incomplete: bool,
):
    """
    Attempt to load the dataset source, returning ``None`` when allowed.

    :param dataset: Dataset identifier or path.
    :type dataset: str
    :param cache_dir: Cache location passed to :func:`load_dataset_source`.
    :type cache_dir: str
    :param allow_incomplete: Flag indicating whether failures should be tolerated.
    :type allow_incomplete: bool
    :returns: Dataset object or ``None`` when unavailable and ``allow_incomplete`` is true.
    :rtype: Any
    :raises ImportError: Re-raised when dataset loading fails and incompleteness is disallowed.
    :raises FileNotFoundError: Re-raised when the dataset path is missing and incompleteness
        is disallowed.
    """

    try:
        return load_dataset_source(dataset, cache_dir)
    except ImportError as exc:
        if allow_incomplete:
            LOGGER.warning(
                (
                    "Unable to load dataset '%s' (%s). Falling back to default study "
                    "specs because allow-incomplete mode is enabled."
                ),
                dataset,
                exc,
            )
            return None
        raise
    except FileNotFoundError as exc:
        if allow_incomplete:
            LOGGER.warning(
                (
                    "Dataset path '%s' missing (%s). Proceeding with default study specs "
                    "because allow-incomplete mode is enabled."
                ),
                dataset,
                exc,
            )
            return None
        raise


def _filter_specs_by_issue(
    specs: Sequence[StudySpec],
    requested_issues: Sequence[str],
    available_issues: Sequence[str],
) -> List[StudySpec]:
    """
    Filter study specifications by requested issues.

    :param specs: Collection of available study specifications.
    :type specs: Sequence[~common.pipeline.types.StudySpec]
    :param requested_issues: Issues provided via CLI tokens.
    :type requested_issues: Sequence[str]
    :param available_issues: Issues known to exist in the dataset.
    :type available_issues: Sequence[str]
    :returns: Filtered list of study specifications.
    :rtype: List[~common.pipeline.types.StudySpec]
    :raises ValueError: If any requested issue is unknown.
    """

    issue_filter = {token for token in requested_issues if token and token.lower() != "all"}
    if not issue_filter:
        return list(specs)
    missing_issues = sorted(set(issue_filter) - set(available_issues))
    if missing_issues:
        raise ValueError(f"Unknown issues requested: {', '.join(missing_issues)}")
    return [spec for spec in specs if spec.issue in issue_filter]


def _filter_specs_by_study(
    specs: Sequence[StudySpec],
    requested_studies: Sequence[str],
) -> List[StudySpec]:
    """
    Reorder and filter study specifications by requested study keys.

    :param specs: Available study specifications.
    :type specs: Sequence[~common.pipeline.types.StudySpec]
    :param requested_studies: Study keys provided by the user.
    :type requested_studies: Sequence[str]
    :returns: Filtered and ordered study specifications.
    :rtype: List[~common.pipeline.types.StudySpec]
    :raises ValueError: If any requested study key is unknown.
    """

    study_filter = [token for token in requested_studies if token and token.lower() != "all"]
    if not study_filter:
        return list(specs)
    key_map = {spec.key: spec for spec in specs}
    missing_studies = [token for token in study_filter if token not in key_map]
    if missing_studies:
        raise ValueError(f"Unknown studies requested: {', '.join(sorted(missing_studies))}")
    ordered: List[StudySpec] = []
    for token in study_filter:
        spec = key_map[token]
        if spec not in ordered:
            ordered.append(spec)
    return ordered


def _build_sweep_configs(args: argparse.Namespace) -> List[SweepConfig]:
    """
    Construct hyper-parameter sweep configurations derived from CLI grids.

    :param args: Parsed CLI namespace.
    :type args: argparse.Namespace
    :returns: List of sweep configurations spanning vectorisers and booster parameters.
    :rtype: List[SweepConfig]
    """

    def _parse_grid(raw: str, caster):
        """
        Parse a comma-delimited grid string into typed values.

        :param raw: Raw comma-separated string.
        :type raw: str
        :param caster: Callable converting individual tokens.
        :type caster: Callable[[str], object]
        :returns: List of cast values.
        :rtype: List[object]
        """
        tokens = _split_tokens(raw)
        return [caster(token) for token in tokens]

    numeric_grids = {
        "learning_rate": _parse_grid(args.learning_rate_grid, float),
        "max_depth": _parse_grid(args.max_depth_grid, int),
        "n_estimators": _parse_grid(args.n_estimators_grid, int),
        "subsample": _parse_grid(args.subsample_grid, float),
        "colsample_bytree": _parse_grid(args.colsample_grid, float),
        "reg_lambda": _parse_grid(args.reg_lambda_grid, float),
        "reg_alpha": _parse_grid(args.reg_alpha_grid, float),
    }

    vectorizer_tokens = _split_tokens(args.text_vectorizer_grid) or ["tfidf"]
    vectorizer_values = [token.lower() for token in vectorizer_tokens]

    def _vectorizer_cli(kind: str) -> Tuple[str, Tuple[str, ...]]:
        """
        Translate a vectoriser identifier into CLI overrides.

        :param kind: Vectoriser name (``tfidf``, ``word2vec``, ``sentence_transformer``).
        :type kind: str
        :returns: Pair of ``(tag, cli_args)`` describing the configuration.
        :rtype: Tuple[str, Tuple[str, ...]]
        :raises ValueError: If the vectoriser kind is unsupported.
        """

        if kind == "tfidf":
            return "tfidf", ()
        if kind == "word2vec":
            cli: List[str] = [
                "--word2vec_size",
                str(args.word2vec_size),
                "--word2vec_window",
                str(args.word2vec_window),
                "--word2vec_min_count",
                str(args.word2vec_min_count),
                "--word2vec_epochs",
                str(args.word2vec_epochs),
                "--word2vec_workers",
                str(args.word2vec_workers),
            ]
            if args.word2vec_model_dir:
                cli.extend(["--word2vec_model_dir", args.word2vec_model_dir])
            tag = f"w2v{args.word2vec_size}"
            return tag, tuple(cli)
        if kind == "sentence_transformer":
            cli = [
                "--sentence_transformer_model",
                args.sentence_transformer_model,
                "--sentence_transformer_batch_size",
                str(args.sentence_transformer_batch_size),
            ]
            if args.sentence_transformer_device:
                cli.extend(
                    [
                        "--sentence_transformer_device",
                        args.sentence_transformer_device,
                    ]
                )
            cli.append(
                "--sentence_transformer_normalize"
                if args.sentence_transformer_normalize
                else "--sentence_transformer_no_normalize"
            )
            model_name = (
                args.sentence_transformer_model.split("/")[-1]
                if args.sentence_transformer_model
                else kind
            )
            tag = f"st_{_sanitize_token(model_name)}"
            return tag, tuple(cli)
        raise ValueError(f"Unsupported text vectorizer '{kind}' in sweep grid.")

    param_keys = (
        "learning_rate",
        "max_depth",
        "n_estimators",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
    )
    param_values = [numeric_grids[key] for key in param_keys]

    configs: List[SweepConfig] = []
    for vectorizer in vectorizer_values:
        tag, vectorizer_cli = _vectorizer_cli(vectorizer)
        for combination in product(*param_values):
            params = dict(zip(param_keys, combination))
            configs.append(
                SweepConfig(
                    text_vectorizer=vectorizer,
                    vectorizer_tag=tag,
                    learning_rate=params["learning_rate"],
                    max_depth=params["max_depth"],
                    n_estimators=params["n_estimators"],
                    subsample=params["subsample"],
                    colsample_bytree=params["colsample_bytree"],
                    reg_lambda=params["reg_lambda"],
                    reg_alpha=params["reg_alpha"],
                    vectorizer_cli=vectorizer_cli,
                )
            )
    return configs


def _resolve_study_specs(
    *,
    dataset: str,
    cache_dir: str,
    requested_issues: Sequence[str],
    requested_studies: Sequence[str],
    allow_incomplete: bool,
) -> List[StudySpec]:
    """
    Resolve participant studies slated for evaluation.

    :param dataset: Dataset identifier or path provided via CLI.
    :type dataset: str
    :param cache_dir: Cache directory used when loading datasets.
    :type cache_dir: str
    :param requested_issues: Issue tokens supplied by the user.
    :type requested_issues: Sequence[str]
    :param requested_studies: Study tokens supplied by the user.
    :type requested_studies: Sequence[str]
    :param allow_incomplete: Flag controlling behaviour when datasets are unavailable.
    :type allow_incomplete: bool
    :returns: Ordered list of study specifications.
    :rtype: List[~common.pipeline.types.StudySpec]
    :raises ValueError: If no studies are selected.
    """

    dataset_source = _load_dataset_source_or_none(
        dataset=dataset,
        cache_dir=cache_dir,
        allow_incomplete=allow_incomplete,
    )
    available_issues = (
        set(issues_in_dataset(dataset_source))
        if dataset_source is not None
        else {spec.issue for spec in DEFAULT_SPECS}
    )

    specs = [
        StudySpec(key=spec.key, issue=spec.issue, label=spec.label)
        for spec in DEFAULT_SPECS
        if spec.issue in available_issues
    ]

    specs = _filter_specs_by_issue(specs, requested_issues, available_issues)
    specs = _filter_specs_by_study(specs, requested_studies)

    if not specs:
        raise ValueError("No studies selected for evaluation.")
    return specs


__all__ = [
    "_parse_args",
    "_repo_root",
    "_default_out_dir",
    "_default_cache_dir",
    "_default_reports_dir",
    "_split_tokens",
    "_sanitize_token",
    "_build_sweep_configs",
    "_resolve_study_specs",
]
