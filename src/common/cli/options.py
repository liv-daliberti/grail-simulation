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

"""Reusable option builders shared across pipeline CLIs."""

from __future__ import annotations

import argparse

from pathlib import Path
from .args import add_comma_separated_argument


def add_jobs_argument(parser: argparse.ArgumentParser) -> None:
    """
    Register the ``--jobs`` argument controlling worker parallelism.

    :param parser: Argument parser receiving the job-count configuration flag.
    :returns: ``None``.
    """
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum number of concurrent workers (default: 1).",
    )


def add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register shared stage-selection arguments used by pipelines.

    :param parser: Argument parser being extended with stage options.
    :returns: ``None``.
    """
    parser.add_argument(
        "--stage",
        choices=["full", "plan", "sweeps", "finalize", "reports"],
        default="full",
        help="Select which portion of the pipeline to execute (default: run all stages).",
    )
    parser.add_argument(
        "--sweep-task-id",
        type=int,
        default=None,
        help="0-based sweep task index to execute when --stage=sweeps.",
    )
    parser.add_argument(
        "--sweep-task-count",
        type=int,
        default=None,
        help="Expected total number of sweep tasks (for validation in --stage=sweeps).",
    )


def add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    """
    Add the shared ``--log-level`` argument.

    :param parser: Argument parser receiving the log-level configuration flag.
    :returns: ``None``.
    """
    parser.add_argument(
        "--log-level",
        "--log_level",
        default="INFO",
        help="Logging level for the pipeline logger.",
    )


def add_overwrite_argument(parser: argparse.ArgumentParser) -> None:
    """
    Expose the standard ``--overwrite`` boolean flag.

    :param parser: Argument parser receiving the overwrite toggle.
    :returns: ``None``.
    """
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting metrics/prediction files in --out_dir.",
    )


def add_studies_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str,
    flags: str = "--studies",
    dest: str = "studies",
) -> None:
    """
    Register the shared ``--studies`` argument used across pipelines.

    :param parser: Argument parser receiving the studies filter option.
    :param help_text: Help text describing the accepted study identifiers.
    :param flags: CLI flag name (defaults to ``--studies``).
    :param dest: Destination attribute name (defaults to ``studies``).
    :returns: ``None``.
    """

    add_comma_separated_argument(
        parser,
        flags=flags,
        dest=dest,
        help_text=help_text,
    )


def add_eval_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_out_dir: str,
    default_cache_dir: str,
    include_llm_args: bool = False,
    include_opinion_args: bool = False,
    include_studies_filter: bool = False,
    dataset_default: str = "",
    issues_default: str = "",
    include_legacy_aliases: bool = True,
) -> None:
    """
    Register standard evaluation arguments on ``parser``.

    - Always adds: ``--eval_max``, ``--dataset``, ``--issues``, ``--out_dir``,
      ``--cache_dir``, overwrite, and log-level flags.
    - When ``include_studies_filter`` is True, adds the shared ``--studies`` filter.
    - When ``include_opinion_args`` is True, adds opinion-specific flags.
    - When ``include_llm_args`` is True, adds LLM invocation parameters.
    - When ``include_legacy_aliases`` is True, add back-compat aliases
      (``--eval-max`` and ``--issue``).

    :param parser: Argument parser to extend.
    :param default_out_dir: Default directory for predictions and metrics when
        ``--out_dir`` is omitted.
    :param default_cache_dir: Default Hugging Face datasets cache directory used
        for ``--cache_dir``.
    :param include_llm_args: When ``True``, include generation parameters such as
        ``--temperature``, ``--top_p``, and ``--max_tokens``.
    :param include_opinion_args: When ``True``, include opinion‑specific flags
        (e.g., ``--opinion_studies``, ``--opinion_max_participants``).
    :param include_studies_filter: When ``True``, add the shared ``--studies`` filter.
    :param dataset_default: Default value used for ``--dataset``.
    :param issues_default: Default value used for ``--issues``.
    :param include_legacy_aliases: When ``True``, also register legacy flag aliases
        for back‑compatibility.
    :returns: ``None``.
    """

    # eval_max with optional legacy alias
    eval_flags = ("--eval-max", "--eval_max") if include_legacy_aliases else ("--eval_max",)
    parser.add_argument(
        *eval_flags,
        type=int,
        default=0,
        dest="eval_max",
        help="Limit evaluation examples (0 means evaluate every row).",
    )

    # dataset, issues (with optional legacy alias)
    parser.add_argument(
        "--dataset",
        default=dataset_default,
        help="Dataset path or Hugging Face dataset id (defaults to config.DATASET_NAME).",
    )
    issue_flags = ("--issues", "--issue") if include_legacy_aliases else ("--issues",)
    parser.add_argument(
        *issue_flags,
        default=issues_default,
        dest="issues",
        help="Comma-separated list of issue labels to evaluate (defaults to all issues).",
    )

    if include_studies_filter:
        add_studies_argument(
            parser,
            help_text=(
                "Comma-separated participant study identifiers to filter (defaults to all studies)."
            ),
        )

    if include_opinion_args:
        add_comma_separated_argument(
            parser,
            flags="--opinion_studies",
            dest="opinion_studies",
            help_text=(
                "Comma-separated opinion study keys to evaluate (defaults to all opinion studies)."
            ),
        )
        parser.add_argument(
            "--opinion_max_participants",
            type=int,
            default=0,
            help="Optional cap on participants per opinion study (0 keeps all).",
        )
        parser.add_argument(
            "--opinion_direction_tolerance",
            type=float,
            default=1e-6,
            help=(
                "Tolerance for treating opinion deltas as no-change "
                "when measuring direction accuracy."
            ),
        )

    # Output dirs and cache (accept both underscore and hyphen aliases)
    parser.add_argument(
        "--out_dir",
        "--out-dir",
        dest="out_dir",
        default=default_out_dir,
        help="Directory for predictions and metrics.",
    )
    add_overwrite_argument(parser)
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        dest="cache_dir",
        default=default_cache_dir,
        help="HF datasets cache directory.",
    )

    if include_llm_args:
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.0,
            help="Sampling temperature.",
        )
        parser.add_argument(
            "--max_tokens",
            type=int,
            default=500,
            help="Maximum number of tokens in the model response.",
        )
        parser.add_argument(
            "--top_p",
            type=float,
            default=1.0,
            help="Nucleus sampling probability mass.",
        )
        parser.add_argument(
            "--request_retries",
            type=int,
            default=5,
            help="Maximum attempts per request (default: 5).",
        )
        parser.add_argument(
            "--request_retry_delay",
            type=float,
            default=1.0,
            help="Seconds to wait between request retries (default: 1.0).",
        )
        parser.add_argument(
            "--deployment",
            default="",
            help="Optional deployment identifier.",
        )

    add_log_level_argument(parser)


def add_standard_eval_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_out_dir: str,
    include_llm_args: bool = False,
    include_opinion_args: bool = False,
    include_studies_filter: bool = False,
    dataset_default: str = "data/cleaned_grail",
    issues_default: str = "",
    include_legacy_aliases: bool = True,
) -> None:
    """
    Add the common evaluation arguments with repo-wide defaults.

    This thin wrapper reduces duplicate-code blocks across CLIs by pinning the
    shared defaults (cache dir, dataset default, legacy aliases) while letting
    callers specify the output root directory.

    :param parser: Argument parser to extend.
    :param default_out_dir: Default output directory (e.g., "models/knn").
    :param include_llm_args: Forwarded to :func:`add_eval_arguments`.
    :param include_opinion_args: Forwarded to :func:`add_eval_arguments`.
    :param include_studies_filter: Forwarded to :func:`add_eval_arguments`.
    :param dataset_default: Dataset default; defaults to "data/cleaned_grail".
    :param issues_default: Issues default; defaults to empty string (all).
    :param include_legacy_aliases: Whether to include legacy flag aliases.
    :returns: ``None``.
    """

    add_eval_arguments(
        parser,
        default_out_dir=default_out_dir,
        default_cache_dir=str(Path.cwd() / "hf_cache"),
        include_llm_args=include_llm_args,
        include_opinion_args=include_opinion_args,
        include_studies_filter=include_studies_filter,
        dataset_default=dataset_default,
        issues_default=issues_default,
        include_legacy_aliases=include_legacy_aliases,
    )


def add_reuse_sweeps_argument(parser: argparse.ArgumentParser) -> None:
    """Add a standard ``--reuse-sweeps`` boolean option to ``parser``.

    :param parser: Argument parser receiving the reuse flag.
    :returns: ``None``.
    """

    parser.add_argument(
        "--reuse-sweeps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse cached sweep artefacts when available "
            "(default: off; pass --reuse-sweeps to enable)."
        ),
    )


def add_reuse_final_argument(parser: argparse.ArgumentParser) -> None:
    """Add a standard ``--reuse-final`` boolean option to ``parser``.

    :param parser: Argument parser receiving the reuse flag.
    :returns: ``None``.
    """

    parser.add_argument(
        "--reuse-final",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Reuse cached finalize-stage artefacts when available "
            "(use --no-reuse-final to force recomputation)."
        ),
    )


def add_allow_incomplete_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared ``--allow-incomplete`` flag used by report stages.

    :param parser: Argument parser receiving the allow-incomplete flag.
    :returns: ``None``.
    """

    parser.add_argument(
        "--allow-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow finalize/report stages to proceed with partial sweeps or metrics "
            "(use --no-allow-incomplete to require completeness)."
        ),
    )
