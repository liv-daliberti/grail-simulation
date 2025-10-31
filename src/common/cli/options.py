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


def add_gpt4o_eval_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_out_dir: str,
    default_cache_dir: str,
) -> None:
    """
    Register the standard GPT-4o evaluation arguments on ``parser``.

    :param parser: Argument parser to extend.
    :param default_out_dir: Default output directory path.
    :param default_cache_dir: Default Hugging Face cache directory.
    :returns: ``None``.
    """

    parser.add_argument(
        "--eval_max",
        type=int,
        default=0,
        help="Limit evaluation examples (0 means evaluate every row).",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset path or Hugging Face dataset id (defaults to config.DATASET_NAME).",
    )
    parser.add_argument(
        "--issues",
        default="",
        help="Comma-separated list of issue labels to evaluate (defaults to all issues).",
    )
    add_studies_argument(
        parser,
        help_text=(
            "Comma-separated participant study identifiers to filter (defaults to all studies)."
        ),
    )
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
            "Tolerance for treating opinion deltas as no-change when measuring direction accuracy."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=default_out_dir,
        help="Directory for predictions and metrics.",
    )
    add_overwrite_argument(parser)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to GPT-4o.",
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
        help="Nucleus sampling probability mass forwarded to GPT-4o.",
    )
    parser.add_argument(
        "--request_retries",
        type=int,
        default=5,
        help="Maximum GPT-4o attempts per request (default: 5).",
    )
    parser.add_argument(
        "--request_retry_delay",
        type=float,
        default=1.0,
        help="Seconds to wait between GPT-4o request retries (default: 1.0).",
    )
    parser.add_argument(
        "--cache_dir",
        default=default_cache_dir,
        help="HF datasets cache directory.",
    )
    parser.add_argument(
        "--deployment",
        default="",
        help="Optional Azure deployment identifier (defaults to the config setting).",
    )
    add_log_level_argument(parser)
