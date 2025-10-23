"""Reusable CLI option helpers shared across pipelines."""

from __future__ import annotations

import argparse


def add_jobs_argument(parser: argparse.ArgumentParser) -> None:
    """

    Add the ``--jobs`` argument controlling parallelism.



    :param parser: Value provided for ``parser``.

    :type parser: argparse.ArgumentParser

    :returns: ``None``.

    :rtype: None

    """


    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum number of concurrent workers (default: 1).",
    )


def add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    """

    Add the common stage and sweep-task arguments.



    :param parser: Value provided for ``parser``.

    :type parser: argparse.ArgumentParser

    :returns: ``None``.

    :rtype: None

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



    :param parser: Value provided for ``parser``.

    :type parser: argparse.ArgumentParser

    :returns: ``None``.

    :rtype: None

    """


    parser.add_argument(
        "--log-level",
        "--log_level",
        default="INFO",
        help="Logging level for the pipeline logger.",
    )


def add_overwrite_argument(parser: argparse.ArgumentParser) -> None:
    """

    Add the standard ``--overwrite`` boolean flag.



    :param parser: Value provided for ``parser``.

    :type parser: argparse.ArgumentParser

    :returns: ``None``.

    :rtype: None

    """


    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting metrics/prediction files in --out_dir.",
    )