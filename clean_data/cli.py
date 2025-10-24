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

"""Command-line entry point that ties together the cleaning workflow.

The CLI parses user arguments, configures :mod:`clean_data.clean_data`
options, orchestrates dataset construction, kicks off prompt analytics,
and handles optional issue-level exports. It mirrors the behaviour of the
original CodeOcean pipeline while exposing a Python-first interface. Usage
of this CLI is governed by the repository's Apache 2.0 license; consult
LICENSE for the full terms.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from clean_data.clean_data import (
    BuildOptions,
    build_clean_dataset,
    export_issue_datasets,
    generate_prompt_stats,
    parse_issue_repo_specs,
    save_dataset,
)


def _default_max_history() -> int:
    """Return the default value for ``--max-history``.

    Reads :envvar:`GRAIL_MAX_HISTORY` when present, otherwise defaults to 12.

    :returns: Integer history window length.
    """

    return int(os.environ.get("GRAIL_MAX_HISTORY", "12"))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Optional list of command-line arguments. When ``None`` the
        values are read from :data:`sys.argv`.
    :returns: Parsed :class:`argparse.Namespace` with CLI options.
    """

    parser = argparse.ArgumentParser(
        description="Build cleaned GRAIL datasets suitable for GRPO training."
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="HF hub id, load_from_disk directory, file path, or CodeOcean capsule directory.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--sol-key",
        default=None,
        help="Alternate column containing the gold next id.",
    )
    parser.add_argument("--max-history", type=int, default=_default_max_history())
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation share when building directly from the CodeOcean capsule.",
    )
    parser.add_argument(
        "--issue-repo",
        action="append",
        default=[],
        help="Optional issue=repo mapping for pushing cleaned splits to the Hugging Face hub.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push cleaned datasets to the hub",
    )
    parser.add_argument(
        "--hub-token",
        default=None,
        help="Token for authenticated Hugging Face pushes",
    )
    parser.add_argument(
        "--prompt-stats-dir",
        default=None,
        help=(
            "If set, generate prompt feature histograms and summary statistics "
            "into this directory."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel worker processes used for filtering and example mapping.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point invoked by the CLI scripts.

    :param argv: Optional command-line argument list.
    """

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    jobs = None
    if args.jobs is not None:
        if args.jobs < 1:
            logging.getLogger("clean_grail").warning(
                "Ignoring invalid --jobs value %d; expected >= 1.", args.jobs
            )
        else:
            jobs = args.jobs

    build_options = BuildOptions(
        validation_ratio=args.validation_ratio,
        train_split=args.train_split,
        validation_split=args.test_split,
        system_prompt=args.system_prompt,
        sol_key=args.sol_key,
        max_history=args.max_history,
        num_proc=jobs,
    )
    full_dataset = build_clean_dataset(args.dataset_name, options=build_options)

    if args.prompt_stats_dir:
        if {"train", "validation"}.issubset(full_dataset.keys()):
            generate_prompt_stats(full_dataset, Path(args.prompt_stats_dir))
            logging.getLogger("clean_grail").info(
                "Prompt statistics package executed; artifacts written to %s",
                args.prompt_stats_dir,
            )
        else:
            logging.getLogger("clean_grail").warning(
                "Prompt stats requested (dir=%s) but dataset lacks both 'train' "
                "and 'validation' splits; skipping.",
                args.prompt_stats_dir,
            )

    save_dataset(full_dataset, Path(args.output_dir))

    issue_repo_map = parse_issue_repo_specs(args.issue_repo)
    export_issue_datasets(
        full_dataset,
        Path(args.output_dir),
        issue_repo_map,
        push_to_hub=args.push_to_hub,
        hub_token=args.hub_token,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
