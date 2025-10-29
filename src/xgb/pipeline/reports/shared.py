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

"""Shared utilities for the XGBoost pipeline report builders."""

from __future__ import annotations

import logging
import re
import shlex
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence

from common.visualization.matplotlib import plt
from common.pipeline.io import write_markdown_lines
from common.reports.utils import start_markdown_report
from ..context import OpinionStudySelection, StudySelection

LOGGER = logging.getLogger("xgb.pipeline.reports")

_SLUG_INVALID = re.compile(r"[^a-z0-9]+")


def _slugify_label(value: str, *, fallback: str = "item") -> str:
    """
    Convert ``value`` into a filesystem-friendly slug.

    Args:
        value: Raw label to normalise.
        fallback: Identifier to use when ``value`` collapses to an empty string.

    Returns:
        Sanitised, lowercase slug containing only ``[a-z0-9_]``.
    """

    normalised = unicodedata.normalize("NFKD", value or "")
    ascii_bytes = normalised.encode("ascii", "ignore")
    ascii_value = ascii_bytes.decode("ascii", "ignore")
    slug = _SLUG_INVALID.sub("_", ascii_value.lower()).strip("_")
    return slug or fallback


# pylint: disable=too-many-return-statements
def _format_shell_command(bits: Sequence[str]) -> str:
    """
    Join CLI arguments into a shell-friendly command.

    :param bits: Individual command-line arguments to join.
    :type bits: Sequence[str]
    :returns: Shell-escaped command string.
    :rtype: str
    """

    return " ".join(shlex.quote(str(bit)) for bit in bits if str(bit))


def _xgb_next_video_command(selection: Optional[StudySelection]) -> Optional[str]:
    """
    Build a reproduction command for a next-video sweep selection.

    :param selection: Selected sweep outcome containing configuration metadata.
    :type selection: Optional[StudySelection]
    :returns: Shell command capable of reproducing the sweep, or ``None`` if unavailable.
    :rtype: Optional[str]
    """

    if selection is None:
        return None
    metrics = selection.outcome.metrics
    params = metrics.get("xgboost_params", {})
    tree_method = (
        params.get("tree_method")
        or metrics.get("config", {}).get("tree_method")
        or "hist"
    )
    dataset = (
        metrics.get("dataset_source")
        or metrics.get("dataset")
        or "data/cleaned_grail"
    )
    extra_fields = metrics.get("extra_fields") or []

    cli_bits = selection.config.cli_args(str(tree_method))
    command: List[str] = [
        "python",
        "-m",
        "xgb.cli",
        "--fit_model",
        "--dataset",
        str(dataset),
        "--issues",
        selection.study.issue,
        "--participant_studies",
        selection.study.key,
    ]
    if extra_fields:
        command.extend(["--extra_text_fields", ",".join(sorted(set(map(str, extra_fields))))])
    command.extend(cli_bits)
    command.extend(["--out_dir", "<run_dir>"])
    return _format_shell_command(command)


def _xgb_opinion_command(selection: Optional[OpinionStudySelection]) -> Optional[str]:
    """
    Build a reproduction command for an opinion sweep selection.

    :param selection: Selected opinion sweep outcome with configuration metadata.
    :type selection: Optional[~xgb.pipeline.context.OpinionStudySelection]
    :returns: Shell command capable of reproducing the opinion pipeline, or ``None``.
    :rtype: Optional[str]
    """

    if selection is None:
        return None
    outcome = selection.outcome
    metrics = outcome.metrics
    config = outcome.config
    config_block = metrics.get("config", {})
    tree_method = (
        config_block.get("tree_method")
        or metrics.get("xgboost_params", {}).get("tree_method")
        or "hist"
    )
    dataset = (
        metrics.get("dataset")
        or metrics.get("dataset_source")
        or "data/cleaned_grail"
    )
    extra_fields = metrics.get("extra_fields") or []
    max_features = config_block.get("max_features")

    command: List[str] = [
        "python",
        "-m",
        "xgb.pipeline",
        "--stage",
        "full",
        "--tasks",
        "opinion",
        "--issues",
        selection.study.issue,
        "--studies",
        selection.study.key,
        "--tree-method",
        str(tree_method),
        "--learning-rate-grid",
        f"{config.learning_rate:g}",
        "--max-depth-grid",
        str(config.max_depth),
        "--n-estimators-grid",
        str(config.n_estimators),
        "--subsample-grid",
        f"{config.subsample:g}",
        "--colsample-grid",
        f"{config.colsample_bytree:g}",
        "--reg-lambda-grid",
        f"{config.reg_lambda:g}",
        "--reg-alpha-grid",
        f"{config.reg_alpha:g}",
        "--text-vectorizer-grid",
        config.text_vectorizer,
        "--out-dir",
        "<models_dir>",
    ]
    if dataset:
        command.extend(["--dataset", str(dataset)])
    if extra_fields:
        command.extend(["--extra-text-fields", ",".join(sorted(set(map(str, extra_fields))))])
    if max_features:
        command.extend(["--max-features", str(max_features)])
    if config.vectorizer_cli:
        vectorizer_bits = list(config.vectorizer_cli)
        idx = 0
        while idx < len(vectorizer_bits):
            token = vectorizer_bits[idx]
            if isinstance(token, str) and token.startswith("--"):
                option = token.replace("_", "-")
                command.append(option)
                idx += 1
                if option.endswith("-normalize") or option.endswith("-no-normalize"):
                    continue
                if idx < len(vectorizer_bits):
                    command.append(str(vectorizer_bits[idx]))
                    idx += 1
            else:
                command.append(str(token))
                idx += 1
    return _format_shell_command(command)


def _write_disabled_report(directory: Path, title: str, message: str) -> None:
    """
    Emit a placeholder report clarifying why a section is absent.

    :param directory: Directory where the placeholder README is written.
    :type directory: Path
    :param title: Report title communicated to readers.
    :type title: str
    :param message: Explanatory message describing the omission.
    :type message: str
    """

    path, lines = start_markdown_report(directory, title=title)
    lines.append(message)
    lines.append("")
    write_markdown_lines(path, lines)


# pylint: disable=undefined-all-variable

__all__ = [
    "LOGGER",
    "plt",
    "_slugify_label",
    "_format_shell_command",
    "_xgb_next_video_command",
    "_xgb_opinion_command",
    "_write_disabled_report",
]
