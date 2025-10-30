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

"""Execution helpers for GPT-4o sweep and final evaluation stages."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

from common.pipeline.gpt4o_models import SweepConfig, SweepOutcome
from common.pipeline.io import load_metrics_json

from .cli import build_parser as build_gpt_parser
from .evaluate import run_eval

LOGGER = logging.getLogger("gpt4o.pipeline.sweeps")


def run_gpt_cli(cli_args: Sequence[str]) -> None:
    """
    Invoke the GPT-4o CLI entry point with the provided arguments.

    :param cli_args: Iterable of CLI tokens forwarded to ``gpt4o.cli``.
    :returns: ``None``.
    """

    parser = build_gpt_parser()
    namespace = parser.parse_args(list(cli_args))
    run_eval(namespace)


def run_sweeps(
    *,
    configs: Sequence[SweepConfig],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    sweep_dir: Path,
) -> List[SweepOutcome]:
    """
    Run GPT-4o sweeps for each configuration and collect outcomes.

    :param configs: Iterable of sweep configurations to evaluate.
    :param base_cli: CLI tokens shared by every sweep invocation.
    :param extra_cli: Additional CLI tokens preserved from ``main``.
    :param sweep_dir: Root directory receiving sweep outputs.
    :returns: Ordered list of sweep outcomes with metrics metadata.
    """

    outcomes: List[SweepOutcome] = []
    for config in configs:
        run_dir = sweep_dir / config.label()
        cli_args: List[str] = []
        cli_args.extend(base_cli)
        cli_args.extend(config.cli_args())
        cli_args.extend(["--out_dir", str(run_dir)])
        cli_args.extend(extra_cli)
        LOGGER.info("[SWEEP] config=%s", config.label())
        run_gpt_cli(cli_args)
        metrics_path = run_dir / "metrics.json"
        metrics = load_metrics_json(metrics_path)
        outcomes.append(
            SweepOutcome(
                config=config,
                accuracy=float(metrics.get("accuracy_overall", 0.0)),
                parsed_rate=float(metrics.get("parsed_rate", 0.0)),
                format_rate=float(metrics.get("format_rate", 0.0)),
                metrics_path=metrics_path,
                metrics=metrics,
            )
        )
    return outcomes


def select_best(outcomes: Sequence[SweepOutcome]) -> SweepOutcome:
    """
    Return the best sweep outcome by accuracy, parsed rate, then format rate.

    :param outcomes: Sequence of evaluated sweep outcomes.
    :returns: Selected outcome achieving the best metrics.
    :raises RuntimeError: If ``outcomes`` is empty.
    """

    if not outcomes:
        raise RuntimeError("No sweep outcomes available for selection.")
    best = outcomes[0]
    for outcome in outcomes[1:]:
        if outcome.accuracy > best.accuracy + 1e-9:
            best = outcome
            continue
        if abs(outcome.accuracy - best.accuracy) <= 1e-9:
            if outcome.parsed_rate > best.parsed_rate + 1e-9:
                best = outcome
                continue
            if abs(outcome.parsed_rate - best.parsed_rate) <= 1e-9:
                if outcome.format_rate > best.format_rate + 1e-9:
                    best = outcome
    return best


def promote_sweep_results(
    *,
    selected: SweepOutcome,
    sweep_dir: Path,
    final_out_dir: Path,
    overwrite: bool,
) -> Tuple[Path, Mapping[str, object]]:
    """
    Copy the best sweep artefacts into the final directory.

    :param selected: Sweep outcome chosen for promotion.
    :param sweep_dir: Root directory containing per-configuration artefacts.
    :param final_out_dir: Destination directory for promoted results.
    :param overwrite: Whether existing final directories should be replaced.
    :returns: Tuple of destination directory and loaded metrics payload.
    :raises FileNotFoundError: If the selected sweep directory is missing.
    """

    source_dir = sweep_dir / selected.config.label()
    if not source_dir.exists():
        raise FileNotFoundError(f"Sweep artefacts not found at {source_dir}")
    dest_dir = final_out_dir / selected.config.label()
    if dest_dir.exists():
        if overwrite:
            LOGGER.info("[PROMOTE] Overwrite enabled; clearing %s", dest_dir)
            shutil.rmtree(dest_dir)
        else:
            LOGGER.info("[PROMOTE] Reusing existing final directory %s", dest_dir)
            metrics_path = dest_dir / "metrics.json"
            return dest_dir, load_metrics_json(metrics_path)

    LOGGER.info("[PROMOTE] Copying sweep artefacts %s -> %s", source_dir, dest_dir)
    shutil.copytree(source_dir, dest_dir)
    metrics_path = dest_dir / "metrics.json"
    metrics = load_metrics_json(metrics_path)
    return dest_dir, metrics


def run_final_evaluation(
    *,
    config: SweepConfig,
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    out_dir: Path,
) -> Tuple[Path, Mapping[str, object]]:
    """
    Run a final evaluation for the selected configuration.

    :param config: Winning sweep configuration.
    :param base_cli: CLI tokens shared between sweep and final invocations.
    :param extra_cli: Additional CLI tokens supplied by the user.
    :param out_dir: Directory where final evaluation results are written.
    :returns: Tuple of run directory and loaded metrics payload.
    """

    run_dir = out_dir / config.label()
    cli_args: List[str] = []
    cli_args.extend(base_cli)
    cli_args.extend(config.cli_args())
    cli_args.extend(["--out_dir", str(run_dir)])
    cli_args.extend(extra_cli)
    run_gpt_cli(cli_args)
    metrics_path = run_dir / "metrics.json"
    metrics = load_metrics_json(metrics_path)
    return run_dir, metrics


__all__ = [
    "run_sweeps",
    "select_best",
    "promote_sweep_results",
    "run_final_evaluation",
    "run_gpt_cli",
]
