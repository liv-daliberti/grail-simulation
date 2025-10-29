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

"""Pipeline runner for the GPT-4o slate baseline."""

# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple

from common.cli.args import add_comma_separated_argument
from common.cli.options import add_log_level_argument, add_overwrite_argument, add_studies_argument

from .opinion import OpinionEvaluationResult, run_opinion_evaluations
from .pipeline_cache import run_reports_stage
from .pipeline_models import PipelinePaths, SweepConfig, SweepOutcome
from .pipeline_reports import run_report_generation
from .pipeline_sweeps import promote_sweep_results, run_sweeps, select_best

LOGGER = logging.getLogger("gpt4o.pipeline")


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse pipeline arguments while preserving additional CLI options.

    :param argv: Optional iterable of CLI tokens to parse (defaults to ``sys.argv``).
    :returns: Tuple containing the parsed namespace and a list of unconsumed tokens.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Hyper-parameter sweeps, selection, and reporting for the GPT-4o slate baseline."
        )
    )
    parser.add_argument("--dataset", default=None, help="Dataset path or HuggingFace dataset id.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Root directory for GPT-4o outputs (default: <repo>/models/gpt-4o).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HF datasets cache directory (default: <repo>/.cache/huggingface/gpt4o).",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directory receiving Markdown reports (default: <repo>/reports/gpt4o).",
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help="Directory for sweep artefacts (default: <out-dir>/sweeps).",
    )
    parser.add_argument(
        "--eval-max",
        type=int,
        default=0,
        help="Limit evaluation rows per run (0 keeps every example).",
    )
    parser.add_argument(
        "--issues",
        default="",
        help="Comma-separated issue labels to filter (defaults to all issues).",
    )
    add_studies_argument(
        parser,
        help_text=(
            "Comma-separated participant study identifiers to filter (defaults to all studies)."
        ),
    )
    add_comma_separated_argument(
        parser,
        flags="--opinion-studies",
        dest="opinion_studies",
        help_text=(
            "Comma-separated opinion study keys to evaluate (defaults to all opinion studies)."
        ),
    )
    parser.add_argument(
        "--temperature-grid",
        default="0.0,0.2,0.4",
        help="Comma-separated temperatures explored during sweeps.",
    )
    parser.add_argument(
        "--max-tokens-grid",
        default="500",
        help="Comma-separated max_token values explored during sweeps.",
    )
    parser.add_argument(
        "--top-p-grid",
        default="1.0",
        help="Comma-separated top_p values explored during sweeps.",
    )
    parser.add_argument(
        "--opinion-max-participants",
        type=int,
        default=0,
        help=(
            "Optional cap on participants per opinion study during GPT-4o evaluation "
            "(0 keeps all)."
        ),
    )
    parser.add_argument(
        "--opinion-direction-tolerance",
        type=float,
        default=1e-6,
        help=(
            "Tolerance for treating opinion deltas as no-change when computing "
            "direction accuracy."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["full", "reports"],
        default="full",
        help="Select which portion of the pipeline to execute (default: run all stages).",
    )
    parser.add_argument(
        "--request-retries",
        dest="request_retries",
        type=int,
        default=5,
        help="Maximum GPT-4o attempts per request (default: 5).",
    )
    parser.add_argument(
        "--request-retry-delay",
        dest="request_retry_delay",
        type=float,
        default=1.0,
        help="Seconds to wait between GPT-4o request retries (default: 1.0).",
    )
    add_log_level_argument(parser)
    add_overwrite_argument(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the planned actions without executing sweeps or evaluations.",
    )
    parsed, extra = parser.parse_known_args(argv)
    return parsed, list(extra)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """
    Return the repository root used to derive default directories.

    :returns: Absolute path to the repository root.
    """

    return Path(__file__).resolve().parents[2]


def _default_out_dir(root: Path) -> Path:
    """
    Return the default pipeline output directory rooted at ``root``.

    :param root: Repository root path.
    :returns: Path to the default GPT-4o output directory.
    """

    return root / "models" / "gpt-4o"


def _default_cache_dir(root: Path) -> Path:
    """
    Return the default Hugging Face cache location under ``root``.

    :param root: Repository root path.
    :returns: Path to the Hugging Face cache directory.
    """

    return root / ".cache" / "huggingface" / "gpt4o"


def _default_reports_dir(root: Path) -> Path:
    """
    Return the default reports directory rooted at ``root``.

    :param root: Repository root path.
    :returns: Path to the reports directory.
    """

    return root / "reports" / "gpt4o"


def _split_tokens(raw: str) -> List[str]:
    """
    Split a comma-separated string into trimmed, non-empty tokens.

    :param raw: Raw comma-delimited input string.
    :returns: Ordered list of trimmed tokens.
    """

    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _parse_float_grid(raw: str, fallback: float) -> List[float]:
    """
    Parse a comma-separated list of floats, falling back to ``fallback``.

    :param raw: Raw comma-separated string of floats.
    :param fallback: Value to use when parsing yields no valid entries.
    :returns: List of parsed float values.
    """

    tokens = _split_tokens(raw)
    values: List[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            LOGGER.warning("Ignoring invalid temperature token '%s'", token)
    return values or [fallback]


def _parse_int_grid(raw: str, fallback: int) -> List[int]:
    """
    Parse a comma-separated list of integers, falling back to ``fallback``.

    :param raw: Raw comma-separated string of integers.
    :param fallback: Value to use when parsing yields no valid entries.
    :returns: List of parsed integer values.
    """

    tokens = _split_tokens(raw)
    values: List[int] = []
    for token in tokens:
        try:
            values.append(int(token))
        except ValueError:
            LOGGER.warning("Ignoring invalid max_tokens token '%s'", token)
    return values or [fallback]


def _resolve_paths(args: argparse.Namespace) -> PipelinePaths:
    """
    Resolve the output and report directories for the pipeline run.

    :param args: Parsed CLI arguments supplied to ``main``.
    :returns: Aggregated pipeline path configuration.
    """

    root = _repo_root()
    out_dir = Path(args.out_dir or _default_out_dir(root))
    reports_dir = Path(args.reports_dir or _default_reports_dir(root))
    sweep_dir = Path(args.sweep_dir or (out_dir / "sweeps"))
    final_out_dir = out_dir / "next_video"
    opinion_dir = out_dir / "opinion"
    final_out_dir.mkdir(parents=True, exist_ok=True)
    opinion_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(args.cache_dir or _default_cache_dir(root))
    return PipelinePaths(
        out_dir=out_dir,
        final_out_dir=final_out_dir,
        opinion_dir=opinion_dir,
        sweep_dir=sweep_dir,
        reports_dir=reports_dir,
        cache_dir=cache_dir,
    )


def _configure_environment(cache_dir: str) -> None:
    """
    Ensure HuggingFace caches default to ``cache_dir``.

    :param cache_dir: Filesystem location used as the HF cache.
    :returns: ``None``.
    """

    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    os.environ.setdefault("HF_HOME", cache_dir)


def _build_base_cli_args(args: argparse.Namespace, *, cache_dir: str) -> List[str]:
    """
    Assemble the common CLI arguments forwarded to ``gpt4o.cli``.

    :param args: Parsed pipeline namespace.
    :param cache_dir: Resolved Hugging Face cache directory.
    :returns: Flat list of CLI argument tokens.
    """

    base_cli: List[str] = [
        "--cache_dir",
        cache_dir,
        "--eval_max",
        str(args.eval_max),
        "--log_level",
        args.log_level.upper(),
    ]
    dataset = args.dataset or ""
    if dataset:
        base_cli.extend(["--dataset", dataset])
    if args.issues:
        base_cli.extend(["--issues", args.issues])
    if args.studies:
        base_cli.extend(["--studies", args.studies])
    if getattr(args, "opinion_studies", ""):
        base_cli.extend(["--opinion_studies", args.opinion_studies])
    if getattr(args, "opinion_max_participants", 0):
        base_cli.extend(["--opinion_max_participants", str(args.opinion_max_participants)])
    if getattr(args, "opinion_direction_tolerance", None) is not None:
        base_cli.extend(
            ["--opinion_direction_tolerance", str(args.opinion_direction_tolerance)]
        )
    if getattr(args, "request_retries", None) is not None:
        base_cli.extend(["--request_retries", str(args.request_retries)])
    if getattr(args, "request_retry_delay", None) is not None:
        base_cli.extend(["--request_retry_delay", str(args.request_retry_delay)])
    if args.overwrite:
        base_cli.append("--overwrite")
    return base_cli


def _build_sweep_configs(args: argparse.Namespace) -> List[SweepConfig]:
    """
    Construct the Cartesian product of temperature, max-token, and top-p sweeps.

    :param args: Parsed pipeline namespace containing grid definitions.
    :returns: List of sweep configurations to evaluate.
    """

    temperatures = _parse_float_grid(args.temperature_grid, 0.0)
    max_tokens_values = _parse_int_grid(args.max_tokens_grid, 32)
    top_p_values = _parse_float_grid(args.top_p_grid, 1.0)
    configs: List[SweepConfig] = []
    for temp in temperatures:
        for max_tokens in max_tokens_values:
            for top_p in top_p_values:
                configs.append(SweepConfig(temperature=temp, max_tokens=max_tokens, top_p=top_p))
    return configs


def _run_full_pipeline(
    *,
    args: argparse.Namespace,
    extra_cli: Sequence[str],
    paths: PipelinePaths,
) -> OpinionEvaluationResult | None:
    """
    Execute sweeps, selection, final evaluation, and report regeneration.

    :param args: Parsed CLI arguments forwarded from :func:`main`.
    :param extra_cli: Additional CLI tokens preserved from user input.
    :param paths: Resolved filesystem paths required by the pipeline.
    :returns: Opinion evaluation result or ``None`` when running a dry run.
    """

    configs = _build_sweep_configs(args)
    LOGGER.info("Planned %d GPT-4o configurations.", len(configs))
    LOGGER.info("Hyper-parameter sweeps evaluate the validation split only (no training stage).")

    if args.dry_run:
        for config in configs:
            LOGGER.info("[DRY-RUN] would evaluate config=%s", config.label())
        return None

    base_cli = _build_base_cli_args(args, cache_dir=paths.cache_dir)

    outcomes = run_sweeps(
        configs=configs,
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=paths.sweep_dir,
    )
    selected = select_best(outcomes)
    LOGGER.info("Selected configuration: %s", selected.config.label())

    final_dir, final_metrics = promote_sweep_results(
        selected=selected,
        sweep_dir=paths.sweep_dir,
        final_out_dir=paths.final_out_dir,
        overwrite=args.overwrite,
    )

    LOGGER.info("Running opinion-shift evaluation for %s", selected.config.label())
    opinion_result = run_opinion_evaluations(
        args=args,
        config_label=selected.config.label(),
        out_dir=paths.opinion_dir / selected.config.label(),
    )

    LOGGER.info("Final metrics stored under %s", final_dir)
    run_report_generation(
        paths=paths,
        repo_root=_repo_root(),
        outcomes=outcomes,
        selected=selected,
        final_metrics=final_metrics,
        opinion_result=opinion_result,
    )
    return opinion_result


def main(argv: Sequence[str] | None = None) -> None:
    """
    Entry point that orchestrates sweep execution and report generation.

    :param argv: Optional CLI argument sequence supplied by callers.
    :returns: ``None``.
    """

    args, extra_cli = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    paths = _resolve_paths(args)
    _configure_environment(paths.cache_dir)

    if args.stage == "reports":
        run_reports_stage(paths, repo_root=_repo_root())
        return

    _run_full_pipeline(args=args, extra_cli=extra_cli, paths=paths)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["main", "SweepConfig", "SweepOutcome", "PipelinePaths"]
