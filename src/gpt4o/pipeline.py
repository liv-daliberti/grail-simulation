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

"""Pipeline runner for the GPT-4o slate baseline.

It assembles sweep planning, evaluation, and report-generation steps for
the GPT-4o workflow used in Grail Simulation experiments."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

from common.cli.args import add_comma_separated_argument
from common.cli.options import add_log_level_argument, add_overwrite_argument
from common.pipeline.io import load_metrics_json, write_markdown_lines
from common.reports.utils import start_markdown_report

from .cli import build_parser as build_gpt_parser
from .evaluate import run_eval
from .opinion import OpinionEvaluationResult, run_opinion_evaluations

LOGGER = logging.getLogger("gpt4o.pipeline")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    """Describe a GPT-4o configuration evaluated during sweeps."""

    temperature: float
    max_tokens: int
    top_p: float

    def label(self) -> str:
        """Return a filesystem-friendly identifier."""

        temp_token = f"temp{self.temperature:g}".replace(".", "p")
        tok_token = f"tok{self.max_tokens}"
        top_p_token = f"tp{self.top_p:g}".replace(".", "p")
        return f"{temp_token}_{tok_token}_{top_p_token}"

    def cli_args(self) -> List[str]:
        """Return CLI overrides encoding this configuration."""

        return [
            "--temperature",
            str(self.temperature),
            "--max_tokens",
            str(self.max_tokens),
            "--top_p",
            str(self.top_p),
        ]


@dataclass
class SweepOutcome:
    """Capture metrics gathered during a sweep run."""

    config: SweepConfig
    accuracy: float
    parsed_rate: float
    format_rate: float
    metrics_path: Path
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class PipelinePaths:
    """Convenience container aggregating resolved output directories."""

    out_dir: Path
    final_out_dir: Path
    opinion_dir: Path
    sweep_dir: Path
    reports_dir: Path
    cache_dir: str


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> Tuple[argparse.Namespace, List[str]]:
    """Parse pipeline arguments while preserving additional CLI options.

    :param argv: Optional iterable of CLI tokens to parse (defaults to ``sys.argv``).
    :returns: ``(namespace, extra_args)`` tuple containing parsed options and leftovers.
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
    add_comma_separated_argument(
        parser,
        flags="--studies",
        dest="studies",
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
            "Optional cap on participants per opinion study during GPT-4o "
            "evaluation (0 keeps all)."
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


def _repo_root() -> Path:
    """Return the repository root used to derive default directories.

    :returns: Absolute path to the repository root.
    """

    return Path(__file__).resolve().parents[2]


def _default_out_dir(root: Path) -> Path:
    """Return the default pipeline output directory rooted at ``root``.

    :param root: Repository root path.
    :returns: Path to the default GPT-4o output directory.
    """

    return root / "models" / "gpt-4o"


def _default_cache_dir(root: Path) -> Path:
    """Return the default Hugging Face cache location under ``root``.

    :param root: Repository root path.
    :returns: Path to the Hugging Face cache directory.
    """

    return root / ".cache" / "huggingface" / "gpt4o"


def _default_reports_dir(root: Path) -> Path:
    """Return the default reports directory rooted at ``root``.

    :param root: Repository root path.
    :returns: Path to the reports directory.
    """

    return root / "reports" / "gpt4o"


def _split_tokens(raw: str) -> List[str]:
    """Split a comma-separated string into trimmed, non-empty tokens.

    :param raw: Raw comma-delimited input string.
    :returns: Ordered list of trimmed tokens.
    """

    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _parse_float_grid(raw: str, fallback: float) -> List[float]:
    """Parse a comma-separated list of floats, falling back to ``fallback``.

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
    """Parse a comma-separated list of integers, falling back to ``fallback``.

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
    """Return the resolved output/report directories for the pipeline run.

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
    """Ensure HuggingFace caches default to ``cache_dir``.

    :param cache_dir: Filesystem location used as the HF cache.
    :returns: ``None``.
    """

    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    os.environ.setdefault("HF_HOME", cache_dir)


def _build_base_cli_args(args: argparse.Namespace, *, cache_dir: str) -> List[str]:
    """Return common CLI arguments forwarded to ``gpt4o.cli``.

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
        base_cli.extend(
            ["--opinion_max_participants", str(args.opinion_max_participants)]
        )
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
    """Construct the Cartesian product of temperature, max-token, and top-p sweeps.

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
                configs.append(
                    SweepConfig(temperature=temp, max_tokens=max_tokens, top_p=top_p)
                )
    return configs


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _run_gpt_cli(cli_args: Sequence[str]) -> None:
    """Invoke the GPT-4o CLI entry point with the provided arguments.

    :param cli_args: Iterable of CLI tokens forwarded to ``gpt4o.cli``.
    :returns: ``None``. Exceptions propagate to the caller.
    """

    parser = build_gpt_parser()
    namespace = parser.parse_args(list(cli_args))
    run_eval(namespace)


def _promote_sweep_results(
    *,
    selected: SweepOutcome,
    sweep_dir: Path,
    final_out_dir: Path,
    overwrite: bool,
) -> Tuple[Path, Mapping[str, object]]:
    """Copy the best sweep artefacts into the final directory.

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

def _run_sweeps(
    *,
    configs: Sequence[SweepConfig],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    sweep_dir: Path,
) -> List[SweepOutcome]:
    """Run GPT-4o sweeps for each configuration and collect outcomes.

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
        _run_gpt_cli(cli_args)
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


def _select_best(outcomes: Sequence[SweepOutcome]) -> SweepOutcome:
    """Return the best sweep outcome by accuracy, parsed rate, then format rate.

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


def _run_final_evaluation(
    *,
    config: SweepConfig,
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    out_dir: Path,
) -> Tuple[Path, Mapping[str, object]]:
    """Run a final evaluation for the selected configuration.

    :param config: Winning sweep configuration.
    :param base_cli: CLI tokens shared between sweep and final invocations.
    :param extra_cli: Additional CLI tokens supplied by the user.
    :param out_dir: Directory where final evaluation results are written.
    :returns: Tuple of run directory and loaded metrics payload.
    :rtype: tuple[pathlib.Path, collections.abc.Mapping[str, object]]
    """

    run_dir = out_dir / config.label()
    cli_args: List[str] = []
    cli_args.extend(base_cli)
    cli_args.extend(config.cli_args())
    cli_args.extend(["--out_dir", str(run_dir)])
    cli_args.extend(extra_cli)
    LOGGER.info("[FINAL] config=%s -> %s", config.label(), run_dir)
    _run_gpt_cli(cli_args)
    metrics_path = run_dir / "metrics.json"
    metrics = load_metrics_json(metrics_path)
    return run_dir, metrics


def _format_rate(value: float) -> str:
    """Format a numeric rate with three decimal places.

    :param value: Raw numeric rate.
    :returns: String representation with three decimal places.
    """

    return f"{value:.3f}"


def _group_highlights(payload: Mapping[str, Mapping[str, object]]) -> List[str]:
    """Return bullet highlights for group-level accuracy extremes.

    :param payload: Mapping of group identifiers to metrics payloads.
    :returns: Markdown bullet lines highlighting best/worst accuracy groups.
    """

    entries: List[Tuple[float, str, int]] = []
    for raw_group, stats in payload.items():
        accuracy = stats.get("accuracy")
        try:
            accuracy_value = float(accuracy)
        except (TypeError, ValueError):
            continue
        eligible_raw = stats.get("n_eligible", 0)
        try:
            eligible_value = int(eligible_raw)
        except (TypeError, ValueError):
            eligible_value = 0
        group_name = str(raw_group or "unspecified")
        entries.append((accuracy_value, group_name, eligible_value))
    if not entries:
        return []
    entries.sort(key=lambda item: item[0], reverse=True)
    lines = [
        f"- Highest accuracy: {entries[0][1]} "
        f"({_format_rate(entries[0][0])}, eligible {entries[0][2]})."
    ]
    if len(entries) > 1:
        lowest = entries[-1]
        lines.append(
            f"- Lowest accuracy: {lowest[1]} "
            f"({_format_rate(lowest[0])}, eligible {lowest[2]})."
        )
    return lines


def _write_catalog_report(reports_dir: Path) -> None:
    """Create the top-level GPT-4o report catalog README.

    :param reports_dir: Directory that will contain the catalog README.
    :returns: ``None``.
    """

    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "README.md"
    lines = [
        "# GPT-4o Report Catalog",
        "",
        "Generated artifacts for the GPT-4o slate-selection baseline:",
        "",
        "- `next_video/` – summary metrics and fairness cuts for the selected configuration.",
        "- `opinion/` – opinion-shift regression metrics across participant studies.",
        "- `hyperparameter_tuning/` – sweep results across temperature and max token settings.",
        "",
        "Model predictions and metrics JSON files live under `models/gpt-4o/`.",
        "",
    ]
    write_markdown_lines(path, lines)


def _write_sweep_report(
    directory: Path,
    outcomes: Sequence[SweepOutcome],
    selected: SweepOutcome,
) -> None:
    """Write the hyper-parameter sweep report summarising all outcomes.

    :param directory: Destination directory for the sweep report.
    :param outcomes: All observed sweep outcomes.
    :param selected: Outcome chosen as the final configuration.
    :returns: ``None``.
    """

    path, lines = start_markdown_report(directory, title="GPT-4o Hyper-parameter Sweep")
    if not outcomes:
        lines.append("No sweep runs were executed.")
        lines.append("")
        write_markdown_lines(path, lines)
        return
    lines.append(
        "The table below captures validation accuracy on eligible slates plus "
        "formatting/parse rates for each temperature/top-p/max-token configuration. "
        "The selected configuration is marked with ✓."
    )
    lines.append("")
    header_cells = [
        "Config",
        "Temperature",
        "Top-p",
        "Max tokens",
        "Accuracy ↑",
        "Parsed ↑",
        "Formatted ↑",
        "Selected",
    ]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for outcome in outcomes:
        mark = "✓" if outcome.config == selected.config else ""
        lines.append(
            f"| `{outcome.config.label()}` | {outcome.config.temperature:.2f} | "
            f"{outcome.config.top_p:.2f} | {outcome.config.max_tokens} | "
            f"{_format_rate(outcome.accuracy)} | "
            f"{_format_rate(outcome.parsed_rate)} | "
            f"{_format_rate(outcome.format_rate)} | {mark} |"
        )
    lines.append("")
    write_markdown_lines(path, lines)


def _write_next_video_report(  # pylint: disable=too-many-statements
    directory: Path,
    selected: SweepOutcome,
    metrics: Mapping[str, object],
) -> None:
    """Write the next-video evaluation report for the selected configuration.

    :param directory: Destination directory for the report.
    :param selected: Winning sweep outcome metadata.
    :param metrics: Final evaluation metrics payload.
    :returns: ``None``.
    """

    path, lines = start_markdown_report(directory, title="GPT-4o Next-Video Baseline")
    lines.append(
        f"- **Selected configuration:** `{selected.config.label()}` "
        f"(temperature={selected.config.temperature:.2f}, top_p={selected.config.top_p:.2f}, "
        f"max_tokens={selected.config.max_tokens})"
    )
    lines.append(
        f"- **Accuracy:** {_format_rate(float(metrics.get('accuracy_overall', 0.0)))} "
        f"on {int(metrics.get('n_eligible', 0))} eligible slates "
        f"out of {int(metrics.get('n_total', 0))} processed."
    )
    lines.append(
        f"- **Parsed rate:** {_format_rate(float(metrics.get('parsed_rate', 0.0)))}  "
        f"**Formatted rate:** {_format_rate(float(metrics.get('format_rate', 0.0)))}"
    )
    filters = metrics.get("filters", {})
    issue_filter = ", ".join(filters.get("issues", [])) if isinstance(filters, Mapping) else ""
    study_filter = ", ".join(filters.get("studies", [])) if isinstance(filters, Mapping) else ""
    filter_parts: List[str] = []
    if issue_filter:
        filter_parts.append(f"issues: {issue_filter}")
    if study_filter:
        filter_parts.append(f"studies: {study_filter}")
    if filter_parts:
        lines.append("- **Filters:** " + ", ".join(filter_parts))
    lines.append("")
    group_metrics = metrics.get("group_metrics", {})

    def _render_group_table(title: str, payload: Mapping[str, Mapping[str, object]]) -> None:
        """Append a Markdown table capturing group-level metrics.

        :param title: Section title used in the report.
        :param payload: Mapping of group identifiers to metric dictionaries.
        :returns: ``None``.
        """
        if not payload:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for group, stats in payload.items():
            seen = int(stats.get("n_seen", 0))
            eligible = int(stats.get("n_eligible", 0))
            accuracy = _format_rate(float(stats.get("accuracy", 0.0)))
            parsed_rate = _format_rate(float(stats.get("parsed_rate", 0.0)))
            format_rate = _format_rate(float(stats.get("format_rate", 0.0)))
            group_name = group or "unspecified"
            line = (
                f"| {group_name} | {seen} | {eligible} | {accuracy} | "
                f"{parsed_rate} | {format_rate} |"
            )
            lines.append(line)
        lines.append("")
        highlight_lines = _group_highlights(payload)
        if highlight_lines:
            lines.append("### Highlights")
            lines.append("")
            lines.extend(highlight_lines)
            lines.append("")

    if isinstance(group_metrics, Mapping):
        by_issue = group_metrics.get("by_issue")
        if isinstance(by_issue, Mapping):
            _render_group_table("Accuracy by Issue", by_issue)  # type: ignore[arg-type]
        by_study = group_metrics.get("by_participant_study")
        if isinstance(by_study, Mapping):
            _render_group_table("Accuracy by Participant Study", by_study)  # type: ignore[arg-type]

    notes = metrics.get("notes")
    if isinstance(notes, str) and notes.strip():
        lines.append("### Notes")
        lines.append("")
        lines.append(notes.strip())
        lines.append("")

    write_markdown_lines(path, lines)


def _fmt_opinion_value(value: object, digits: int = 3) -> str:
    """Format opinion metrics with a consistent fallback.

    :param value: Raw opinion metric value.
    :param digits: Number of decimal places to display.
    :returns: Stringified metric or ``"n/a"`` if unavailable.
    """

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


OPINION_CSV_FIELDS = [
    "study",
    "issue",
    "participants",
    "eligible",
    "mae_after",
    "rmse_after",
    "direction_accuracy",
    "baseline_direction_accuracy",
    "mae_change",
    "rmse_change",
]


def _opinion_summary_lines(
    selected: SweepOutcome, opinion: OpinionEvaluationResult
) -> List[str]:
    """Return headline opinion metrics for the report introduction.

    :param selected: Winning sweep configuration.
    :param opinion: Aggregated opinion evaluation results.
    :returns: Markdown bullet lines describing the evaluation summary.
    """
    lines = [
        (
            f"- **Selected configuration:** `{selected.config.label()}` "
            f"(temperature={selected.config.temperature:.2f}, "
            f"top_p={selected.config.top_p:.2f}, max_tokens="
            f"{selected.config.max_tokens})"
        )
    ]
    total_participants = sum(result.participants for result in opinion.studies.values())
    lines.append(f"- **Participants evaluated:** {total_participants}")
    combined = opinion.combined_metrics or {}
    if combined:
        lines.append(
            "- **Overall metrics:** "
            f"MAE={_fmt_opinion_value(combined.get('mae_after'))}, "
            f"RMSE={_fmt_opinion_value(combined.get('rmse_after'))}, "
            f"Direction accuracy={_fmt_opinion_value(combined.get('direction_accuracy'))}"
        )
    lines.append("")
    return lines


def _build_opinion_table(
    opinion: OpinionEvaluationResult,
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Build the Markdown opinion table and CSV export rows.

    :param opinion: Aggregated opinion evaluation results.
    :returns: Tuple of markdown lines and CSV row dictionaries.
    """
    table_lines: List[str] = []
    csv_rows: List[Dict[str, object]] = []
    header = [
        "Study",
        "Issue",
        "Participants",
        "Eligible",
        "MAE (after)",
        "RMSE (after)",
        "Direction ↑",
        "No-change ↑",
        "Δ Accuracy",
    ]
    table_lines.append("| " + " | ".join(header) + " |")
    table_lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for result in opinion.studies.values():
        metrics = result.metrics
        baseline = result.baseline
        direction_accuracy = metrics.get("direction_accuracy")
        baseline_direction = baseline.get("direction_accuracy")
        delta = (
            float(direction_accuracy) - float(baseline_direction)
            if isinstance(direction_accuracy, (int, float))
            and isinstance(baseline_direction, (int, float))
            else None
        )

        table_lines.append(
            "| "
            + " | ".join(
                [
                    result.study_label,
                    result.issue.replace("_", " "),
                    str(result.participants),
                    str(result.eligible),
                    _fmt_opinion_value(metrics.get("mae_after")),
                    _fmt_opinion_value(metrics.get("rmse_after")),
                    _fmt_opinion_value(direction_accuracy),
                    _fmt_opinion_value(baseline_direction),
                    _fmt_opinion_value(delta),
                ]
            )
            + " |"
        )

        csv_rows.append(
            {
                "study": result.study_key,
                "issue": result.issue,
                "participants": result.participants,
                "eligible": result.eligible,
                "mae_after": metrics.get("mae_after"),
                "rmse_after": metrics.get("rmse_after"),
                "direction_accuracy": direction_accuracy,
                "baseline_direction_accuracy": baseline_direction,
                "mae_change": metrics.get("mae_change"),
                "rmse_change": metrics.get("rmse_change"),
            }
        )

    table_lines.append("")
    return table_lines, csv_rows


def _write_opinion_csv(directory: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Write the opinion metrics CSV and return its path.

    :param directory: Directory where the CSV should be stored.
    :param rows: Iterable of opinion metric rows to serialise.
    :returns: Path to the written CSV file.
    """
    csv_path = directory / "opinion_metrics.csv"
    if not rows:
        if csv_path.exists():
            csv_path.unlink()
        return csv_path
    with csv_path.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=OPINION_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _artifact_lines(opinion: OpinionEvaluationResult, repo_root: Path) -> List[str]:
    """Return Markdown bullets linking to opinion artefacts.

    :param opinion: Opinion evaluation payload containing artefact paths.
    :param repo_root: Repository root used for relative path rendering.
    :returns: Markdown bullet lines referencing metrics, predictions, and QA logs.
    """
    lines: List[str] = []
    for result in opinion.studies.values():
        metrics_rel = _relative_path(repo_root, result.metrics_path)
        preds_rel = _relative_path(repo_root, result.predictions_path)
        qa_rel = _relative_path(repo_root, result.qa_log_path)
        lines.append(
            f"- `{result.study_label}` metrics: `{metrics_rel}` "
            f"predictions: `{preds_rel}` QA log: `{qa_rel}`"
        )
    lines.append("")
    return lines


def _relative_path(root: Path, target: Path) -> Path:
    """Return ``target`` relative to ``root`` when possible.

    :param root: Root directory for relative resolution.
    :param target: Filesystem path to convert.
    :returns: Relative path when ``target`` lives under ``root``; otherwise ``target``.
    """
    try:
        return target.relative_to(root)
    except ValueError:
        return target


def _write_opinion_report(
    directory: Path,
    *,
    selected: SweepOutcome,
    opinion: OpinionEvaluationResult | None,
) -> None:
    """Create the opinion regression summary document.

    :param directory: Destination directory for the opinion report.
    :param selected: Winning sweep configuration.
    :param opinion: Opinion evaluation payload (may be ``None``).
    :returns: ``None``.
    """

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "README.md"
    lines: List[str] = ["# GPT-4o Opinion Shift", ""]

    if opinion is None or not opinion.studies:
        lines.append("No opinion evaluations were produced during this pipeline invocation.")
        lines.append("")
        write_markdown_lines(path, lines)
        return

    lines.extend(_opinion_summary_lines(selected, opinion))

    table_lines, csv_rows = _build_opinion_table(opinion)
    lines.extend(table_lines)
    lines.append("`opinion_metrics.csv` summarises per-study metrics.")
    lines.append("")

    _write_opinion_csv(directory, csv_rows)

    repo_root = _repo_root()
    lines.append("### Artefacts")
    lines.append("")
    lines.extend(_artifact_lines(opinion, repo_root))

    write_markdown_lines(path, lines)


def _write_reports(
    *,
    reports_dir: Path,
    outcomes: Sequence[SweepOutcome],
    selected: SweepOutcome,
    final_metrics: Mapping[str, object],
    opinion_result: OpinionEvaluationResult | None,
) -> None:
    """Regenerate catalog, sweep, opinion, and next-video reports.

    :param reports_dir: Root reports directory.
    :param outcomes: All sweep outcomes produced during the run.
    :param selected: Outcome chosen as the final configuration.
    :param final_metrics: Metrics payload from the promoted evaluation.
    :param opinion_result: Opinion evaluation payload (may be ``None``).
    :returns: ``None``.
    """

    _write_catalog_report(reports_dir)
    _write_sweep_report(reports_dir / "hyperparameter_tuning", outcomes, selected)
    _write_next_video_report(reports_dir / "next_video", selected, final_metrics)
    _write_opinion_report(reports_dir / "opinion", selected=selected, opinion=opinion_result)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Execute sweeps, selection, final evaluation, and report regeneration.

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

    configs = _build_sweep_configs(args)
    LOGGER.info("Planned %d GPT-4o configurations.", len(configs))
    LOGGER.info("Hyper-parameter sweeps evaluate the validation split only (no training stage).")

    if args.dry_run:
        for config in configs:
            LOGGER.info("[DRY-RUN] would evaluate config=%s", config.label())
        return

    base_cli = _build_base_cli_args(args, cache_dir=paths.cache_dir)

    outcomes = _run_sweeps(
        configs=configs,
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=paths.sweep_dir,
    )
    selected = _select_best(outcomes)
    LOGGER.info("Selected configuration: %s", selected.config.label())

    final_dir, final_metrics = _promote_sweep_results(
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
    _write_reports(
        reports_dir=paths.reports_dir,
        outcomes=outcomes,
        selected=selected,
        final_metrics=final_metrics,
        opinion_result=opinion_result,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
