"""End-to-end orchestration for the GPT-4o slate baseline.

This module mirrors the structure of the KNN and XGBoost pipelines:

1. Hyper-parameter sweeps over temperature / max_token settings.
2. Selection of the best-performing configuration.
3. Final evaluation run with the chosen configuration.
4. Markdown report regeneration under ``reports/gpt4o``.

Each sweep delegates to :mod:`gpt4o.cli` to avoid drift between the public CLI
and the scripted pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .cli import build_parser as build_gpt_parser
from .evaluate import run_eval

LOGGER = logging.getLogger("gpt4o.pipeline")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    """Describe a GPT-4o configuration evaluated during sweeps."""

    temperature: float
    max_tokens: int

    def label(self) -> str:
        """Return a filesystem-friendly identifier."""

        temp_token = f"temp{self.temperature:g}".replace(".", "p")
        return f"{temp_token}_tok{self.max_tokens}"

    def cli_args(self) -> List[str]:
        """Return CLI overrides encoding this configuration."""

        return [
            "--temperature",
            str(self.temperature),
            "--max_tokens",
            str(self.max_tokens),
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


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> Tuple[argparse.Namespace, List[str]]:
    """Parse pipeline arguments while preserving additional CLI options."""

    parser = argparse.ArgumentParser(
        description="Hyper-parameter sweeps, selection, and reporting for the GPT-4o slate baseline."
    )
    parser.add_argument("--dataset", default=None, help="Dataset path or HuggingFace dataset id.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Root directory for GPT-4o outputs (default: <repo>/models/gpt4o).",
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
    parser.add_argument(
        "--studies",
        default="",
        help="Comma-separated participant study identifiers to filter (defaults to all studies).",
    )
    parser.add_argument(
        "--temperature-grid",
        default="0.0,0.2,0.4",
        help="Comma-separated temperatures explored during sweeps.",
    )
    parser.add_argument(
        "--max-tokens-grid",
        default="32,48",
        help="Comma-separated max_token values explored during sweeps.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the pipeline logger.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting of existing sweep and evaluation directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the planned actions without executing sweeps or evaluations.",
    )
    parsed, extra = parser.parse_known_args(argv)
    return parsed, list(extra)


def _repo_root() -> Path:
    """Return the repository root used to derive default directories."""

    return Path(__file__).resolve().parents[2]


def _default_out_dir(root: Path) -> Path:
    """Return the default pipeline output directory rooted at ``root``."""

    return root / "models" / "gpt4o"


def _default_cache_dir(root: Path) -> Path:
    """Return the default Hugging Face cache location under ``root``."""

    return root / ".cache" / "huggingface" / "gpt4o"


def _default_reports_dir(root: Path) -> Path:
    """Return the default reports directory rooted at ``root``."""

    return root / "reports" / "gpt4o"


def _split_tokens(raw: str) -> List[str]:
    """Split a comma-separated string into trimmed, non-empty tokens."""

    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _parse_float_grid(raw: str, fallback: float) -> List[float]:
    """Parse a comma-separated list of floats, falling back to ``fallback``."""

    tokens = _split_tokens(raw)
    values: List[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            LOGGER.warning("Ignoring invalid temperature token '%s'", token)
    return values or [fallback]


def _parse_int_grid(raw: str, fallback: int) -> List[int]:
    """Parse a comma-separated list of integers, falling back to ``fallback``."""

    tokens = _split_tokens(raw)
    values: List[int] = []
    for token in tokens:
        try:
            values.append(int(token))
        except ValueError:
            LOGGER.warning("Ignoring invalid max_tokens token '%s'", token)
    return values or [fallback]


def _build_sweep_configs(args: argparse.Namespace) -> List[SweepConfig]:
    """Construct the Cartesian product of temperature and max-token sweeps."""

    temperatures = _parse_float_grid(args.temperature_grid, 0.0)
    max_tokens_values = _parse_int_grid(args.max_tokens_grid, 32)
    configs: List[SweepConfig] = []
    for temp in temperatures:
        for max_tokens in max_tokens_values:
            configs.append(SweepConfig(temperature=temp, max_tokens=max_tokens))
    return configs


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _run_gpt_cli(cli_args: Sequence[str]) -> None:
    """Invoke the GPT-4o CLI entry point with the provided arguments."""

    parser = build_gpt_parser()
    namespace = parser.parse_args(list(cli_args))
    run_eval(namespace)


def _load_metrics(path: Path) -> Mapping[str, object]:
    """Load evaluation metrics from ``path``."""

    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_sweeps(
    *,
    configs: Sequence[SweepConfig],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    sweep_dir: Path,
) -> List[SweepOutcome]:
    """Run GPT-4o sweeps for each configuration and collect outcomes."""

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
        metrics = _load_metrics(metrics_path)
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
    """Return the best sweep outcome by accuracy, parsed rate, then format rate."""

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
    """Run a final evaluation for the selected configuration."""

    run_dir = out_dir / config.label()
    cli_args: List[str] = []
    cli_args.extend(base_cli)
    cli_args.extend(config.cli_args())
    cli_args.extend(["--out_dir", str(run_dir)])
    cli_args.extend(extra_cli)
    LOGGER.info("[FINAL] config=%s -> %s", config.label(), run_dir)
    _run_gpt_cli(cli_args)
    metrics_path = run_dir / "metrics.json"
    metrics = _load_metrics(metrics_path)
    return run_dir, metrics


def _format_rate(value: float) -> str:
    """Format a numeric rate with three decimal places."""

    return f"{value:.3f}"


def _write_catalog_report(reports_dir: Path) -> None:
    """Create the top-level GPT-4o report catalog README."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "README.md"
    lines = [
        "# GPT-4o Report Catalog",
        "",
        "Generated artefacts for the GPT-4o slate-selection baseline:",
        "",
        "- `next_video/` – summary metrics and fairness cuts for the selected configuration.",
        "- `hyperparameter_tuning/` – sweep results across temperature and max token settings.",
        "",
        "Model predictions and metrics JSON files live under `models/gpt4o/`.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sweep_report(directory: Path, outcomes: Sequence[SweepOutcome], selected: SweepOutcome) -> None:
    """Write the hyper-parameter sweep report summarising all outcomes."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "README.md"
    lines: List[str] = []
    lines.append("# GPT-4o Hyper-parameter Sweep")
    lines.append("")
    if not outcomes:
        lines.append("No sweep runs were executed.")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    lines.append(
        "The table below captures accuracy on eligible slates plus formatting/parse rates for "
        "each temperature/max-token configuration. The selected configuration is marked with ✓."
    )
    lines.append("")
    lines.append("| Config | Temperature | Max tokens | Accuracy ↑ | Parsed ↑ | Formatted ↑ | Selected |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for outcome in outcomes:
        mark = "✓" if outcome.config == selected.config else ""
        lines.append(
            f"| `{outcome.config.label()}` | {outcome.config.temperature:.2f} | "
            f"{outcome.config.max_tokens} | {_format_rate(outcome.accuracy)} | "
            f"{_format_rate(outcome.parsed_rate)} | {_format_rate(outcome.format_rate)} | {mark} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_next_video_report(
    directory: Path,
    selected: SweepOutcome,
    metrics: Mapping[str, object],
) -> None:
    """Write the next-video evaluation report for the selected configuration."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "README.md"
    lines: List[str] = []
    lines.append("# GPT-4o Next-Video Baseline")
    lines.append("")
    lines.append(
        f"- **Selected configuration:** `{selected.config.label()}` "
        f"(temperature={selected.config.temperature:.2f}, max_tokens={selected.config.max_tokens})"
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
        """Append a Markdown table capturing group-level metrics."""
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
            lines.append(
                f"| {group or 'unspecified'} | {seen} | {eligible} | {accuracy} | {parsed_rate} | {format_rate} |"
            )
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

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reports(
    *,
    reports_dir: Path,
    outcomes: Sequence[SweepOutcome],
    selected: SweepOutcome,
    final_metrics: Mapping[str, object],
) -> None:
    """Regenerate catalog, sweep, and next-video reports."""

    _write_catalog_report(reports_dir)
    _write_sweep_report(reports_dir / "hyperparameter_tuning", outcomes, selected)
    _write_next_video_report(reports_dir / "next_video", selected, final_metrics)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Execute sweeps, selection, final evaluation, and report regeneration."""

    args, extra_cli = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    root = _repo_root()
    out_dir = Path(args.out_dir or _default_out_dir(root))
    cache_dir = str(args.cache_dir or _default_cache_dir(root))
    reports_dir = Path(args.reports_dir or _default_reports_dir(root))
    sweep_dir = Path(args.sweep_dir or (out_dir / "sweeps"))
    final_out_dir = out_dir / "next_video"
    final_out_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Propagate cache configuration to the subprocess environment.
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    os.environ.setdefault("HF_HOME", cache_dir)

    configs = _build_sweep_configs(args)
    LOGGER.info("Planned %d GPT-4o configurations.", len(configs))

    if args.dry_run:
        for config in configs:
            LOGGER.info("[DRY-RUN] would evaluate config=%s", config.label())
        return

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
    if args.overwrite:
        base_cli.append("--overwrite")

    outcomes = _run_sweeps(
        configs=configs,
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=sweep_dir,
    )
    selected = _select_best(outcomes)
    LOGGER.info("Selected configuration: %s", selected.config.label())

    final_dir, final_metrics = _run_final_evaluation(
        config=selected.config,
        base_cli=base_cli,
        extra_cli=extra_cli,
        out_dir=final_out_dir,
    )

    LOGGER.info("Final metrics stored under %s", final_dir)
    _write_reports(
        reports_dir=reports_dir,
        outcomes=outcomes,
        selected=selected,
        final_metrics=final_metrics,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
