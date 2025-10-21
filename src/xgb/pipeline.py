"""High-level orchestration for the XGBoost baselines."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .cli import build_parser as build_xgb_parser
from .data import DEFAULT_DATASET_SOURCE, issues_in_dataset, load_dataset_source
from .evaluate import run_eval
from .model import XGBoostBoosterParams
from .opinion import OpinionTrainConfig, run_opinion_eval

LOGGER = logging.getLogger("xgb.pipeline")


@dataclass(frozen=True)
class SweepConfig:
    """Hyper-parameter configuration evaluated during sweeps."""

    learning_rate: float
    max_depth: int
    n_estimators: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float

    def label(self) -> str:
        """Filesystem-friendly label."""

        return (
            f"lr{self.learning_rate:g}_depth{self.max_depth}_"
            f"estim{self.n_estimators}_sub{self.subsample:g}_"
            f"col{self.colsample_bytree:g}_l2{self.reg_lambda:g}_l1{self.reg_alpha:g}"
        ).replace(".", "p")

    def booster_params(self, tree_method: str) -> XGBoostBoosterParams:
        return XGBoostBoosterParams(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method=tree_method,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
        )

    def cli_args(self, tree_method: str) -> List[str]:
        return [
            "--xgb_learning_rate",
            str(self.learning_rate),
            "--xgb_max_depth",
            str(self.max_depth),
            "--xgb_n_estimators",
            str(self.n_estimators),
            "--xgb_subsample",
            str(self.subsample),
            "--xgb_colsample_bytree",
            str(self.colsample_bytree),
            "--xgb_tree_method",
            tree_method,
            "--xgb_reg_lambda",
            str(self.reg_lambda),
            "--xgb_reg_alpha",
            str(self.reg_alpha),
        ]


@dataclass(frozen=True)
class IssueSpec:
    """Descriptor for a single issue present in the dataset."""

    name: str
    label: str

    @property
    def slug(self) -> str:
        return self.name.replace(" ", "_")


@dataclass
class SweepOutcome:
    """Metrics captured for a (issue, config) combination."""

    issue: IssueSpec
    config: SweepConfig
    accuracy: float
    coverage: float
    evaluated: int
    metrics_path: Path
    metrics: Mapping[str, object]


@dataclass
class IssueSelection:
    """Selected configuration for the final evaluation of an issue."""

    issue: IssueSpec
    outcome: SweepOutcome

    @property
    def config(self) -> SweepConfig:
        return self.outcome.config


def _parse_args(argv: Sequence[str] | None) -> Tuple[argparse.Namespace, List[str]]:
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
        help="Directory for hyper-parameter sweep outputs (default: <out-dir>/sweeps).",
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
    parser.add_argument(
        "--studies",
        default="",
        help="Comma-separated opinion study keys (defaults to all studies).",
    )
    parser.add_argument(
        "--extra-text-fields",
        default="",
        help="Comma-separated extra text fields appended to prompt documents.",
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
        default="hist",
        help="Tree construction algorithm passed to XGBoost.",
    )
    parser.add_argument(
        "--learning-rate-grid",
        default="0.05,0.1,0.2",
        help="Comma-separated learning rates explored during sweeps.",
    )
    parser.add_argument(
        "--max-depth-grid",
        default="4,6",
        help="Comma-separated integer depths explored during sweeps.",
    )
    parser.add_argument(
        "--n-estimators-grid",
        default="200,400",
        help="Comma-separated boosting round counts explored during sweeps.",
    )
    parser.add_argument(
        "--subsample-grid",
        default="0.7,0.9",
        help="Comma-separated subsample ratios explored during sweeps.",
    )
    parser.add_argument(
        "--colsample-grid",
        default="0.7,1.0",
        help="Comma-separated column subsample ratios explored during sweeps.",
    )
    parser.add_argument(
        "--reg-lambda-grid",
        default="1.0",
        help="Comma-separated L2 regularisation weights explored during sweeps.",
    )
    parser.add_argument(
        "--reg-alpha-grid",
        default="0.0,0.5",
        help="Comma-separated L1 regularisation weights explored during sweeps.",
    )
    parser.add_argument(
        "--save-model-dir",
        default=None,
        help="Optional directory used to persist the final slate models.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the pipeline logger.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the planned actions without executing sweeps or evaluations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing sweep and evaluation outputs.",
    )

    parsed, extra = parser.parse_known_args(argv)
    return parsed, list(extra)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_out_dir(root: Path) -> Path:
    return root / "models" / "xgb"


def _default_cache_dir(root: Path) -> Path:
    return root / ".cache" / "huggingface" / "xgb"


def _default_reports_dir(root: Path) -> Path:
    return root / "reports" / "xgb"


def _split_tokens(raw: str) -> List[str]:
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _build_sweep_configs(args: argparse.Namespace) -> List[SweepConfig]:
    lr_values = [float(x) for x in _split_tokens(args.learning_rate_grid)]
    depth_values = [int(x) for x in _split_tokens(args.max_depth_grid)]
    estimator_values = [int(x) for x in _split_tokens(args.n_estimators_grid)]
    subsample_values = [float(x) for x in _split_tokens(args.subsample_grid)]
    colsample_values = [float(x) for x in _split_tokens(args.colsample_grid)]
    reg_lambda_values = [float(x) for x in _split_tokens(args.reg_lambda_grid)]
    reg_alpha_values = [float(x) for x in _split_tokens(args.reg_alpha_grid)]

    configs: List[SweepConfig] = []
    for values in product(
        lr_values,
        depth_values,
        estimator_values,
        subsample_values,
        colsample_values,
        reg_lambda_values,
        reg_alpha_values,
    ):
        configs.append(
            SweepConfig(
                learning_rate=values[0],
                max_depth=values[1],
                n_estimators=values[2],
                subsample=values[3],
                colsample_bytree=values[4],
                reg_lambda=values[5],
                reg_alpha=values[6],
            )
        )
    return configs


def _resolve_issue_specs(
    *,
    dataset: str,
    cache_dir: str,
    requested: Sequence[str],
) -> List[IssueSpec]:
    ds = load_dataset_source(dataset, cache_dir)
    available = issues_in_dataset(ds)
    selected = available if not requested else [issue for issue in available if issue in requested]
    specs = [IssueSpec(name=issue, label=issue.replace("_", " ").title()) for issue in selected]
    missing = set(requested) - {spec.name for spec in specs}
    if missing:
        raise ValueError(f"Unknown issues requested: {', '.join(sorted(missing))}")
    return specs


def _run_xgb_cli(args: Sequence[str]) -> None:
    parser = build_xgb_parser()
    namespace = parser.parse_args(list(args))
    run_eval(namespace)


def _load_metrics(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_sweeps(
    *,
    issues: Sequence[IssueSpec],
    configs: Sequence[SweepConfig],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    sweep_dir: Path,
    tree_method: str,
    overwrite: bool,
) -> List[SweepOutcome]:
    outcomes: List[SweepOutcome] = []
    for config in configs:
        for issue in issues:
            run_root = sweep_dir / issue.slug / config.label()
            run_root.mkdir(parents=True, exist_ok=True)
            cli_args: List[str] = []
            cli_args.extend(base_cli)
            cli_args.extend(config.cli_args(tree_method))
            cli_args.extend(["--issues", issue.name])
            cli_args.extend(["--out_dir", str(run_root)])
            cli_args.extend(extra_cli)
            LOGGER.info(
                "[SWEEP] issue=%s config=%s",
                issue.name,
                config.label(),
            )
            _run_xgb_cli(cli_args)
            metrics_path = run_root / issue.slug / "metrics.json"
            metrics = _load_metrics(metrics_path)
            outcomes.append(
                SweepOutcome(
                    issue=issue,
                    config=config,
                    accuracy=float(metrics.get("accuracy", 0.0)),
                    coverage=float(metrics.get("coverage", 0.0)),
                    evaluated=int(metrics.get("evaluated", 0)),
                    metrics_path=metrics_path,
                    metrics=metrics,
                )
            )
    return outcomes


def _select_best_configs(outcomes: Sequence[SweepOutcome]) -> Dict[str, IssueSelection]:
    selections: Dict[str, IssueSelection] = {}

    for outcome in outcomes:
        current = selections.get(outcome.issue.name)
        if current is None:
            selections[outcome.issue.name] = IssueSelection(issue=outcome.issue, outcome=outcome)
            continue
        incumbent = current.outcome
        if outcome.accuracy > incumbent.accuracy + 1e-9:
            selections[outcome.issue.name] = IssueSelection(issue=outcome.issue, outcome=outcome)
            continue
        if incumbent.accuracy - outcome.accuracy <= 1e-9:
            if outcome.coverage > incumbent.coverage + 1e-9:
                selections[outcome.issue.name] = IssueSelection(
                    issue=outcome.issue,
                    outcome=outcome,
                )
            elif (
                abs(outcome.coverage - incumbent.coverage) <= 1e-9
                and outcome.evaluated > incumbent.evaluated
            ):
                selections[outcome.issue.name] = IssueSelection(
                    issue=outcome.issue,
                    outcome=outcome,
                )
    return selections


def _run_final_evaluations(
    *,
    selections: Mapping[str, IssueSelection],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    out_dir: Path,
    tree_method: str,
    save_model_dir: Path | None,
    overwrite: bool,
) -> Dict[str, Mapping[str, object]]:
    metrics_by_issue: Dict[str, Mapping[str, object]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for issue_name, selection in selections.items():
        cli_args: List[str] = []
        cli_args.extend(base_cli)
        cli_args.extend(selection.config.cli_args(tree_method))
        cli_args.extend(["--issues", issue_name])
        cli_args.extend(["--out_dir", str(out_dir)])
        if save_model_dir is not None:
            cli_args.extend(["--save_model", str(save_model_dir)])
        cli_args.extend(extra_cli)
        LOGGER.info(
            "[FINAL] issue=%s config=%s",
            issue_name,
            selection.config.label(),
        )
        _run_xgb_cli(cli_args)
        metrics_path = out_dir / selection.issue.slug / "metrics.json"
        metrics_by_issue[issue_name] = _load_metrics(metrics_path)
    return metrics_by_issue


def _run_opinion_stage(
    *,
    selections: Mapping[str, IssueSelection],
    args: argparse.Namespace,
    base_out_dir: Path,
    extra_fields: Sequence[str],
    studies: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    if not selections:
        LOGGER.warning("Skipping opinion stage; no selections available.")
        return {}

    opinion_out_dir = base_out_dir / "opinion"
    opinion_config = OpinionTrainConfig(
        max_participants=args.opinion_max_participants,
        seed=args.seed,
        max_features=args.max_features if args.max_features > 0 else None,
        booster=next(iter(selections.values())).config.booster_params(args.tree_method),
    )
    return run_opinion_eval(
        dataset=args.dataset or DEFAULT_DATASET_SOURCE,
        cache_dir=args.cache_dir,
        out_dir=opinion_out_dir,
        feature_space="tfidf",
        extra_fields=extra_fields,
        train_config=opinion_config,
        studies=studies,
        overwrite=args.overwrite,
    )


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _write_reports(
    *,
    reports_dir: Path,
    sweep_dir: Path,
    outcomes: Sequence[SweepOutcome],
    selections: Mapping[str, IssueSelection],
    final_metrics: Mapping[str, Mapping[str, object]],
    opinion_metrics: Mapping[str, Mapping[str, object]],
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    _write_catalog_report(reports_dir)
    _write_hyperparameter_report(reports_dir / "hyperparameter_tuning", outcomes, selections)
    _write_next_video_report(reports_dir / "next_video", final_metrics)
    _write_opinion_report(reports_dir / "opinion", opinion_metrics)


def _write_catalog_report(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "README.md"
    lines: List[str] = []
    lines.append("# XGBoost Report Catalog")
    lines.append("")
    lines.append("Generated artefacts for the XGBoost baselines:")
    lines.append("")
    lines.append("- `next_video/` – slate-ranking accuracy for the selected configuration.")
    lines.append("- `opinion/` – post-study opinion regression results.")
    lines.append("- `hyperparameter_tuning/` – notes from the sweep that selected the configuration.")
    lines.append("")
    lines.append("Model checkpoints and raw metrics live under `models/xgb/`.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_hyperparameter_report(
    directory: Path,
    outcomes: Sequence[SweepOutcome],
    selections: Mapping[str, IssueSelection],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "README.md"
    lines: List[str] = []
    lines.append("# Hyper-parameter Tuning")
    lines.append("")
    lines.append(
        "This summary lists the XGBoost configurations explored for each issue. The selected configuration is highlighted in bold."
    )
    lines.append("")
    per_issue: Dict[str, List[SweepOutcome]] = {}
    for outcome in outcomes:
        per_issue.setdefault(outcome.issue.name, []).append(outcome)
    for issue_name, issue_outcomes in sorted(per_issue.items()):
        selection = selections.get(issue_name)
        lines.append(f"## {issue_name.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Config | Accuracy ↑ | Coverage ↑ | Evaluated |")
        lines.append("| --- | ---: | ---: | ---: |")
        for outcome in sorted(
            issue_outcomes,
            key=lambda item: item.accuracy,
            reverse=True,
        ):
            label = outcome.config.label()
            formatted = (
                f"**{label}**" if selection and outcome.config == selection.config else label
            )
            lines.append(
                f"| {formatted} | {_format_float(outcome.accuracy)} | "
                f"{_format_float(outcome.coverage)} | {outcome.evaluated} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_next_video_report(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "README.md"
    lines: List[str] = []
    lines.append("# XGBoost Next-Video Baseline")
    lines.append("")
    lines.append("Accuracy on the validation split for the selected slate configuration.")
    lines.append("")
    lines.append("| Issue | Accuracy ↑ | Coverage ↑ | Evaluated |")
    lines.append("| --- | ---: | ---: | ---: |")
    for issue_name, summary in sorted(metrics.items()):
        lines.append(
            f"| {issue_name.replace('_', ' ').title()} | "
            f"{_format_float(float(summary.get('accuracy', 0.0)))} | "
            f"{_format_float(float(summary.get('coverage', 0.0)))} | "
            f"{int(summary.get('evaluated', 0))} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_opinion_report(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "README.md"
    lines: List[str] = []
    lines.append("# XGBoost Opinion Regression")
    lines.append("")
    if not metrics:
        lines.append("No opinion runs were produced during this pipeline invocation.")
        lines.append("")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.append("MAE / RMSE / R² scores for predicting the post-study opinion index.")
    lines.append("")
    lines.append("| Study | Participants | MAE ↓ | RMSE ↓ | R² ↑ | Baseline MAE ↓ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for study_key, summary in sorted(metrics.items()):
        label = summary.get("label", study_key)
        metrics_block = summary.get("metrics", {})
        baseline = summary.get("baseline", {})
        lines.append(
            f"| {label} | {int(summary.get('n_participants', 0))} | "
            f"{_format_float(float(metrics_block.get('mae_after', 0.0)))} | "
            f"{_format_float(float(metrics_block.get('rmse_after', 0.0)))} | "
            f"{_format_float(float(metrics_block.get('r2_after', 0.0)))} | "
            f"{_format_float(float(baseline.get('mae_before', 0.0)))} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args, extra_cli = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    root = _repo_root()
    dataset = args.dataset or str(root / "data" / "cleaned_grail")
    cache_dir = args.cache_dir or str(_default_cache_dir(root))
    out_dir = Path(args.out_dir or _default_out_dir(root))
    sweep_dir = Path(args.sweep_dir or (out_dir / "sweeps"))
    reports_dir = Path(args.reports_dir or _default_reports_dir(root))

    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    os.environ.setdefault("HF_HOME", cache_dir)

    issues = _resolve_issue_specs(
        dataset=dataset,
        cache_dir=cache_dir,
        requested=_split_tokens(args.issues),
    )
    extra_fields = _split_tokens(args.extra_text_fields)
    studies = _split_tokens(args.studies)

    base_cli: List[str] = [
        "--fit_model",
        "--dataset",
        dataset,
        "--cache_dir",
        cache_dir,
        "--max_train",
        str(args.max_train),
        "--max_features",
        str(args.max_features),
        "--eval_max",
        str(args.eval_max),
        "--seed",
        str(args.seed),
    ]
    if extra_fields:
        base_cli.extend(["--extra_text_fields", ",".join(extra_fields)])
    base_cli.append("--log_level")
    base_cli.append(args.log_level.upper())
    if args.overwrite:
        base_cli.append("--overwrite")

    configs = _build_sweep_configs(args)

    if args.dry_run:
        LOGGER.info("Dry-run mode. Planned %d configurations.", len(configs))
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    outcomes = _run_sweeps(
        issues=issues,
        configs=configs,
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=sweep_dir,
        tree_method=args.tree_method,
        overwrite=args.overwrite,
    )
    selections = _select_best_configs(outcomes)
    if not selections:
        raise RuntimeError("Failed to select a configuration for any issue.")
    LOGGER.info("Selected configurations: %s", ", ".join(selections.keys()))

    final_metrics = _run_final_evaluations(
        selections=selections,
        base_cli=base_cli,
        extra_cli=extra_cli,
        out_dir=out_dir / "next_video",
        tree_method=args.tree_method,
        save_model_dir=Path(args.save_model_dir) if args.save_model_dir else None,
        overwrite=args.overwrite,
    )

    opinion_metrics = _run_opinion_stage(
        selections=selections,
        args=args,
        base_out_dir=out_dir,
        extra_fields=extra_fields,
        studies=studies,
    )

    _write_reports(
        reports_dir=reports_dir,
        sweep_dir=sweep_dir,
        outcomes=outcomes,
        selections=selections,
        final_metrics=final_metrics,
        opinion_metrics=opinion_metrics,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
