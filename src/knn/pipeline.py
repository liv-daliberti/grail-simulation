"""High-level orchestration for the KNN baselines.

Running :mod:`knn.pipeline` executes the full workflow requested by
``training/training-knn.sh``:

1. Hyper-parameter sweeps for TF-IDF and Word2Vec feature spaces.
2. Aggregate-selection of the best configuration per feature space.
3. Final slate evaluation runs with the selected configurations.
4. Opinion-regression evaluation using the same feature settings.
5. Regeneration of all Markdown reports under ``reports/knn/``.

The module reuses the existing :mod:`knn.cli` entry points to avoid code drift
between the scripted workflow and the public command-line interface.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from knn.cli import build_parser
from knn.data import DEFAULT_DATASET_SOURCE, issues_in_dataset, load_dataset_source
from knn.evaluate import run_eval

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    """Describe a single hyper-parameter configuration to evaluate."""

    feature_space: str
    metric: str
    text_fields: Tuple[str, ...]
    word2vec_size: int | None = None
    word2vec_window: int | None = None
    word2vec_min_count: int | None = None
    word2vec_epochs: int | None = None
    word2vec_workers: int | None = None

    @property
    def label(self) -> str:
        """Return a filesystem-friendly identifier for this configuration."""

        text_label = "none"
        if self.text_fields:
            text_label = "_".join(field.replace("_", "") for field in self.text_fields)
        parts = [f"metric-{self.metric}", f"text-{text_label}"]
        if self.feature_space == "word2vec":
            parts.extend(
                [
                    f"sz{self.word2vec_size}",
                    f"win{self.word2vec_window}",
                    f"min{self.word2vec_min_count}",
                ]
            )
        return "_".join(parts)

    def cli_args(self, *, word2vec_model_dir: Path | None) -> List[str]:
        """Return CLI overrides implementing this configuration."""

        args = [
            "--feature-space",
            self.feature_space,
            "--knn-metric",
            self.metric,
            "--knn-text-fields",
            ",".join(self.text_fields) if self.text_fields else "",
        ]
        if self.feature_space == "word2vec":
            if (
                self.word2vec_size is None
                or self.word2vec_window is None
                or self.word2vec_min_count is None
            ):
                raise ValueError("Word2Vec configuration must define size/window/min_count")
            args.extend(
                [
                    "--word2vec-size",
                    str(self.word2vec_size),
                    "--word2vec-window",
                    str(self.word2vec_window),
                    "--word2vec-min-count",
                    str(self.word2vec_min_count),
                ]
            )
            if self.word2vec_epochs is not None:
                args.extend(["--word2vec-epochs", str(self.word2vec_epochs)])
            if self.word2vec_workers is not None:
                args.extend(["--word2vec-workers", str(self.word2vec_workers)])
            if word2vec_model_dir is not None:
                args.extend(["--word2vec-model-dir", str(word2vec_model_dir)])
        return args


@dataclass
class SweepOutcome:
    """Captures metrics for a configuration/issue pair."""

    issue: str
    feature_space: str
    config: SweepConfig
    accuracy: float
    best_k: int
    eligible: int
    metrics_path: Path
    metrics: Mapping[str, object]


@dataclass
class SweepSelection:
    """Best configuration chosen for a feature space."""

    config: SweepConfig
    per_issue: "OrderedDict[str, SweepOutcome]"
    weighted_accuracy: float

    @property
    def primary_issue(self) -> str:
        """Return the issue contributing the largest eligible population."""

        return max(self.per_issue.items(), key=lambda item: item[1].eligible)[0]

    @property
    def primary_best_k(self) -> int:
        """Return the best ``k`` for the primary issue."""

        return self.per_issue[self.primary_issue].best_k


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> Tuple[argparse.Namespace, List[str]]:
    """Parse known pipeline arguments while preserving passthrough flags."""

    parser = argparse.ArgumentParser(
        description="End-to-end sweeps, evaluation, and report regeneration for the KNN baselines."
    )
    parser.add_argument("--dataset", default=None, help="Dataset path or HuggingFace dataset id.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Root directory for KNN outputs (default: <repo>/models/knn).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HF datasets cache directory (default: <repo>/.cache/huggingface/knn).",
    )
    parser.add_argument(
        "--word2vec-model-dir",
        default=None,
        help="Directory for persisted Word2Vec models (default: <out-dir>/word2vec_models).",
    )
    parser.add_argument(
        "--issues",
        default="",
        help="Comma-separated list of issues to evaluate. Defaults to all issues in the dataset.",
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help="Directory for hyper-parameter sweeps (default: <out-dir>/sweeps).",
    )
    parser.add_argument(
        "--k-sweep",
        default=None,
        help="Comma-separated list of k values tested during sweeps and final runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and log the workflow without launching sweeps or evaluations.",
    )
    parsed, extra = parser.parse_known_args(argv)
    return parsed, extra


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Return the repository root (two parents above this module)."""

    return Path(__file__).resolve().parents[2]


def _default_dataset(root: Path) -> str:
    return str(root / "data" / "cleaned_grail")


def _default_cache_dir(root: Path) -> str:
    return str(root / ".cache" / "huggingface" / "knn")


def _default_out_dir(root: Path) -> str:
    return str(root / "models" / "knn")


def _default_word2vec_workers() -> int:
    env_value = os.environ.get("WORD2VEC_WORKERS")
    if env_value:
        return max(1, int(env_value))
    max_workers = int(os.environ.get("MAX_WORD2VEC_WORKERS", "40"))
    n_cpus = os.cpu_count() or 1
    return max(1, min(n_cpus, max_workers))


def _snake_to_title(value: str) -> str:
    return value.replace("_", " ").title()


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_sweep_configs(
    *,
    word2vec_epochs: int,
    word2vec_workers: int,
) -> List[SweepConfig]:
    """Return the grid of configurations evaluated during sweeps."""

    text_options: Tuple[Tuple[str, ...], ...] = ((), ("viewer_profile", "state_text"))
    tfidf_metrics = ("cosine", "l2")
    word2vec_metrics = ("cosine", "l2")
    word2vec_sizes = tuple(
        int(token)
        for token in os.environ.get("WORD2VEC_SWEEP_SIZES", "128,256").split(",")
        if token.strip()
    )
    word2vec_windows = tuple(
        int(token)
        for token in os.environ.get("WORD2VEC_SWEEP_WINDOWS", "5,10").split(",")
        if token.strip()
    )
    word2vec_min_counts = tuple(
        int(token)
        for token in os.environ.get("WORD2VEC_SWEEP_MIN_COUNTS", "1").split(",")
        if token.strip()
    )

    configs: List[SweepConfig] = []

    for metric in tfidf_metrics:
        for fields in text_options:
            configs.append(
                SweepConfig(
                    feature_space="tfidf",
                    metric=metric,
                    text_fields=fields,
                )
            )

    for metric in word2vec_metrics:
        for fields in text_options:
            for size in word2vec_sizes:
                for window in word2vec_windows:
                    for min_count in word2vec_min_counts:
                        configs.append(
                            SweepConfig(
                                feature_space="word2vec",
                                metric=metric,
                                text_fields=fields,
                                word2vec_size=size,
                                word2vec_window=window,
                                word2vec_min_count=min_count,
                                word2vec_epochs=word2vec_epochs,
                                word2vec_workers=word2vec_workers,
                            )
                        )
    return configs


def _run_knn_cli(argv: Sequence[str]) -> None:
    """Execute the KNN CLI entrypoint with the supplied arguments."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "task", "slate") == "slate":
        run_eval(args)
        return
    if args.task == "opinion":
        from .opinion import run_opinion_eval

        run_opinion_eval(args)
        return
    raise ValueError(f"Unsupported task '{args.task}'.")


def _load_metrics(run_dir: Path, issue_slug: str) -> Tuple[Mapping[str, object], Path]:
    """Load the evaluation metrics JSON for ``issue_slug``."""

    metrics_path = run_dir / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle), metrics_path


def _load_opinion_metrics(out_dir: Path, feature_space: str) -> Dict[str, Mapping[str, object]]:
    """Return metrics keyed by study for the opinion task."""

    result: Dict[str, Mapping[str, object]] = {}
    base_dir = out_dir / "opinion" / feature_space
    if not base_dir.exists():
        return result
    for study_dir in sorted(base_dir.iterdir()):
        if not study_dir.is_dir():
            continue
        metrics_path = study_dir / f"opinion_knn_{study_dir.name}_validation_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as handle:
            result[study_dir.name] = json.load(handle)
    return result


def _run_sweeps(
    *,
    issues: Sequence[str],
    configs: Sequence[SweepConfig],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    sweep_dir: Path,
    word2vec_model_base: Path,
) -> List[SweepOutcome]:
    """Execute hyper-parameter sweeps and collect per-run metrics."""

    outcomes: List[SweepOutcome] = []
    for config in configs:
        for issue in issues:
            issue_slug = issue.replace(" ", "_")
            run_root = _ensure_dir(sweep_dir / config.feature_space / issue_slug / config.label)
            model_dir = None
            if config.feature_space == "word2vec":
                model_dir = _ensure_dir(word2vec_model_base / "sweeps" / issue_slug / config.label)
            cli_args: List[str] = []
            cli_args.extend(base_cli)
            cli_args.extend(config.cli_args(word2vec_model_dir=model_dir))
            cli_args.extend(["--issues", issue])
            cli_args.extend(["--out-dir", str(run_root)])
            cli_args.extend(extra_cli)
            LOGGER.info(
                "[SWEEP] feature=%s issue=%s label=%s",
                config.feature_space,
                issue_slug,
                config.label,
            )
            _run_knn_cli(cli_args)
            metrics, metrics_path = _load_metrics(run_root, issue_slug)
            outcomes.append(
                SweepOutcome(
                    issue=issue,
                    feature_space=config.feature_space,
                    config=config,
                    accuracy=float(metrics.get("accuracy_overall", 0.0)),
                    best_k=int(metrics.get("best_k", 0)),
                    eligible=int(metrics.get("n_eligible", 0)),
                    metrics_path=metrics_path,
                    metrics=metrics,
                )
            )
    return outcomes


def _select_best_configs(
    *,
    outcomes: Sequence[SweepOutcome],
    issues: Sequence[str],
) -> Dict[str, SweepSelection]:
    """Select the aggregate best configuration per feature space."""

    grouped: Dict[Tuple[str, str], MutableMapping[str, SweepOutcome]] = {}
    for outcome in outcomes:
        key = (outcome.feature_space, outcome.config.label)
        per_issue = grouped.setdefault(key, {})
        per_issue[outcome.issue] = outcome

    selections: Dict[str, SweepSelection] = {}
    for (feature_space, _label), per_issue in grouped.items():
        if any(issue not in per_issue for issue in issues):
            continue
        ordered = OrderedDict(
            (issue, per_issue[issue]) for issue in issues if issue in per_issue
        )
        eligible_total = sum(item.eligible for item in ordered.values())
        if eligible_total <= 0:
            continue
        weighted_accuracy = sum(
            item.accuracy * item.eligible for item in ordered.values()
        ) / eligible_total
        selection = SweepSelection(
            config=next(iter(ordered.values())).config,
            per_issue=ordered,
            weighted_accuracy=weighted_accuracy,
        )
        incumbent = selections.get(feature_space)
        if incumbent is None or selection.weighted_accuracy > incumbent.weighted_accuracy:
            selections[feature_space] = selection
    if not selections:
        raise RuntimeError("Failed to select a best configuration for any feature space.")
    return selections


def _run_final_evaluations(
    *,
    selections: Mapping[str, SweepSelection],
    issues: Sequence[str],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    out_dir: Path,
    word2vec_model_dir: Path,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """Run final slate evaluations and return metrics grouped by feature space."""

    metrics_by_feature: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space, selection in selections.items():
        LOGGER.info(
            "[FINAL] feature=%s weighted_accuracy=%.3f",
            feature_space,
            selection.weighted_accuracy,
        )
        feature_out_dir = _ensure_dir(out_dir / feature_space)
        model_dir = word2vec_model_dir if feature_space == "word2vec" else None
        cli_args: List[str] = []
        cli_args.extend(base_cli)
        cli_args.extend(selection.config.cli_args(word2vec_model_dir=model_dir))
        cli_args.extend(["--issues", ",".join(issues)])
        cli_args.extend(["--out-dir", str(feature_out_dir)])
        cli_args.extend(["--knn-k", str(selection.primary_best_k)])
        cli_args.extend(extra_cli)
        _run_knn_cli(cli_args)
        feature_metrics: Dict[str, Mapping[str, object]] = {}
        for issue in issues:
            issue_slug = issue.replace(" ", "_")
            metrics, _path = _load_metrics(feature_out_dir, issue_slug)
            feature_metrics[issue] = metrics
        metrics_by_feature[feature_space] = feature_metrics
    return metrics_by_feature


def _run_opinion_evaluations(
    *,
    selections: Mapping[str, SweepSelection],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    out_dir: Path,
    word2vec_model_dir: Path,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """Run opinion regression for each feature space and return metrics."""

    metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space, selection in selections.items():
        LOGGER.info("[OPINION] feature=%s", feature_space)
        feature_out_dir = _ensure_dir(out_dir)
        model_dir = word2vec_model_dir if feature_space == "word2vec" else None
        cli_args: List[str] = []
        cli_args.extend(base_cli)
        cli_args.extend(selection.config.cli_args(word2vec_model_dir=model_dir))
        cli_args.extend(["--task", "opinion"])
        cli_args.extend(["--out-dir", str(feature_out_dir)])
        cli_args.extend(["--knn-k", str(selection.primary_best_k)])
        cli_args.extend(extra_cli)
        _run_knn_cli(cli_args)
        metrics[feature_space] = _load_opinion_metrics(feature_out_dir, feature_space)
    return metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _build_hyperparameter_report(
    *,
    output_path: Path,
    selections: Mapping[str, SweepSelection],
    k_sweep: str,
) -> None:
    lines: List[str] = []
    lines.append("# KNN Hyperparameter Tuning Notes")
    lines.append("")
    lines.append("This document consolidates the selected grid searches for the KNN baselines.")
    lines.append("")
    lines.append("## Next-Video Prediction")
    lines.append("")
    lines.append("The latest sweeps cover both TF-IDF and Word2Vec feature spaces with:")
    lines.append(f"- `k ∈ {{{k_sweep}}}`")
    lines.append("- Distance metrics: cosine and L2")
    lines.append("- Text-field augmentations: none, `viewer_profile,state_text`")
    lines.append("- Word2Vec variants: vector size ∈ {128, 256}, window ∈ {5, 10}, min_count ∈ {1}")
    lines.append("")
    lines.append("| Feature space | Metric | Text fields | Vec size | Window | Min count | Issue | Accuracy | Best k |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |")
    for feature_space in ("tfidf", "word2vec"):
        selection = selections.get(feature_space)
        if not selection:
            continue
        config = selection.config
        text_label = ",".join(config.text_fields) if config.text_fields else "none"
        size = str(config.word2vec_size) if config.word2vec_size is not None else "—"
        window = str(config.word2vec_window) if config.word2vec_window is not None else "—"
        min_count = (
            str(config.word2vec_min_count) if config.word2vec_min_count is not None else "—"
        )
        for issue, outcome in selection.per_issue.items():
            lines.append(
                f"| {feature_space.upper()} | {config.metric} | {text_label} | {size} | "
                f"{window} | {min_count} | {_snake_to_title(issue)} | "
                f"{_format_float(outcome.accuracy)} | {outcome.best_k} |"
            )
    lines.append("")
    lines.append("### Observations")
    lines.append("")
    for feature_space in ("tfidf", "word2vec"):
        selection = selections.get(feature_space)
        if not selection:
            continue
        config = selection.config
        text_label = "no extra fields" if not config.text_fields else f"extra fields `{','.join(config.text_fields)}`"
        if feature_space == "word2vec":
            config_bits = (
                f"{config.metric} distance, {text_label}, size={config.word2vec_size}, "
                f"window={config.word2vec_window}, min_count={config.word2vec_min_count}"
            )
        else:
            config_bits = f"{config.metric} distance with {text_label}"
        per_issue_summary = ", ".join(
            f"{_snake_to_title(issue)} accuracy {_format_float(outcome.accuracy)} (k={outcome.best_k})"
            for issue, outcome in selection.per_issue.items()
        )
        lines.append(f"- {feature_space.upper()}: {per_issue_summary} using {config_bits}.")
    lines.append("")
    lines.append("## Post-Study Opinion Regression")
    lines.append("")
    lines.append("Opinion runs reuse the slate-selected configurations per feature space.")
    lines.append("See `reports/knn/opinion/README.md` for detailed metrics and plots.")
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_next_video_report(
    *,
    output_path: Path,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    selections: Mapping[str, SweepSelection],
) -> None:
    if not metrics_by_feature:
        raise RuntimeError("No slate metrics available to build the next-video report.")

    any_feature = next(iter(metrics_by_feature.values()))
    any_issue = next(iter(any_feature.values()))
    dataset_name = any_issue.get("dataset", DEFAULT_DATASET_SOURCE)
    split = any_issue.get("split", "validation")

    lines: List[str] = []
    lines.append("# KNN Next-Video Baseline")
    lines.append("")
    lines.append(
        "This report summarises the slate-ranking KNN model that predicts the next video a viewer will click."
    )
    lines.append("")
    lines.append(f"- Dataset: `{dataset_name}`")
    lines.append(f"- Split: {split}")
    lines.append("- Metric: accuracy on eligible slates (gold index present)")
    lines.append("")

    for feature_space in ("tfidf", "word2vec"):
        selection = selections.get(feature_space)
        metrics = metrics_by_feature.get(feature_space, {})
        if not metrics:
            continue
        config = selection.config if selection else None
        if feature_space == "tfidf":
            lines.append("## TF-IDF Feature Space")
        else:
            lines.append("## Word2Vec Feature Space")
        lines.append("")
        lines.append("| Issue | Accuracy ↑ | Best k | Most-frequent baseline ↑ |")
        lines.append("| --- | ---: | ---: | ---: |")
        for issue, data in metrics.items():
            baseline = data.get("baseline_most_frequent_gold_index", {})
            baseline_acc = float(baseline.get("accuracy", 0.0))
            lines.append(
                f"| {_snake_to_title(issue)} | "
                f"{_format_float(float(data.get('accuracy_overall', 0.0)))} | "
                f"{int(data.get('best_k', 0))} | "
                f"{_format_float(baseline_acc)} |"
            )
        lines.append("")
        if config:
            if feature_space == "word2vec":
                config_desc = (
                    f"{config.metric} distance, "
                    f"size={config.word2vec_size}, window={config.word2vec_window}, "
                    f"min_count={config.word2vec_min_count}"
                )
            else:
                config_desc = f"{config.metric} distance"
            text_label = (
                "no extra fields"
                if not config.text_fields
                else f"extra fields `{','.join(config.text_fields)}`"
            )
            lines.append(
                f"Selected configuration: {config_desc} with {text_label} "
                f"(weighted accuracy {selection.weighted_accuracy:.3f})."
            )
            lines.append("")

    lines.append("## Observations")
    lines.append("")
    for feature_space in ("tfidf", "word2vec"):
        metrics = metrics_by_feature.get(feature_space, {})
        if not metrics:
            continue
        bullet_bits = []
        for issue, data in metrics.items():
            baseline = data.get("baseline_most_frequent_gold_index", {})
            baseline_acc = float(baseline.get("accuracy", 0.0))
            bullet_bits.append(
                f"{_snake_to_title(issue)} accuracy {_format_float(float(data.get('accuracy_overall', 0.0)))} "
                f"(baseline {_format_float(baseline_acc)})"
            )
        lines.append(f"- {feature_space.upper()}: " + "; ".join(bullet_bits) + ".")
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_opinion_report(
    *,
    output_path: Path,
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    selections: Mapping[str, SweepSelection],
) -> None:
    if not metrics:
        raise RuntimeError("No opinion metrics available to build the opinion report.")

    lines: List[str] = []
    lines.append("# KNN Opinion Shift Study")
    lines.append("")
    lines.append(
        "This study evaluates a second KNN baseline that predicts each participant's post-study opinion index."
    )
    lines.append(
        "Models reuse the slate-selected feature configurations and compare against a no-change baseline."
    )
    lines.append("")
    lines.append("- Dataset: `data/cleaned_grail`")
    lines.append("- Splits: train for neighbour lookup, validation for evaluation")
    lines.append(
        "- Metrics: MAE / RMSE / R² on the predicted post index, plus a no-change (pre-index) baseline"
    )
    lines.append("")

    for feature_space in ("tfidf", "word2vec"):
        studies = metrics.get(feature_space, {})
        if not studies:
            continue
        selection = selections.get(feature_space)
        if feature_space == "tfidf":
            lines.append("## TF-IDF Feature Space")
        else:
            lines.append("## Word2Vec Feature Space")
        lines.append("")
        lines.append(
            "| Study | Participants | Best k | MAE ↓ | RMSE ↓ | R² ↑ | No-change MAE ↓ |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for study_key in sorted(studies.keys()):
            data = studies[study_key]
            best_metrics = data.get("best_metrics", {})
            baseline = data.get("baseline", {})
            label = data.get("label", _snake_to_title(study_key))
            lines.append(
                f"| {label} | {int(data.get('n_participants', 0))} | "
                f"{int(data.get('best_k', 0))} | "
                f"{_format_float(float(best_metrics.get('mae_after', 0.0)))} | "
                f"{_format_float(float(best_metrics.get('rmse_after', 0.0)))} | "
                f"{_format_float(float(best_metrics.get('r2_after', 0.0)))} | "
                f"{_format_float(float(baseline.get('mae_using_before', 0.0)))} |"
            )
        lines.append("")
        if selection:
            if feature_space == "word2vec":
                config_bits = (
                    f"{selection.config.metric} distance, "
                    f"size={selection.config.word2vec_size}, "
                    f"window={selection.config.word2vec_window}, "
                    f"min_count={selection.config.word2vec_min_count}"
                )
            else:
                config_bits = f"{selection.config.metric} distance"
            text_label = (
                "no extra fields"
                if not selection.config.text_fields
                else f"extra fields `{','.join(selection.config.text_fields)}`"
            )
            lines.append(f"Configuration: {config_bits} with {text_label}.")
            lines.append("")

    lines.append("### Opinion Change Heatmaps")
    lines.append("")
    lines.append(
        "Plots are refreshed under `reports/knn/<feature-space>/opinion/` for MAE, R², and change heatmaps."
    )
    lines.append("")

    lines.append("## Takeaways")
    lines.append("")
    if "tfidf" in metrics and "word2vec" in metrics:
        for study_key in sorted(metrics["tfidf"].keys()):
            tfidf_metrics = metrics["tfidf"][study_key]
            word2vec_metrics = metrics["word2vec"].get(study_key)
            tfidf_r2 = float(tfidf_metrics.get("best_metrics", {}).get("r2_after", 0.0))
            best_space = "TF-IDF"
            best_r2 = tfidf_r2
            best_k = int(tfidf_metrics.get("best_k", 0))
            if word2vec_metrics:
                word2vec_r2 = float(word2vec_metrics.get("best_metrics", {}).get("r2_after", 0.0))
                if word2vec_r2 > best_r2:
                    best_space = "Word2Vec"
                    best_r2 = word2vec_r2
                    best_k = int(word2vec_metrics.get("best_k", 0))
            lines.append(
                f"- {tfidf_metrics.get('label', _snake_to_title(study_key))}: "
                f"{best_space} achieves the highest R² ({best_r2:.3f}) at k={best_k}."
            )
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_reports(
    *,
    repo_root: Path,
    selections: Mapping[str, SweepSelection],
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    opinion_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    k_sweep: str,
) -> None:
    """Write refreshed Markdown reports under ``reports/knn``."""

    reports_root = repo_root / "reports" / "knn"
    _build_hyperparameter_report(
        output_path=reports_root / "hyperparameter_tuning.md",
        selections=selections,
        k_sweep=k_sweep,
    )
    _build_next_video_report(
        output_path=reports_root / "next_video.md",
        metrics_by_feature=metrics_by_feature,
        selections=selections,
    )
    _build_opinion_report(
        output_path=reports_root / "opinion" / "README.md",
        metrics=opinion_metrics,
        selections=selections,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    args, extra_cli = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    root = _repo_root()
    dataset = args.dataset or os.environ.get("DATASET") or _default_dataset(root)
    out_dir = Path(args.out_dir or os.environ.get("OUT_DIR") or _default_out_dir(root))
    cache_dir = args.cache_dir or os.environ.get("CACHE_DIR") or _default_cache_dir(root)
    sweep_dir = Path(args.sweep_dir or os.environ.get("KNN_SWEEP_DIR") or (out_dir / "sweeps"))
    word2vec_model_dir = Path(
        args.word2vec_model_dir or os.environ.get("WORD2VEC_MODEL_DIR") or (out_dir / "word2vec_models")
    )
    k_sweep = args.k_sweep or os.environ.get("KNN_K_SWEEP") or "1,2,3,4,5,10,15,20,25,50,100"
    issues_arg = args.issues or os.environ.get("KNN_ISSUES", "")
    word2vec_epochs = int(os.environ.get("WORD2VEC_EPOCHS", "10"))
    word2vec_workers = _default_word2vec_workers()

    if issues_arg:
        issues = [token.strip() for token in issues_arg.split(",") if token.strip()]
    else:
        dataset_obj = load_dataset_source(dataset, cache_dir)
        issues = issues_in_dataset(dataset_obj)
    if not issues:
        raise RuntimeError("No issues available for evaluation.")
    issues = sorted(issues)

    LOGGER.info("Dataset: %s", dataset)
    LOGGER.info("Issues: %s", ", ".join(issues))
    LOGGER.info("Output directory: %s", out_dir)

    base_cli = ["--dataset", dataset, "--cache-dir", cache_dir, "--fit-index", "--overwrite"]
    if k_sweep:
        base_cli.extend(["--knn-k-sweep", k_sweep])

    configs = _build_sweep_configs(
        word2vec_epochs=word2vec_epochs,
        word2vec_workers=word2vec_workers,
    )

    if args.dry_run:
        LOGGER.info("[DRY RUN] Planned %d sweep configurations.", len(configs))
        return

    sweep_outcomes = _run_sweeps(
        issues=issues,
        configs=configs,
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=sweep_dir,
        word2vec_model_base=word2vec_model_dir,
    )

    selections = _select_best_configs(outcomes=sweep_outcomes, issues=issues)

    slate_metrics = _run_final_evaluations(
        selections=selections,
        issues=issues,
        base_cli=base_cli,
        extra_cli=extra_cli,
        out_dir=out_dir,
        word2vec_model_dir=word2vec_model_dir,
    )

    opinion_metrics = _run_opinion_evaluations(
        selections=selections,
        base_cli=base_cli,
        extra_cli=extra_cli,
        out_dir=out_dir,
        word2vec_model_dir=word2vec_model_dir,
    )

    _generate_reports(
        repo_root=root,
        selections=selections,
        metrics_by_feature=slate_metrics,
        opinion_metrics=opinion_metrics,
        k_sweep=k_sweep,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
