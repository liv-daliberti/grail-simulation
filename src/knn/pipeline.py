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

# pylint: disable=line-too-long,too-many-lines

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from knn.cli import build_parser
from knn.data import DEFAULT_DATASET_SOURCE
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

    study: "StudySpec"
    feature_space: str
    config: SweepConfig
    accuracy: float
    best_k: int
    eligible: int
    metrics_path: Path
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class StudySpec:
    """Describe a participant study and its associated issue."""

    key: str
    issue: str
    label: str

    @property
    def study_slug(self) -> str:
        """Return a filesystem-safe slug for the study key."""

        return self.key.replace(" ", "_")

    @property
    def issue_slug(self) -> str:
        """Return a filesystem-safe slug for the associated issue."""

        return self.issue.replace(" ", "_")


@dataclass
class StudySelection:
    """Selected configuration for a specific study within a feature space."""

    study: StudySpec
    outcome: SweepOutcome

    @property
    def config(self) -> SweepConfig:
        """Return the sweep configuration selected for the study."""

        return self.outcome.config

    @property
    def accuracy(self) -> float:
        """Return the held-out accuracy achieved by the selection."""

        return self.outcome.accuracy

    @property
    def best_k(self) -> int:
        """Return the optimal ``k`` discovered during sweeps."""

        return self.outcome.best_k


@dataclass(frozen=True)
class PipelineContext:
    """Normalized configuration for a pipeline run."""

    dataset: str
    out_dir: Path
    cache_dir: str
    sweep_dir: Path
    word2vec_model_dir: Path
    k_sweep: str
    study_tokens: Tuple[str, ...]
    word2vec_epochs: int
    word2vec_workers: int


@dataclass(frozen=True)
class ReportBundle:
    """Inputs required to render the Markdown summaries."""

    selections: Mapping[str, Mapping[str, StudySelection]]
    studies: Sequence[StudySpec]
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    opinion_metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    k_sweep: str


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
        "--studies",
        default="",
        help="Comma-separated list of participant study keys (study1,study2,study3). Defaults to all studies.",
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
    """Return the default dataset path rooted at ``root``."""

    return str(root / "data" / "cleaned_grail")


def _default_cache_dir(root: Path) -> str:
    """Return the default Hugging Face cache directory under ``root``."""

    return str(root / ".cache" / "huggingface" / "knn")


def _default_out_dir(root: Path) -> str:
    """Return the default KNN output directory rooted at ``root``."""

    return str(root / "models" / "knn")


def _build_pipeline_context(args: argparse.Namespace, root: Path) -> PipelineContext:
    """Normalise CLI/environment options into a reusable context object."""

    dataset = args.dataset or os.environ.get("DATASET") or _default_dataset(root)
    out_dir_value = args.out_dir or os.environ.get("OUT_DIR") or _default_out_dir(root)
    out_dir = Path(out_dir_value)
    cache_dir_value = args.cache_dir or os.environ.get("CACHE_DIR") or _default_cache_dir(root)
    sweep_dir = Path(
        args.sweep_dir or os.environ.get("KNN_SWEEP_DIR") or (out_dir / "sweeps")
    )
    word2vec_model_dir = Path(
        args.word2vec_model_dir
        or os.environ.get("WORD2VEC_MODEL_DIR")
        or (out_dir / "word2vec_models")
    )
    k_sweep = args.k_sweep or os.environ.get("KNN_K_SWEEP") or "1,2,3,4,5,10,15,20,25,50,100"
    study_tokens = tuple(
        _split_tokens(getattr(args, "studies", ""))
        or _split_tokens(os.environ.get("KNN_STUDIES", ""))
        or _split_tokens(args.issues or "")
        or _split_tokens(os.environ.get("KNN_ISSUES", ""))
    )
    word2vec_epochs = int(os.environ.get("WORD2VEC_EPOCHS", "10"))
    word2vec_workers = _default_word2vec_workers()
    return PipelineContext(
        dataset=dataset,
        out_dir=out_dir,
        cache_dir=str(cache_dir_value),
        sweep_dir=sweep_dir,
        word2vec_model_dir=word2vec_model_dir,
        k_sweep=k_sweep,
        study_tokens=study_tokens,
        word2vec_epochs=word2vec_epochs,
        word2vec_workers=word2vec_workers,
    )


def _default_word2vec_workers() -> int:
    """Return the worker count for Word2Vec training based on environment hints."""

    env_value = os.environ.get("WORD2VEC_WORKERS")
    if env_value:
        return max(1, int(env_value))
    max_workers = int(os.environ.get("MAX_WORD2VEC_WORKERS", "40"))
    n_cpus = os.cpu_count() or 1
    return max(1, min(n_cpus, max_workers))


def _snake_to_title(value: str) -> str:
    """Convert a snake_case string into Title Case."""

    return value.replace("_", " ").title()


def _format_float(value: float) -> str:
    """Format a floating-point metric with three decimal places."""

    return f"{value:.3f}"


def _ensure_dir(path: Path) -> Path:
    """Ensure ``path`` exists and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


_STUDY_SPECS_CACHE: Tuple[StudySpec, ...] | None = None


def _study_specs() -> Tuple[StudySpec, ...]:
    """Return the cached participant-study specifications."""

    global _STUDY_SPECS_CACHE  # pylint: disable=global-statement
    if _STUDY_SPECS_CACHE is None:
        from .opinion import DEFAULT_SPECS as _DEFAULT_OPINION_SPECS  # lazy import

        _STUDY_SPECS_CACHE = tuple(
            StudySpec(spec.key, spec.issue, spec.label) for spec in _DEFAULT_OPINION_SPECS
        )
    return _STUDY_SPECS_CACHE


def _split_tokens(raw: str | None) -> List[str]:
    """Return a cleaned list of comma-separated tokens."""

    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _warn_if_issue_tokens_used(args: argparse.Namespace) -> None:
    """Log a gentle reminder that ``--issues`` is deprecated for the pipeline."""

    if not _split_tokens(getattr(args, "studies", "")) and _split_tokens(args.issues or ""):
        LOGGER.warning("`--issues` is deprecated for the pipeline; interpreting as study keys.")


def _issue_slug_for_study(study: StudySpec) -> str:
    """Return the slug used for filesystem artifacts for ``study``."""

    return f"{study.issue_slug}_{study.study_slug}"


def _resolve_studies(tokens: Sequence[str]) -> List[StudySpec]:
    """Return participant studies matching ``tokens``."""

    available = list(_study_specs())
    if not tokens:
        return available

    key_map = {spec.key.lower(): spec for spec in available}
    issue_map: Dict[str, List[StudySpec]] = {}
    for spec in available:
        issue_map.setdefault(spec.issue.lower(), []).append(spec)

    resolved: List[StudySpec] = []
    seen: set[str] = set()

    for token in tokens:
        normalised = token.strip().lower()
        if not normalised or normalised == "all":
            for spec in available:
                if spec.key not in seen:
                    resolved.append(spec)
                    seen.add(spec.key)
            continue
        if normalised in key_map:
            spec = key_map[normalised]
            if spec.key not in seen:
                resolved.append(spec)
                seen.add(spec.key)
            continue
        if normalised in issue_map:
            for spec in issue_map[normalised]:
                if spec.key not in seen:
                    resolved.append(spec)
                    seen.add(spec.key)
            continue
        valid = sorted({spec.key for spec in available} | {spec.issue for spec in available})
        raise ValueError(f"Unknown study token '{token}'. Expected one of {valid}.")
    return resolved


def _build_base_cli(context: PipelineContext) -> List[str]:
    """Return the base CLI arguments shared across pipeline steps."""

    base_cli = ["--dataset", context.dataset, "--cache-dir", context.cache_dir, "--fit-index", "--overwrite"]
    if context.k_sweep:
        base_cli.extend(["--knn-k-sweep", context.k_sweep])
    return base_cli


def _log_run_configuration(studies: Sequence[StudySpec], context: PipelineContext) -> None:
    """Emit a concise summary of the resolved pipeline configuration."""

    LOGGER.info("Dataset: %s", context.dataset)
    LOGGER.info(
        "Studies: %s",
        ", ".join(f"{spec.key} ({spec.issue})" for spec in studies),
    )
    LOGGER.info("Output directory: %s", context.out_dir)


def _log_dry_run(configs: Sequence[SweepConfig]) -> None:
    """Log the number of configurations planned during ``--dry-run``."""

    LOGGER.info("[DRY RUN] Planned %d sweep configurations.", len(configs))


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
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    sweep_dir: Path,
    word2vec_model_base: Path,
) -> List[SweepOutcome]:
    """Execute hyper-parameter sweeps and collect per-run metrics."""

    outcomes: List[SweepOutcome] = []
    for config in configs:
        for study in studies:
            issue_slug = _issue_slug_for_study(study)
            run_root = _ensure_dir(
                sweep_dir / config.feature_space / study.study_slug / config.label
            )
            model_dir = None
            if config.feature_space == "word2vec":
                model_dir = _ensure_dir(
                    word2vec_model_base / "sweeps" / study.study_slug / config.label
                )
            cli_args: List[str] = []
            cli_args.extend(base_cli)
            cli_args.extend(config.cli_args(word2vec_model_dir=model_dir))
            cli_args.extend(["--issues", study.issue])
            cli_args.extend(["--participant-studies", study.key])
            cli_args.extend(["--out-dir", str(run_root)])
            cli_args.extend(extra_cli)
            LOGGER.info(
                "[SWEEP] feature=%s study=%s issue=%s label=%s",
                config.feature_space,
                study.key,
                study.issue,
                config.label,
            )
            _run_knn_cli(cli_args)
            metrics, metrics_path = _load_metrics(run_root, issue_slug)
            outcomes.append(
                SweepOutcome(
                    study=study,
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
    studies: Sequence[StudySpec],
) -> Dict[str, Dict[str, StudySelection]]:
    """Select the best configuration per feature space and study."""

    def _is_better(candidate: SweepOutcome, incumbent: SweepOutcome) -> bool:
        """Return ``True`` when ``candidate`` should replace ``incumbent``."""

        if candidate.accuracy > incumbent.accuracy + 1e-9:
            return True
        if candidate.accuracy + 1e-9 < incumbent.accuracy:
            return False
        if candidate.eligible > incumbent.eligible:
            return True
        if candidate.eligible < incumbent.eligible:
            return False
        return candidate.best_k < incumbent.best_k

    selections: Dict[str, Dict[str, StudySelection]] = {}
    for outcome in outcomes:
        per_feature = selections.setdefault(outcome.feature_space, {})
        study_key = outcome.study.key
        current = per_feature.get(study_key)
        if current is None or _is_better(outcome, current.outcome):
            per_feature[study_key] = StudySelection(study=outcome.study, outcome=outcome)

    expected_keys = [study.key for study in studies]
    for feature_space, per_feature in selections.items():
        missing = [key for key in expected_keys if key not in per_feature]
        if missing:
            raise RuntimeError(
                f"Missing sweep selections for feature={feature_space}: {', '.join(missing)}"
            )
    if not selections:
        raise RuntimeError("Failed to select a best configuration for any feature space.")
    return selections


def _run_final_evaluations(
    *,
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    out_dir: Path,
    word2vec_model_dir: Path,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """Run final slate evaluations and return metrics grouped by feature space."""

    metrics_by_feature: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space, per_study in selections.items():
        feature_metrics: Dict[str, Mapping[str, object]] = {}
        for study in studies:
            selection = per_study.get(study.key)
            if selection is None:
                continue
            LOGGER.info(
                "[FINAL] feature=%s study=%s issue=%s accuracy=%.3f",
                feature_space,
                study.key,
                study.issue,
                selection.accuracy,
            )
            feature_out_dir = _ensure_dir(out_dir / feature_space / study.study_slug)
            model_dir = None
            if feature_space == "word2vec":
                model_dir = _ensure_dir(word2vec_model_dir / study.study_slug)
            cli_args: List[str] = []
            cli_args.extend(base_cli)
            cli_args.extend(selection.config.cli_args(word2vec_model_dir=model_dir))
            cli_args.extend(["--issues", study.issue])
            cli_args.extend(["--participant-studies", study.key])
            cli_args.extend(["--out-dir", str(feature_out_dir)])
            cli_args.extend(["--knn-k", str(selection.best_k)])
            cli_args.extend(extra_cli)
            _run_knn_cli(cli_args)
            issue_slug = _issue_slug_for_study(study)
            metrics, _path = _load_metrics(feature_out_dir, issue_slug)
            feature_metrics[study.key] = metrics
        if feature_metrics:
            metrics_by_feature[feature_space] = feature_metrics
    return metrics_by_feature


def _run_opinion_evaluations(
    *,
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    out_dir: Path,
    word2vec_model_dir: Path,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """Run opinion regression for each feature space and return metrics."""

    metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space, per_study in selections.items():
        LOGGER.info("[OPINION] feature=%s", feature_space)
        feature_out_dir = _ensure_dir(out_dir)
        for study in studies:
            selection = per_study.get(study.key)
            if selection is None:
                continue
            LOGGER.info("[OPINION] study=%s issue=%s", study.key, study.issue)
            model_dir = None
            if feature_space == "word2vec":
                model_dir = _ensure_dir(word2vec_model_dir / study.study_slug)
            cli_args: List[str] = []
            cli_args.extend(base_cli)
            cli_args.extend(selection.config.cli_args(word2vec_model_dir=model_dir))
            cli_args.extend(["--task", "opinion"])
            cli_args.extend(["--out-dir", str(feature_out_dir)])
            cli_args.extend(["--knn-k", str(selection.best_k)])
            cli_args.extend(["--opinion-studies", study.key])
            cli_args.extend(extra_cli)
            _run_knn_cli(cli_args)
        metrics[feature_space] = _load_opinion_metrics(feature_out_dir, feature_space)
    return metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _hyperparameter_report_intro(k_sweep: str) -> List[str]:
    """Return the Markdown header introducing the hyperparameter report."""

    return [
        "# KNN Hyperparameter Tuning Notes",
        "",
        "This document consolidates the selected grid searches for the KNN baselines.",
        "",
        "## Next-Video Prediction",
        "",
        "The latest sweeps cover both TF-IDF and Word2Vec feature spaces with:",
        f"- `k ∈ {{{k_sweep}}}`",
        "- Distance metrics: cosine and L2",
        "- Text-field augmentations: none, `viewer_profile,state_text`",
        "- Word2Vec variants: vector size ∈ {128, 256}, window ∈ {5, 10}, min_count ∈ {1}",
        "",
        "| Feature space | Study | Metric | Text fields | Vec size | Window | Min count | Accuracy | Best k |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]


def _hyperparameter_table_section(
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Render the hyperparameter summary table for each feature space."""

    lines: List[str] = []
    for feature_space in ("tfidf", "word2vec"):
        per_study = selections.get(feature_space, {})
        lines.extend(_hyperparameter_feature_rows(feature_space, per_study, studies))
    lines.append("")
    return lines


def _hyperparameter_feature_rows(
    feature_space: str,
    per_study: Mapping[str, StudySelection],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Return table rows covering ``feature_space`` selections."""

    rows: List[str] = []
    for study in studies:
        selection = per_study.get(study.key)
        if not selection:
            continue
        rows.append(_format_hyperparameter_row(feature_space, study, selection))
    return rows


def _format_hyperparameter_row(
    feature_space: str,
    study: StudySpec,
    selection: StudySelection,
) -> str:
    """Format a Markdown table row summarising a sweep selection."""

    config = selection.config
    text_label = ",".join(config.text_fields) if config.text_fields else "none"
    size = str(config.word2vec_size) if config.word2vec_size is not None else "—"
    window = str(config.word2vec_window) if config.word2vec_window is not None else "—"
    min_count = str(config.word2vec_min_count) if config.word2vec_min_count is not None else "—"
    return (
        f"| {feature_space.upper()} | {study.label} | {config.metric} | {text_label} | "
        f"{size} | {window} | {min_count} | {_format_float(selection.accuracy)} | "
        f"{selection.best_k} |"
    )


def _describe_text_fields(fields: Sequence[str]) -> str:
    """Return human-friendly text describing optional auxiliary fields."""

    return "no extra fields" if not fields else f"extra fields `{','.join(fields)}`"


def _format_word2vec_descriptor(config: SweepConfig, text_info: str) -> str:
    """Return a descriptive string for a Word2Vec configuration."""

    return (
        f"{config.metric} distance, {text_info}, "
        f"size={config.word2vec_size}, "
        f"window={config.word2vec_window}, "
        f"min_count={config.word2vec_min_count}"
    )


def _hyperparameter_observations_section(
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Build a bullet list summarising key sweep observations."""

    lines: List[str] = ["### Observations", ""]
    for feature_space in ("tfidf", "word2vec"):
        per_study = selections.get(feature_space, {})
        summary = _hyperparameter_feature_observation(feature_space, per_study, studies)
        if summary:
            lines.append(f"- {feature_space.upper()}: {summary}.")
    lines.append("")
    return lines


def _hyperparameter_feature_observation(
    feature_space: str,
    per_study: Mapping[str, StudySelection],
    studies: Sequence[StudySpec],
) -> str | None:
    """Summarise configurations for ``feature_space`` across studies."""

    bullet_bits: List[str] = []
    for study in studies:
        selection = per_study.get(study.key)
        if not selection:
            continue
        config = selection.config
        text_info = _describe_text_fields(config.text_fields)
        if feature_space == "word2vec":
            config_bits = _format_word2vec_descriptor(config, text_info)
        else:
            config_bits = f"{config.metric} distance with {text_info}"
        detail = (
            f"{study.label}: accuracy {_format_float(selection.accuracy)} "
            f"(k={selection.best_k}) using {config_bits}"
        )
        bullet_bits.append(detail)
    return "; ".join(bullet_bits) if bullet_bits else None


def _hyperparameter_opinion_section() -> List[str]:
    """Return the blurb linking to opinion-regression sweeps."""

    return [
        "",
        "## Post-Study Opinion Regression",
        "",
        "Opinion runs reuse the per-study slate configurations gathered above.",
        "See `reports/knn/opinion/README.md` for detailed metrics and plots.",
        "",
    ]


def _next_video_dataset_info(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Tuple[str, str]:
    for per_feature in metrics_by_feature.values():
        for study_metrics in per_feature.values():
            dataset = study_metrics.get("dataset", DEFAULT_DATASET_SOURCE)
            split = study_metrics.get("split", "validation")
            return str(dataset), str(split)
    raise RuntimeError("No slate metrics available to build the next-video report.")


def _next_video_intro(dataset_name: str, split: str) -> List[str]:
    """Return the introductory Markdown section for the next-video report."""

    return [
        "# KNN Next-Video Baseline",
        "",
        "This report summarises the slate-ranking KNN model that predicts the next video a viewer will click.",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Split: {split}",
        "- Metric: accuracy on eligible slates (gold index present)",
        "",
    ]


def _feature_space_heading(feature_space: str) -> str:
    """Return the Markdown heading for ``feature_space``."""

    return "## TF-IDF Feature Space" if feature_space == "tfidf" else "## Word2Vec Feature Space"


def _baseline_accuracy(data: Mapping[str, object]) -> float:
    """Return the baseline most-frequent gold-index accuracy from ``data``."""

    baseline = data.get("baseline_most_frequent_gold_index", {})
    getter = getattr(baseline, "get", None)
    if callable(getter):
        return float(baseline.get("accuracy", 0.0))
    return 0.0


def _next_video_feature_section(
    feature_space: str,
    metrics: Mapping[str, Mapping[str, object]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Render the next-video metrics table for ``feature_space``."""

    if not metrics:
        return []
    lines: List[str] = [
        _feature_space_heading(feature_space),
        "",
        "| Study | Accuracy ↑ | Best k | Most-frequent baseline ↑ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for study in studies:
        data = metrics.get(study.key)
        if not data:
            continue
        accuracy = _format_float(float(data.get("accuracy_overall", 0.0)))
        best_k = int(data.get("best_k", 0))
        baseline_acc = _format_float(_baseline_accuracy(data))
        lines.append(f"| {study.label} | {accuracy} | {best_k} | {baseline_acc} |")
    lines.append("")
    return lines


def _next_video_observations(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Generate bullet-point observations comparing feature spaces."""

    lines: List[str] = ["## Observations", ""]
    for feature_space in ("tfidf", "word2vec"):
        metrics = metrics_by_feature.get(feature_space, {})
        if not metrics:
            continue
        bullet_bits: List[str] = []
        for study in studies:
            data = metrics.get(study.key)
            if not data:
                continue
            accuracy = _format_float(float(data.get("accuracy_overall", 0.0)))
            baseline_acc = _format_float(_baseline_accuracy(data))
            detail = f"{study.label} accuracy {accuracy} (baseline {baseline_acc})"
            bullet_bits.append(detail)
        if bullet_bits:
            lines.append(f"- {feature_space.upper()}: " + "; ".join(bullet_bits) + ".")
    lines.append("")
    return lines


def _build_hyperparameter_report(
    *,
    output_path: Path,
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
    k_sweep: str,
) -> None:
    """Write the hyperparameter tuning summary to ``output_path``."""

    lines: List[str] = []
    lines.extend(_hyperparameter_report_intro(k_sweep))
    lines.extend(_hyperparameter_table_section(selections, studies))
    lines.extend(_hyperparameter_observations_section(selections, studies))
    lines.extend(_hyperparameter_opinion_section())
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_next_video_report(
    *,
    output_path: Path,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> None:
    """Compose the next-video evaluation report at ``output_path``."""

    if not metrics_by_feature:
        raise RuntimeError("No slate metrics available to build the next-video report.")

    dataset_name, split = _next_video_dataset_info(metrics_by_feature)
    lines: List[str] = []
    lines.extend(_next_video_intro(dataset_name, split))
    for feature_space in ("tfidf", "word2vec"):
        metrics = metrics_by_feature.get(feature_space, {})
        lines.extend(_next_video_feature_section(feature_space, metrics, studies))
    lines.extend(_next_video_observations(metrics_by_feature, studies))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _opinion_report_intro() -> List[str]:
    """Return the introductory Markdown section for the opinion report."""

    return [
        "# KNN Opinion Shift Study",
        "",
        "This study evaluates a second KNN baseline that predicts each participant's post-study opinion index.",
        "- Dataset: `data/cleaned_grail`",
        "- Splits: train for neighbour lookup, validation for evaluation",
        "- Metrics: MAE / RMSE / R² on the predicted post index, plus a no-change (pre-index) baseline",
        "",
    ]


def _opinion_feature_sections(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Render opinion metrics tables grouped by feature space."""

    lines: List[str] = []
    for feature_space in ("tfidf", "word2vec"):
        per_feature = metrics.get(feature_space, {})
        if not per_feature:
            continue
        lines.extend(
            [
                _feature_space_heading(feature_space),
                "",
                "| Study | Participants | Best k | MAE ↓ | RMSE ↓ | R² ↑ | No-change MAE ↓ |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for study in studies:
            data = per_feature.get(study.key)
            if not data:
                continue
            lines.append(_format_opinion_row(study, data))
        lines.append("")
    return lines


def _format_opinion_row(study: StudySpec, data: Mapping[str, object]) -> str:
    """Return a Markdown table row for opinion metrics."""

    best_metrics = data.get("best_metrics", {})
    baseline = data.get("baseline", {})
    label = data.get("label", study.label)
    participants = int(data.get("n_participants", 0))
    best_k = int(data.get("best_k", 0))
    mae_after = _format_float(float(best_metrics.get("mae_after", 0.0)))
    rmse_after = _format_float(float(best_metrics.get("rmse_after", 0.0)))
    r2_after = _format_float(float(best_metrics.get("r2_after", 0.0)))
    mae_baseline = _format_float(float(baseline.get("mae_using_before", 0.0)))
    return (
        f"| {label} | {participants} | {best_k} | {mae_after} | "
        f"{rmse_after} | {r2_after} | {mae_baseline} |"
    )


def _opinion_heatmap_section() -> List[str]:
    return [
        "### Opinion Change Heatmaps",
        "",
        "Plots are refreshed under `reports/knn/opinion/<feature-space>/` for MAE, R², and change heatmaps.",
        "",
    ]


def _best_opinion_space(
    tfidf_metrics: Mapping[str, object],
    word2vec_metrics: Mapping[str, object] | None,
) -> Tuple[str, float, int]:
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
    return best_space, best_r2, best_k


def _opinion_takeaways(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    lines: List[str] = ["## Takeaways", ""]
    tfidf_metrics = metrics.get("tfidf")
    word2vec_metrics = metrics.get("word2vec")
    if tfidf_metrics and word2vec_metrics:
        for study in studies:
            tfidf_data = tfidf_metrics.get(study.key)
            if not tfidf_data:
                continue
            word2vec_data = word2vec_metrics.get(study.key)
            best_space, best_r2, best_k = _best_opinion_space(tfidf_data, word2vec_data)
            label = tfidf_data.get("label", study.label)
            lines.append(
                f"- {label}: {best_space} achieves the highest R² ({best_r2:.3f}) at k={best_k}."
            )
    lines.append("")
    return lines


def _build_opinion_report(
    *,
    output_path: Path,
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> None:
    if not metrics:
        raise RuntimeError("No opinion metrics available to build the opinion report.")

    lines: List[str] = []
    lines.extend(_opinion_report_intro())
    lines.extend(_opinion_feature_sections(metrics, studies))
    lines.extend(_opinion_heatmap_section())
    lines.extend(_opinion_takeaways(metrics, studies))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _generate_reports(repo_root: Path, report_bundle: ReportBundle) -> None:
    """Write refreshed Markdown reports under ``reports/knn``."""

    reports_root = repo_root / "reports" / "knn"
    _build_hyperparameter_report(
        output_path=reports_root / "hyperparameter_tuning.md",
        selections=report_bundle.selections,
        studies=report_bundle.studies,
        k_sweep=report_bundle.k_sweep,
    )
    _build_next_video_report(
        output_path=reports_root / "next_video.md",
        metrics_by_feature=report_bundle.metrics_by_feature,
        studies=report_bundle.studies,
    )
    _build_opinion_report(
        output_path=reports_root / "opinion" / "README.md",
        metrics=report_bundle.opinion_metrics,
        studies=report_bundle.studies,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Coordinate sweeps, evaluations, and report generation for the KNN pipeline."""

    args, extra_cli = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    root = _repo_root()
    context = _build_pipeline_context(args, root)
    _warn_if_issue_tokens_used(args)

    studies = _resolve_studies(context.study_tokens)
    if not studies:
        raise RuntimeError("No studies available for evaluation.")

    _log_run_configuration(studies, context)

    base_cli = _build_base_cli(context)
    configs = _build_sweep_configs(
        word2vec_epochs=context.word2vec_epochs,
        word2vec_workers=context.word2vec_workers,
    )

    if args.dry_run:
        _log_dry_run(configs)
        return

    sweep_outcomes = _run_sweeps(
        studies=studies,
        configs=configs,
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=context.sweep_dir,
        word2vec_model_base=context.word2vec_model_dir,
    )

    selections = _select_best_configs(outcomes=sweep_outcomes, studies=studies)

    slate_metrics = _run_final_evaluations(
        selections=selections,
        studies=studies,
        base_cli=base_cli,
        extra_cli=extra_cli,
        out_dir=context.out_dir,
        word2vec_model_dir=context.word2vec_model_dir,
    )

    opinion_metrics = _run_opinion_evaluations(
        selections=selections,
        studies=studies,
        base_cli=base_cli,
        extra_cli=extra_cli,
        out_dir=context.out_dir,
        word2vec_model_dir=context.word2vec_model_dir,
    )

    report_bundle = ReportBundle(
        selections=selections,
        studies=studies,
        metrics_by_feature=slate_metrics,
        opinion_metrics=opinion_metrics,
        k_sweep=context.k_sweep,
    )
    _generate_reports(root, report_bundle)


if __name__ == "__main__":  # pragma: no cover
    main()
