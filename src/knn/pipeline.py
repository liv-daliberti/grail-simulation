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
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
    sentence_transformer_model: str | None = None
    sentence_transformer_device: str | None = None
    sentence_transformer_batch_size: int | None = None
    sentence_transformer_normalize: bool | None = None

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
        if self.feature_space == "sentence_transformer" and self.sentence_transformer_model:
            model_name = Path(self.sentence_transformer_model).name or self.sentence_transformer_model
            cleaned = re.sub(r"[^a-zA-Z0-9]+", "", model_name)
            parts.append(f"model-{cleaned or 'st'}")
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
        if self.feature_space == "sentence_transformer":
            if self.sentence_transformer_model:
                args.extend(["--sentence-transformer-model", self.sentence_transformer_model])
            if self.sentence_transformer_device:
                args.extend(["--sentence-transformer-device", self.sentence_transformer_device])
            if self.sentence_transformer_batch_size is not None:
                args.extend(
                    ["--sentence-transformer-batch-size", str(self.sentence_transformer_batch_size)]
                )
            if self.sentence_transformer_normalize is not None:
                args.append(
                    "--sentence-transformer-normalize"
                    if self.sentence_transformer_normalize
                    else "--sentence-transformer-no-normalize"
                )
        return args


@dataclass
class SweepOutcome:
    """Captures metrics for a configuration/issue pair."""

    order_index: int
    study: "StudySpec"
    feature_space: str
    config: SweepConfig
    accuracy: float
    best_k: int
    eligible: int
    metrics_path: Path
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class SweepTask:
    """Container describing a single sweep execution request."""

    index: int
    study: "StudySpec"
    config: SweepConfig
    base_cli: Tuple[str, ...]
    extra_cli: Tuple[str, ...]
    run_root: Path
    word2vec_model_dir: Path | None
    issue: str
    issue_slug: str
    metrics_path: Path


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
    sentence_model: str
    sentence_device: str | None
    sentence_batch_size: int
    sentence_normalize: bool
    feature_spaces: Tuple[str, ...]
    jobs: int
    reuse_sweeps: bool = False
    reuse_final: bool = True
    allow_incomplete: bool = False


@dataclass(frozen=True)
class ReportBundle:
    """Inputs required to render the Markdown summaries."""

    selections: Mapping[str, Mapping[str, StudySelection]]
    sweep_outcomes: Sequence[SweepOutcome]
    studies: Sequence[StudySpec]
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    opinion_metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    k_sweep: str
    loso_metrics: Optional[Mapping[str, Mapping[str, Mapping[str, object]]]] = None
    feature_spaces: Tuple[str, ...] = ("tfidf", "word2vec", "sentence_transformer")
    sentence_model: Optional[str] = None
    allow_incomplete: bool = False


@dataclass(frozen=True)
class MetricSummary:
    """Normalized slice of common slate metrics."""

    accuracy: Optional[float] = None
    accuracy_ci: Optional[Tuple[float, float]] = None
    baseline: Optional[float] = None
    baseline_ci: Optional[Tuple[float, float]] = None
    random_baseline: Optional[float] = None
    best_k: Optional[int] = None
    n_total: Optional[int] = None
    n_eligible: Optional[int] = None


@dataclass(frozen=True)
class OpinionSummary:
    """Normalized view of opinion-regression metrics."""

    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    mae_change: Optional[float] = None
    baseline_mae: Optional[float] = None
    mae_delta: Optional[float] = None
    best_k: Optional[int] = None
    participants: Optional[int] = None
    dataset: Optional[str] = None
    split: Optional[str] = None


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
        "--sentence-transformer-model",
        default=None,
        help="SentenceTransformer model name used during sweeps (default: sentence-transformers/all-mpnet-base-v2).",
    )
    parser.add_argument(
        "--sentence-transformer-device",
        default=None,
        help="Optional device override for sentence-transformer encoding (e.g. cuda, cpu).",
    )
    parser.add_argument(
        "--sentence-transformer-batch-size",
        type=int,
        default=None,
        help="Batch size for sentence-transformer encoding during sweeps (default: 32).",
    )
    parser.add_argument(
        "--sentence-transformer-normalize",
        dest="sentence_transformer_normalize",
        action="store_true",
        help="Enable L2-normalisation for sentence-transformer embeddings (default).",
    )
    parser.add_argument(
        "--sentence-transformer-no-normalize",
        dest="sentence_transformer_normalize",
        action="store_false",
        help="Disable L2-normalisation for sentence-transformer embeddings.",
    )
    parser.set_defaults(sentence_transformer_normalize=True)
    parser.add_argument(
        "--feature-spaces",
        default="",
        help="Comma-separated feature spaces to evaluate (default: tfidf,word2vec,sentence_transformer).",
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
        "--reuse-sweeps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing sweep metrics when present instead of rerunning the full grid (use --no-reuse-sweeps to force a full rerun).",
    )
    parser.add_argument(
        "--reuse-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached finalize-stage artefacts when present instead of rerunning evaluations (use --no-reuse-final to force recomputation).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow finalize/report stages to proceed with partial sweep data (use --no-allow-incomplete to require complete sweeps).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum number of concurrent sweep workers (default: 1).",
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
    k_sweep = (
        args.k_sweep
        or os.environ.get("KNN_K_SWEEP")
        or "1,2,3,4,5,10,15,20,25,50,75,100,125,150"
    )
    study_tokens = tuple(
        _split_tokens(getattr(args, "studies", ""))
        or _split_tokens(os.environ.get("KNN_STUDIES", ""))
        or _split_tokens(args.issues or "")
        or _split_tokens(os.environ.get("KNN_ISSUES", ""))
    )
    word2vec_epochs = int(os.environ.get("WORD2VEC_EPOCHS", "10"))
    word2vec_workers = _default_word2vec_workers()
    sentence_model = (
        args.sentence_transformer_model
        or os.environ.get("SENTENCE_TRANSFORMER_MODEL")
        or "sentence-transformers/all-mpnet-base-v2"
    )
    sentence_device_raw = (
        args.sentence_transformer_device
        or os.environ.get("SENTENCE_TRANSFORMER_DEVICE")
        or ""
    )
    sentence_device = sentence_device_raw or None
    sentence_batch_size = int(
        args.sentence_transformer_batch_size
        or os.environ.get("SENTENCE_TRANSFORMER_BATCH_SIZE", "32")
    )
    if "SENTENCE_TRANSFORMER_NORMALIZE" in os.environ and not getattr(args, "sentence_transformer_normalize", None) == False:
        sentence_normalize_env = os.environ.get("SENTENCE_TRANSFORMER_NORMALIZE", "1")
        sentence_normalize = sentence_normalize_env not in {"0", "false", "False"}
    else:
        sentence_normalize = bool(getattr(args, "sentence_transformer_normalize", True))
    feature_spaces_tokens = (
        _split_tokens(getattr(args, "feature_spaces", ""))
        or _split_tokens(os.environ.get("KNN_FEATURE_SPACES", ""))
    )
    allowed_spaces = {"tfidf", "word2vec", "sentence_transformer"}
    resolved_feature_spaces = tuple(
        space
        for space in (token.lower() for token in feature_spaces_tokens) if space in allowed_spaces
    )
    if not resolved_feature_spaces:
        resolved_feature_spaces = ("tfidf", "word2vec", "sentence_transformer")
    reuse_sweeps = getattr(args, "reuse_sweeps", True)
    reuse_env = os.environ.get("KNN_REUSE_SWEEPS")
    if reuse_env is not None:
        reuse_sweeps = reuse_env.lower() not in {"0", "false", "no"}
    reuse_final = getattr(args, "reuse_final", True)
    reuse_final_env = os.environ.get("KNN_REUSE_FINAL")
    if reuse_final_env is not None:
        reuse_final = reuse_final_env.lower() not in {"0", "false", "no"}
    jobs_value = getattr(args, "jobs", 1) or 1
    env_jobs = os.environ.get("KNN_JOBS")
    if env_jobs:
        try:
            jobs_value = int(env_jobs)
        except ValueError:
            LOGGER.warning("Ignoring invalid KNN_JOBS value '%s'.", env_jobs)
    jobs = max(1, jobs_value)
    allow_incomplete = getattr(args, "allow_incomplete", True)
    allow_env = os.environ.get("KNN_ALLOW_INCOMPLETE")
    if allow_env is not None:
        allow_incomplete = allow_env.lower() not in {"0", "false", "no"}
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
        sentence_model=sentence_model,
        sentence_device=sentence_device,
        sentence_batch_size=sentence_batch_size,
        sentence_normalize=sentence_normalize,
        feature_spaces=resolved_feature_spaces,
        jobs=jobs,
        reuse_sweeps=reuse_sweeps,
        reuse_final=reuse_final,
        allow_incomplete=allow_incomplete,
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


def _format_optional_float(value: Optional[float]) -> str:
    """Format optional floating-point metrics."""

    return _format_float(value) if value is not None else "—"


def _format_delta(delta: Optional[float]) -> str:
    """Format a signed improvement metric."""

    return f"{delta:+.3f}" if delta is not None else "—"


def _format_count(value: Optional[int]) -> str:
    """Format integer counts with thousands separators."""

    if value is None:
        return "—"
    return f"{value:,}"


def _format_k(value: Optional[int]) -> str:
    """Format the selected k hyperparameter."""

    if value is None or value <= 0:
        return "—"
    return str(value)


def _format_uncertainty_details(uncertainty: Mapping[str, object]) -> str:
    """Format auxiliary uncertainty metadata for reporting."""

    if not isinstance(uncertainty, Mapping):
        return ""
    detail_bits: List[str] = []
    for key in ("n_bootstrap", "n_groups", "n_rows", "seed"):
        value = uncertainty.get(key)
        if value is None:
            continue
        detail_bits.append(f"{key}={value}")
    return f" ({', '.join(detail_bits)})" if detail_bits else ""


def _safe_float(value: object) -> Optional[float]:
    """Best-effort conversion to float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> Optional[int]:
    """Best-effort conversion to int."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_ci(ci_value: object) -> Optional[Tuple[float, float]]:
    """Return a numeric confidence-interval tuple when available."""

    if isinstance(ci_value, Mapping):
        low = _safe_float(ci_value.get("low"))
        high = _safe_float(ci_value.get("high"))
        if low is not None and high is not None:
            return (low, high)
        return None
    if isinstance(ci_value, (tuple, list)) and len(ci_value) == 2:
        low = _safe_float(ci_value[0])
        high = _safe_float(ci_value[1])
        if low is not None and high is not None:
            return (low, high)
    return None


def _extract_metric_summary(data: Mapping[str, object]) -> MetricSummary:
    """Collect reusable slate metrics fields."""

    accuracy = _safe_float(data.get("accuracy_overall"))
    best_k = _safe_int(data.get("best_k"))
    n_total = _safe_int(data.get("n_total"))
    n_eligible = _safe_int(data.get("n_eligible"))
    accuracy_ci = _parse_ci(data.get("accuracy_ci_95") or data.get("accuracy_uncertainty", {}).get("ci95"))

    baseline_ci = _parse_ci(data.get("baseline_ci_95") or data.get("baseline_uncertainty", {}).get("ci95"))
    baseline_data = data.get("baseline_most_frequent_gold_index", {})
    baseline = None
    if isinstance(baseline_data, Mapping):
        baseline = _safe_float(baseline_data.get("accuracy"))

    random_baseline = _safe_float(data.get("random_baseline_expected_accuracy"))

    return MetricSummary(
        accuracy=accuracy,
        accuracy_ci=accuracy_ci,
        baseline=baseline,
        baseline_ci=baseline_ci,
        random_baseline=random_baseline,
        best_k=best_k,
        n_total=n_total,
        n_eligible=n_eligible,
    )


def _extract_opinion_summary(data: Mapping[str, object]) -> OpinionSummary:
    """Collect opinion regression metrics into a normalized structure."""

    best_metrics = data.get("best_metrics", {})
    baseline = data.get("baseline", {})
    mae_after = _safe_float(best_metrics.get("mae_after"))
    baseline_mae = _safe_float(baseline.get("mae_using_before"))
    mae_delta = (
        baseline_mae - mae_after if mae_after is not None and baseline_mae is not None else None
    )
    return OpinionSummary(
        mae=mae_after,
        rmse=_safe_float(best_metrics.get("rmse_after")),
        r2=_safe_float(best_metrics.get("r2_after")),
        mae_change=_safe_float(best_metrics.get("mae_change")),
        baseline_mae=baseline_mae,
        mae_delta=mae_delta,
        best_k=_safe_int(data.get("best_k")),
        participants=_safe_int(data.get("n_participants")),
        dataset=str(data.get("dataset")) if data.get("dataset") else None,
        split=str(data.get("split")) if data.get("split") else None,
    )


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
    LOGGER.info("Parallel jobs: %d", context.jobs)


def _log_dry_run(configs: Sequence[SweepConfig]) -> None:
    """Log the number of configurations planned during ``--dry-run``."""

    LOGGER.info("[DRY RUN] Planned %d sweep configurations.", len(configs))


def _build_sweep_configs(context: PipelineContext) -> List[SweepConfig]:
    """Return the grid of configurations evaluated during sweeps."""

    text_options: Tuple[Tuple[str, ...], ...] = ((), ("viewer_profile", "state_text"))
    feature_spaces = context.feature_spaces

    configs: List[SweepConfig] = []

    if "tfidf" in feature_spaces:
        tfidf_metrics = ("cosine", "l2")
        for metric in tfidf_metrics:
            for fields in text_options:
                configs.append(
                    SweepConfig(
                        feature_space="tfidf",
                        metric=metric,
                        text_fields=fields,
                    )
                )

    if "word2vec" in feature_spaces:
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
        word2vec_epochs_options = tuple(
            int(token)
            for token in os.environ.get("WORD2VEC_SWEEP_EPOCHS", str(context.word2vec_epochs)).split(",")
            if token.strip()
        )
        word2vec_workers_options = tuple(
            int(token)
            for token in os.environ.get("WORD2VEC_SWEEP_WORKERS", str(context.word2vec_workers)).split(",")
            if token.strip()
        )
        for metric in word2vec_metrics:
            for fields in text_options:
                for size in word2vec_sizes:
                    for window in word2vec_windows:
                        for min_count in word2vec_min_counts:
                            for epochs in word2vec_epochs_options:
                                for workers in word2vec_workers_options:
                                    configs.append(
                                        SweepConfig(
                                            feature_space="word2vec",
                                            metric=metric,
                                            text_fields=fields,
                                            word2vec_size=size,
                                            word2vec_window=window,
                                            word2vec_min_count=min_count,
                                            word2vec_epochs=epochs,
                                            word2vec_workers=workers,
                                        )
                                    )

    if "sentence_transformer" in feature_spaces:
        st_metrics = ("cosine", "l2")
        for metric in st_metrics:
            for fields in text_options:
                configs.append(
                    SweepConfig(
                        feature_space="sentence_transformer",
                        metric=metric,
                        text_fields=fields,
                        sentence_transformer_model=context.sentence_model,
                        sentence_transformer_device=context.sentence_device,
                        sentence_transformer_batch_size=context.sentence_batch_size,
                        sentence_transformer_normalize=context.sentence_normalize,
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


def _load_final_metrics_from_disk(
    *,
    out_dir: Path,
    feature_spaces: Sequence[str],
    studies: Sequence[StudySpec],
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """Load slate metrics written by prior runs instead of recomputing them."""

    metrics_by_feature: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space in feature_spaces:
        feature_dir = out_dir / feature_space
        if not feature_dir.exists():
            continue
        per_study: Dict[str, Mapping[str, object]] = {}
        for study in studies:
            study_dir = feature_dir / study.study_slug
            try:
                metrics, _ = _load_metrics(study_dir, _issue_slug_for_study(study))
            except FileNotFoundError:
                continue
            per_study[study.key] = metrics
        if per_study:
            metrics_by_feature[feature_space] = per_study
    return metrics_by_feature


def _load_loso_metrics_from_disk(
    *,
    out_dir: Path,
    feature_spaces: Sequence[str],
    studies: Sequence[StudySpec],
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """Load leave-one-study-out metrics produced by previous pipeline runs."""

    cross_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space in feature_spaces:
        loso_dir = out_dir / feature_space / "loso"
        if not loso_dir.exists():
            continue
        per_study: Dict[str, Mapping[str, object]] = {}
        for study in studies:
            holdout_dir = loso_dir / study.study_slug
            try:
                metrics, _ = _load_metrics(holdout_dir, _issue_slug_for_study(study))
            except FileNotFoundError:
                continue
            per_study[study.key] = metrics
        if per_study:
            cross_metrics[feature_space] = per_study
    return cross_metrics


def _sweep_outcome_from_metrics(
    task: SweepTask,
    metrics: Mapping[str, object],
    metrics_path: Path,
) -> SweepOutcome:
    """Translate cached metrics into a :class:`SweepOutcome`."""

    return SweepOutcome(
        order_index=task.index,
        study=task.study,
        feature_space=task.config.feature_space,
        config=task.config,
        accuracy=float(metrics.get("accuracy_overall", 0.0)),
        best_k=int(metrics.get("best_k", 0)),
        eligible=int(metrics.get("n_eligible", 0)),
        metrics_path=metrics_path,
        metrics=metrics,
    )


def _prepare_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    sweep_dir: Path,
    word2vec_model_base: Path,
    reuse_existing: bool,
) -> Tuple[List[SweepTask], List[SweepOutcome]]:
    """Return tasks requiring execution and cached outcomes when available."""

    pending_tasks: List[SweepTask] = []
    cached_outcomes: List[SweepOutcome] = []
    base_cli_tuple = tuple(base_cli)
    extra_cli_tuple = tuple(extra_cli)

    task_index = 0
    for config in configs:
        for study in studies:
            issue_slug = _issue_slug_for_study(study)
            run_root = sweep_dir / config.feature_space / study.study_slug / config.label
            metrics_path = run_root / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
            word2vec_model_dir = None
            if config.feature_space == "word2vec":
                word2vec_model_dir = word2vec_model_base / "sweeps" / study.study_slug / config.label
            task = SweepTask(
                index=task_index,
                study=study,
                config=config,
                base_cli=base_cli_tuple,
                extra_cli=extra_cli_tuple,
                run_root=run_root,
                word2vec_model_dir=word2vec_model_dir,
                issue=study.issue,
                issue_slug=issue_slug,
                metrics_path=metrics_path,
            )
            task_index += 1
            if reuse_existing and metrics_path.exists():
                LOGGER.info(
                    "[SWEEP][SKIP] feature=%s study=%s label=%s (metrics cached)",
                    config.feature_space,
                    study.key,
                    config.label,
                )
                metrics, cached_path = _load_metrics(run_root, issue_slug)
                cached_outcomes.append(_sweep_outcome_from_metrics(task, metrics, cached_path))
                continue
            pending_tasks.append(task)
    return pending_tasks, cached_outcomes


def _merge_sweep_outcomes(
    cached: Sequence[SweepOutcome],
    executed: Sequence[SweepOutcome],
) -> List[SweepOutcome]:
    """Combine cached and freshly executed sweep outcomes preserving order."""

    by_index: Dict[int, SweepOutcome] = {}
    for outcome in cached:
        by_index[outcome.order_index] = outcome
    for outcome in executed:
        if outcome.order_index in by_index:
            LOGGER.warning(
                "Duplicate sweep outcome detected for index=%d (feature=%s study=%s). Overwriting cached result.",
                outcome.order_index,
                outcome.feature_space,
                outcome.study.key,
            )
        by_index[outcome.order_index] = outcome
    return [by_index[index] for index in sorted(by_index)]


def _execute_sweep_tasks(
    tasks: Sequence[SweepTask],
    *,
    jobs: int,
) -> List[SweepOutcome]:
    """Run the supplied sweep tasks (possibly in parallel)."""

    if not tasks:
        return []

    jobs = max(1, jobs)
    if jobs == 1:
        return [_execute_sweep_task(task) for task in tasks]

    LOGGER.info("Launching %d parallel sweep workers across %d tasks.", jobs, len(tasks))
    ordered_results: List[Optional[SweepOutcome]] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_index = {
            executor.submit(_execute_sweep_task, task): index for index, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            ordered_results[index] = future.result()

    results: List[SweepOutcome] = []
    for maybe_outcome in ordered_results:
        if maybe_outcome is None:
            raise RuntimeError("Sweep task completed without returning metrics.")
        results.append(maybe_outcome)
    return results


def _emit_sweep_plan(tasks: Sequence[SweepTask]) -> None:
    """Print a human-readable sweep plan to stdout."""

    print(f"TOTAL_TASKS={len(tasks)}")
    if not tasks:
        return
    print("INDEX\tSTUDY\tISSUE\tFEATURE_SPACE\tLABEL")
    for display_index, task in enumerate(tasks):
        print(
            f"{display_index}\t{task.study.key}\t{task.issue}\t"
            f"{task.config.feature_space}\t{task.config.label}"
        )


def _format_sweep_task_descriptor(task: SweepTask) -> str:
    """Return a short string describing a sweep task (feature/study/label)."""

    return f"{task.config.feature_space}:{task.study.key}:{task.config.label}"


def _run_sweeps(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    sweep_dir: Path,
    word2vec_model_base: Path,
    reuse_existing: bool,
    jobs: int,
) -> List[SweepOutcome]:
    """Execute hyper-parameter sweeps and collect per-run metrics."""

    pending_tasks, cached_outcomes = _prepare_sweep_tasks(
        studies=studies,
        configs=configs,
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=sweep_dir,
        word2vec_model_base=word2vec_model_base,
        reuse_existing=reuse_existing,
    )
    executed_outcomes = _execute_sweep_tasks(pending_tasks, jobs=jobs)
    return _merge_sweep_outcomes(cached_outcomes, executed_outcomes)


def _execute_sweep_task(task: SweepTask) -> SweepOutcome:
    """Execute a single sweep task and return the captured metrics."""

    run_root = _ensure_dir(task.run_root)
    model_dir = None
    if task.config.feature_space == "word2vec":
        if task.word2vec_model_dir is None:
            raise RuntimeError("Word2Vec sweep task missing model directory.")
        model_dir = _ensure_dir(task.word2vec_model_dir)

    cli_args: List[str] = list(task.base_cli)
    cli_args.extend(task.config.cli_args(word2vec_model_dir=model_dir))
    cli_args.extend(["--issues", task.issue])
    cli_args.extend(["--participant-studies", task.study.key])
    cli_args.extend(["--out-dir", str(run_root)])
    cli_args.extend(task.extra_cli)

    LOGGER.info(
        "[SWEEP] feature=%s study=%s issue=%s label=%s",
        task.config.feature_space,
        task.study.key,
        task.study.issue,
        task.config.label,
    )
    _run_knn_cli(cli_args)
    metrics, metrics_path = _load_metrics(run_root, task.issue_slug)
    return SweepOutcome(
        order_index=task.index,
        study=task.study,
        feature_space=task.config.feature_space,
        config=task.config,
        accuracy=float(metrics.get("accuracy_overall", 0.0)),
        best_k=int(metrics.get("best_k", 0)),
        eligible=int(metrics.get("n_eligible", 0)),
        metrics_path=metrics_path,
        metrics=metrics,
    )


def _select_best_configs(
    *,
    outcomes: Sequence[SweepOutcome],
    studies: Sequence[StudySpec],
    allow_incomplete: bool = False,
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
            if allow_incomplete:
                LOGGER.warning(
                    "Missing sweep selections for feature=%s: %s. Continuing because allow-incomplete mode is enabled.",
                    feature_space,
                    ", ".join(missing),
                )
            else:
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
    reuse_existing: bool,
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
            issue_slug = _issue_slug_for_study(study)
            metrics_path = feature_out_dir / issue_slug / f"knn_eval_{issue_slug}_validation_metrics.json"
            if reuse_existing and metrics_path.exists():
                try:
                    metrics, _ = _load_metrics(feature_out_dir, issue_slug)
                except FileNotFoundError:
                    LOGGER.warning(
                        "[FINAL][MISS] feature=%s study=%s expected cached metrics at %s but none found.",
                        feature_space,
                        study.key,
                        metrics_path,
                    )
                else:
                    feature_metrics[study.key] = metrics
                    LOGGER.info(
                        "[FINAL][SKIP] feature=%s study=%s (metrics cached).",
                        feature_space,
                        study.key,
                    )
                    continue
            cli_args: List[str] = []
            cli_args.extend(base_cli)
            cli_args.extend(selection.config.cli_args(word2vec_model_dir=model_dir))
            cli_args.extend(["--issues", study.issue])
            cli_args.extend(["--participant-studies", study.key])
            cli_args.extend(["--out-dir", str(feature_out_dir)])
            cli_args.extend(["--knn-k", str(selection.best_k)])
            cli_args.extend(extra_cli)
            _run_knn_cli(cli_args)
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
    reuse_existing: bool,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """Run opinion regression for each feature space and return metrics."""

    metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for feature_space, per_study in selections.items():
        LOGGER.info("[OPINION] feature=%s", feature_space)
        feature_out_dir = _ensure_dir(out_dir)
        cached_metrics = _load_opinion_metrics(feature_out_dir, feature_space) if reuse_existing else {}
        for study in studies:
            selection = per_study.get(study.key)
            if selection is None:
                continue
            if reuse_existing and study.key in cached_metrics:
                LOGGER.info(
                    "[OPINION][SKIP] feature=%s study=%s (metrics cached).",
                    feature_space,
                    study.key,
                )
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
# Cross-study evaluations
# ---------------------------------------------------------------------------


def _run_cross_study_evaluations(
    *,
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
    base_cli: Sequence[str],
    extra_cli: Sequence[str],
    out_dir: Path,
    word2vec_model_dir: Path,
    reuse_existing: bool,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """Run leave-one-study-out evaluations and return metrics grouped by feature space."""

    cross_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
    cached_cross = (
        _load_loso_metrics_from_disk(
            out_dir=out_dir,
            feature_spaces=tuple(selections.keys()),
            studies=studies,
        )
        if reuse_existing
        else {}
    )
    for feature_space, per_study in selections.items():
        feature_metrics: Dict[str, Mapping[str, object]] = dict(cached_cross.get(feature_space, {}))
        feature_out_dir = _ensure_dir(out_dir / feature_space / "loso")
        for study in studies:
            selection = per_study.get(study.key)
            if selection is None:
                continue
            if reuse_existing and study.key in feature_metrics:
                LOGGER.info(
                    "[LOSO][SKIP] feature=%s holdout=%s (metrics cached).",
                    feature_space,
                    study.key,
                )
                continue
            train_studies = [spec.key for spec in studies if spec.key != study.key]
            if not train_studies:
                LOGGER.warning(
                    "[LOSO] Skipping feature=%s holdout=%s (no alternate studies)",
                    feature_space,
                    study.key,
                )
                continue

            model_dir = None
            if feature_space == "word2vec":
                model_dir = _ensure_dir(word2vec_model_dir / "loso" / study.study_slug)

            holdout_out_dir = _ensure_dir(feature_out_dir / study.study_slug)
            cli_args: List[str] = []
            cli_args.extend(base_cli)
            cli_args.extend(selection.config.cli_args(word2vec_model_dir=model_dir))
            cli_args.extend(["--out-dir", str(holdout_out_dir)])
            cli_args.extend(["--knn-k", str(selection.best_k)])
            cli_args.extend(["--train-participant-studies", ",".join(train_studies)])
            cli_args.extend(["--eval-participant-studies", study.key])
            cli_args.extend(["--train-issues", "all"])
            cli_args.extend(["--eval-issues", study.issue])
            cli_args.extend(extra_cli)

            LOGGER.info(
                "[LOSO] feature=%s holdout=%s train_studies=%s",
                feature_space,
                study.key,
                ",".join(train_studies),
            )
            _run_knn_cli(cli_args)

            issue_slug = _issue_slug_for_study(study)
            try:
                metrics, metrics_path = _load_metrics(holdout_out_dir, issue_slug)
            except FileNotFoundError:
                LOGGER.warning(
                    "[LOSO] Missing metrics for feature=%s holdout=%s (expected slug=%s)",
                    feature_space,
                    study.key,
                    issue_slug,
                )
                continue
            feature_metrics[study.key] = metrics
            LOGGER.info(
                "[LOSO] feature=%s holdout=%s metrics=%s",
                feature_space,
                study.key,
                metrics_path,
            )
        if feature_metrics:
            cross_metrics[feature_space] = feature_metrics
    return cross_metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _hyperparameter_report_intro(
    k_sweep: str,
    feature_spaces: Sequence[str],
    sentence_model: Optional[str],
) -> List[str]:
    """Return the Markdown header introducing the hyperparameter report."""

    feature_label = ", ".join(space.replace("_", "-" ).upper() for space in feature_spaces)
    lines = [
        "# KNN Hyperparameter Tuning Notes",
        "",
        "This document consolidates the selected grid searches for the KNN baselines.",
        "",
        "## Next-Video Prediction",
        "",
        f"The latest sweeps cover the {feature_label} feature spaces with:",
        f"- `k ∈ {{{k_sweep}}}`",
        "- Distance metrics: cosine and L2",
        "- Text-field augmentations: none, `viewer_profile,state_text`",
    ]
    if "word2vec" in feature_spaces:
        lines.append("- Word2Vec variants: vector size ∈ {128, 256}, window ∈ {5, 10}, min_count ∈ {1}")
    if "sentence_transformer" in feature_spaces and sentence_model:
        lines.append(f"- Sentence-transformer model: `{sentence_model}`")
    lines.extend(
        [
            "",
            "| Feature space | Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    return lines


def _hyperparameter_table_section(
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Render the hyperparameter summary table for each feature space."""

    lines: List[str] = []
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in selections
    ]
    for space in selections:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
        per_study = selections.get(feature_space, {})
        lines.extend(_hyperparameter_feature_rows(feature_space, per_study, studies))
    lines.append("")
    return lines


def _hyperparameter_leaderboard_section(
    *,
    sweep_outcomes: Sequence[SweepOutcome],
    selections: Mapping[str, Mapping[str, StudySelection]],
    studies: Sequence[StudySpec],
    top_n: int,
) -> List[str]:
    """Return detailed leaderboards for the top-performing sweep configurations."""

    if not sweep_outcomes:
        return []

    lines: List[str] = ["### Configuration Leaderboards", ""]
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in selections
    ]
    seen_spaces: set[str] = set(ordered_spaces)
    for outcome in sweep_outcomes:
        if outcome.feature_space not in seen_spaces:
            ordered_spaces.append(outcome.feature_space)
            seen_spaces.add(outcome.feature_space)

    per_feature: Dict[str, Dict[str, List[SweepOutcome]]] = {}
    for outcome in sweep_outcomes:
        per_feature.setdefault(outcome.feature_space, {}).setdefault(outcome.study.key, []).append(outcome)

    for feature_space in ordered_spaces:
        feature_outcomes = per_feature.get(feature_space)
        if not feature_outcomes:
            continue
        lines.append(_feature_space_heading(feature_space))
        lines.append("")
        for study in studies:
            study_outcomes = feature_outcomes.get(study.key, [])
            if not study_outcomes:
                continue
            ranked = sorted(
                study_outcomes,
                key=lambda item: (item.accuracy, item.eligible, -item.best_k),
                reverse=True,
            )
            top_results = ranked[: max(1, top_n)]
            selected = selections.get(feature_space, {}).get(study.key)
            selected_label = selected.config.label if selected else None
            best_accuracy = top_results[0].accuracy if top_results else 0.0
            lines.append(f"#### {study.label}")
            lines.append("")
            lines.append("| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |")
            lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
            for idx, outcome in enumerate(top_results, start=1):
                config_label = outcome.config.label
                label_display = f"**{config_label}**" if config_label == selected_label else config_label
                delta = max(0.0, best_accuracy - outcome.accuracy)
                lines.append(
                    "| {rank} | {label} | {acc} | {delta} | {k} | {eligible} |".format(
                        rank=idx,
                        label=label_display,
                        acc=_format_float(outcome.accuracy),
                        delta=_format_float(delta),
                        k=outcome.best_k,
                        eligible=outcome.eligible,
                    )
                )
            lines.append("")
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
    if feature_space == "sentence_transformer":
        model = config.sentence_transformer_model or "sentence-transformer"
    elif feature_space == "word2vec":
        model = "word2vec"
    else:
        model = "tfidf"
    metrics_map = selection.outcome.metrics or {}
    summary = _extract_metric_summary(metrics_map) if metrics_map else MetricSummary()
    delta = (
        summary.accuracy - summary.baseline
        if summary.accuracy is not None and summary.baseline is not None
        else None
    )
    eligible = summary.n_eligible if summary.n_eligible is not None else selection.outcome.eligible
    return (
        f"| {feature_space.upper()} | {study.label} | {config.metric} | {text_label} | {model} | "
        f"{size} | {window} | {min_count} | {_format_optional_float(summary.accuracy)} | "
        f"{_format_optional_float(summary.baseline)} | {_format_delta(delta)} | "
        f"{_format_k(summary.best_k or selection.best_k)} | {_format_count(eligible)} |"
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
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in selections
    ]
    for space in selections:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
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
        metrics_map = selection.outcome.metrics or {}
        summary = _extract_metric_summary(metrics_map) if metrics_map else MetricSummary()
        accuracy_value = summary.accuracy if summary.accuracy is not None else selection.accuracy
        baseline_value = summary.baseline
        delta_value = (
            summary.accuracy - summary.baseline
            if summary.accuracy is not None and summary.baseline is not None
            else None
        )
        text_info = _describe_text_fields(config.text_fields)
        if feature_space == "word2vec":
            config_bits = _format_word2vec_descriptor(config, text_info)
        elif feature_space == "sentence_transformer":
            model_name = config.sentence_transformer_model or "sentence-transformer"
            config_bits = f"{config.metric} distance, {text_info}, model={model_name}"
        else:
            config_bits = f"{config.metric} distance with {text_info}"
        detail = (
            f"{study.label}: accuracy {_format_optional_float(accuracy_value)} "
            f"(baseline {_format_optional_float(baseline_value)}, Δ {_format_delta(delta_value)}, "
            f"k={_format_k(summary.best_k or selection.best_k)}) using {config_bits}"
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
    """Extract a representative dataset name and split from metrics payloads."""

    for per_feature in metrics_by_feature.values():
        for study_metrics in per_feature.values():
            dataset = study_metrics.get("dataset", DEFAULT_DATASET_SOURCE)
            split = study_metrics.get("split", "validation")
            return str(dataset), str(split)
    raise RuntimeError("No slate metrics available to build the next-video report.")


def _next_video_uncertainty_info(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Optional[Mapping[str, object]]:
    """Return the first uncertainty payload available for reporting."""

    for per_feature in metrics_by_feature.values():
        for study_metrics in per_feature.values():
            uncertainty = study_metrics.get("uncertainty")
            if isinstance(uncertainty, Mapping):
                return uncertainty
    return None


def _next_video_intro(
    dataset_name: str,
    split: str,
    uncertainty: Optional[Mapping[str, object]] = None,
) -> List[str]:
    """Return the introductory Markdown section for the next-video report."""

    return [
        "# KNN Next-Video Baseline",
        "",
        "This report summarises the slate-ranking KNN model that predicts the next video a viewer will click.",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Split: {split}",
        "- Metric: accuracy on eligible slates (gold index present)",
        "- Baseline column: accuracy from always recommending the most-frequent gold index for the study.",
        "- Δ column: improvement over that baseline accuracy.",
        "- Random column: expected accuracy from uniformly sampling one candidate per slate.",
        *(
            [
                "- Uncertainty: "
                + str(uncertainty.get("method", "unknown"))
                + _format_uncertainty_details(uncertainty)
            ]
            if uncertainty
            else []
        ),
        "",
    ]


def _feature_space_heading(feature_space: str) -> str:
    """Return the Markdown heading for ``feature_space``."""

    if feature_space == "tfidf":
        return "## TF-IDF Feature Space"
    if feature_space == "word2vec":
        return "## Word2Vec Feature Space"
    if feature_space == "sentence_transformer":
        return "## Sentence-Transformer Feature Space"
    return f"## {feature_space.replace('_', ' ').title()} Feature Space"


def _format_ci(ci_value: object) -> str:
    """Format a 95% confidence interval if present."""

    if isinstance(ci_value, Mapping):
        low = _safe_float(ci_value.get("low"))
        high = _safe_float(ci_value.get("high"))
        if low is not None and high is not None:
            return f"[{low:.3f}, {high:.3f}]"
        return "—"
    if isinstance(ci_value, (tuple, list)) and len(ci_value) == 2:
        low = _safe_float(ci_value[0])
        high = _safe_float(ci_value[1])
        if low is not None and high is not None:
            return f"[{low:.3f}, {high:.3f}]"
    return "—"


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
        "| Study | Accuracy ↑ | 95% CI | Δ vs baseline ↑ | Baseline ↑ | Random ↑ | Best k | Eligible | Total |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for study in studies:
        data = metrics.get(study.key)
        if not data:
            continue
        summary = _extract_metric_summary(data)
        delta = (
            summary.accuracy - summary.baseline
            if summary.accuracy is not None and summary.baseline is not None
            else None
        )
        lines.append(
            f"| {study.label} | {_format_optional_float(summary.accuracy)} | "
            f"{_format_ci(summary.accuracy_ci)} | {_format_delta(delta)} | "
            f"{_format_optional_float(summary.baseline)} | "
            f"{_format_optional_float(summary.random_baseline)} | "
            f"{_format_k(summary.best_k)} | {_format_count(summary.n_eligible)} | "
            f"{_format_count(summary.n_total)} |"
        )
    lines.append("")
    return lines


def _next_video_loso_section(
    loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Render the leave-one-study-out summary table."""

    lines: List[str] = [
        "## Leave-One-Study-Out Evaluation",
        "",
        "| Feature space | Holdout study | Accuracy ↑ | 95% CI | Δ vs baseline ↑ | Baseline ↑ |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    ordered_spaces = [space for space in ("tfidf", "word2vec", "sentence_transformer") if space in loso_metrics]
    for space in loso_metrics:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
        feature_data = loso_metrics.get(feature_space, {})
        if not feature_data:
            continue
        for study in studies:
            data = feature_data.get(study.key)
            if not data:
                continue
            summary = _extract_metric_summary(data)
            delta = (
                summary.accuracy - summary.baseline
                if summary.accuracy is not None and summary.baseline is not None
                else None
            )
            ci_text = _format_ci(summary.accuracy_ci)
            lines.append(
                f"| {feature_space.upper()} | {study.label} | {_format_optional_float(summary.accuracy)} | "
                f"{ci_text} | {_format_delta(delta)} | {_format_optional_float(summary.baseline)} |"
            )
    lines.append("")
    lines.append(
        "Holdout runs fit the index on participants from the remaining studies and evaluate on "
        "the withheld study's validation rows."
    )
    lines.append("")
    return lines


def _next_video_observations(
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Generate bullet-point observations comparing feature spaces."""

    lines: List[str] = ["## Observations", ""]
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in metrics_by_feature
    ]
    for space in metrics_by_feature:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
        metrics = metrics_by_feature.get(feature_space, {})
        if not metrics:
            continue
        bullet_bits: List[str] = []
        deltas: List[float] = []
        randoms: List[float] = []
        for study in studies:
            data = metrics.get(study.key)
            if not data:
                continue
            summary = _extract_metric_summary(data)
            if summary.accuracy is None:
                continue
            delta_val = (
                summary.accuracy - summary.baseline
                if summary.baseline is not None
                else None
            )
            if delta_val is not None:
                deltas.append(delta_val)
            if summary.random_baseline is not None:
                randoms.append(summary.random_baseline)
            detail = (
                f"{study.label}: {_format_optional_float(summary.accuracy)} "
                f"(baseline {_format_optional_float(summary.baseline)}, "
                f"Δ {_format_delta(delta_val)}, k={_format_k(summary.best_k)}, "
                f"eligible {_format_count(summary.n_eligible)})"
            )
            bullet_bits.append(detail)
        extras: List[str] = []
        if deltas:
            mean_delta = sum(deltas) / len(deltas)
            extras.append(f"mean Δ {_format_delta(mean_delta)}")
        if randoms:
            mean_random = sum(randoms) / len(randoms)
            extras.append(f"mean random {_format_optional_float(mean_random)}")
        if extras:
            bullet_bits.append("averages: " + ", ".join(extras))
        if bullet_bits:
            lines.append(f"- {feature_space.upper()}: " + "; ".join(bullet_bits) + ".")
    lines.append("- Random values correspond to the expected accuracy from a uniform guess across the slate options.")
    lines.append("")
    return lines


def _build_hyperparameter_report(
    *,
    output_dir: Path,
    selections: Mapping[str, Mapping[str, StudySelection]],
    sweep_outcomes: Sequence[SweepOutcome],
    studies: Sequence[StudySpec],
    k_sweep: str,
    feature_spaces: Sequence[str],
    sentence_model: Optional[str],
) -> None:
    """Write the hyperparameter tuning summary under ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "README.md"
    lines: List[str] = []
    lines.extend(_hyperparameter_report_intro(k_sweep, feature_spaces, sentence_model))
    lines.extend(_hyperparameter_table_section(selections, studies))
    lines.extend(
        _hyperparameter_leaderboard_section(
            sweep_outcomes=sweep_outcomes,
            selections=selections,
            studies=studies,
            top_n=3,
        )
    )
    lines.extend(_hyperparameter_observations_section(selections, studies))
    lines.extend(_hyperparameter_opinion_section())
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_next_video_report(
    *,
    output_dir: Path,
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
    feature_spaces: Sequence[str],
    loso_metrics: Optional[Mapping[str, Mapping[str, Mapping[str, object]]]] = None,
    allow_incomplete: bool = False,
) -> None:
    """Compose the next-video evaluation report under ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "README.md"
    if not metrics_by_feature:
        if not allow_incomplete:
            raise RuntimeError("No slate metrics available to build the next-video report.")
        placeholder = [
            "# KNN Next-Video Baseline",
            "",
            "Final slate metrics are not available yet. Rerun the pipeline with `--stage=finalize` once sweeps finish.",
            "",
            "This placeholder was generated with `--allow-incomplete` enabled.",
            "",
        ]
        output_path.write_text("\n".join(placeholder), encoding="utf-8")
        return

    dataset_name, split = _next_video_dataset_info(metrics_by_feature)
    uncertainty = _next_video_uncertainty_info(metrics_by_feature)
    lines: List[str] = []
    lines.extend(_next_video_intro(dataset_name, split, uncertainty))
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in feature_spaces
    ]
    for space in feature_spaces:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
        metrics = metrics_by_feature.get(feature_space, {})
        lines.extend(_next_video_feature_section(feature_space, metrics, studies))
    lines.extend(_next_video_observations(metrics_by_feature, studies))
    if loso_metrics:
        lines.extend(_next_video_loso_section(loso_metrics, studies))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _opinion_report_intro(dataset_name: str, split: str) -> List[str]:
    """Return the introductory Markdown section for the opinion report."""

    return [
        "# KNN Opinion Shift Study",
        "",
        "This study evaluates a second KNN baseline that predicts each participant's post-study opinion index.",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Split: {split}",
        "- Metrics: MAE / RMSE / R² on the predicted post index, compared against a no-change baseline.",
        "",
    ]


def _opinion_dataset_info(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Tuple[str, str]:
    """Extract dataset metadata from the opinion metrics bundle."""

    for per_feature in metrics.values():
        for study_metrics in per_feature.values():
            summary = _extract_opinion_summary(study_metrics)
            return (
                str(summary.dataset or DEFAULT_DATASET_SOURCE),
                str(summary.split or "validation"),
            )
    return (DEFAULT_DATASET_SOURCE, "validation")


def _opinion_feature_sections(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Render opinion metrics tables grouped by feature space."""

    lines: List[str] = []
    ordered_spaces = [
        space
        for space in ("tfidf", "word2vec", "sentence_transformer")
        if space in metrics
    ]
    for space in metrics:
        if space not in ordered_spaces:
            ordered_spaces.append(space)
    for feature_space in ordered_spaces:
        per_feature = metrics.get(feature_space, {})
        if not per_feature:
            continue
        lines.extend(
            [
                _feature_space_heading(feature_space),
                "",
                "| Study | Participants | Best k | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | Baseline MAE ↓ |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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

    summary = _extract_opinion_summary(data)
    label = str(data.get("label", study.label))
    participants_text = _format_count(summary.participants)
    return (
        f"| {label} | {participants_text} | {_format_k(summary.best_k)} | "
        f"{_format_optional_float(summary.mae)} | {_format_delta(summary.mae_delta)} | "
        f"{_format_optional_float(summary.rmse)} | {_format_optional_float(summary.r2)} | "
        f"{_format_optional_float(summary.mae_change)} | "
        f"{_format_optional_float(summary.baseline_mae)} |"
    )


def _opinion_heatmap_section() -> List[str]:
    """Return the Markdown section referencing opinion heatmaps."""

    return [
        "### Opinion Change Heatmaps",
        "",
        "Plots are refreshed under `reports/knn/opinion/<feature-space>/` for MAE, R², and change heatmaps.",
        "",
    ]


def _opinion_takeaways(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
) -> List[str]:
    """Generate takeaway bullets comparing opinion performance."""

    lines: List[str] = ["## Takeaways", ""]
    for study in studies:
        per_study: Dict[str, Tuple[OpinionSummary, Mapping[str, object]]] = {}
        for feature_space, per_feature in metrics.items():
            data = per_feature.get(study.key)
            if not data:
                continue
            per_study[feature_space] = (_extract_opinion_summary(data), data)
        if not per_study:
            continue

        label = next((data.get("label") for _summary, data in per_study.values() if data.get("label")), study.label)
        best_r2_value: Optional[float] = None
        best_r2_space: Optional[str] = None
        best_r2_k: Optional[int] = None
        best_delta_value: Optional[float] = None
        best_delta_space: Optional[str] = None
        for feature_space, (summary, _data) in per_study.items():
            if summary.r2 is not None:
                if best_r2_value is None or summary.r2 > best_r2_value:
                    best_r2_value = summary.r2
                    best_r2_space = feature_space
                    best_r2_k = summary.best_k
            if summary.mae_delta is not None:
                if best_delta_value is None or summary.mae_delta > best_delta_value:
                    best_delta_value = summary.mae_delta
                    best_delta_space = feature_space

        bullet_bits: List[str] = []
        if best_r2_value is not None and best_r2_space is not None:
            bullet_bits.append(
                f"best R² {_format_optional_float(best_r2_value)} with {best_r2_space.upper()} "
                f"(k={_format_k(best_r2_k)})"
            )
        if best_delta_value is not None and best_delta_space is not None:
            bullet_bits.append(
                f"largest MAE reduction {_format_delta(best_delta_value)} via {best_delta_space.upper()}"
            )
        if bullet_bits:
            lines.append(f"- {label}: " + "; ".join(bullet_bits) + ".")
    lines.append("")
    return lines


def _build_opinion_report(
    *,
    output_path: Path,
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    studies: Sequence[StudySpec],
    allow_incomplete: bool = False,
) -> None:
    """Compose the opinion regression report at ``output_path``."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        if not allow_incomplete:
            raise RuntimeError("No opinion metrics available to build the opinion report.")
        placeholder = [
            "# KNN Opinion Shift Study",
            "",
            "Opinion regression metrics are not available yet. Execute the finalize stage to refresh these results.",
            "",
            "This placeholder was generated with `--allow-incomplete` enabled.",
            "",
        ]
        output_path.write_text("\n".join(placeholder), encoding="utf-8")
        return
    dataset_name, split = _opinion_dataset_info(metrics)
    lines: List[str] = []
    lines.extend(_opinion_report_intro(dataset_name, split))
    lines.extend(_opinion_feature_sections(metrics, studies))
    lines.extend(_opinion_heatmap_section())
    lines.extend(_opinion_takeaways(metrics, studies))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _generate_reports(repo_root: Path, report_bundle: ReportBundle) -> None:
    """Write refreshed Markdown reports under ``reports/knn``."""

    reports_root = repo_root / "reports" / "knn"
    feature_spaces = report_bundle.feature_spaces
    allow_incomplete = report_bundle.allow_incomplete
    legacy_hyper_file = reports_root / "hyperparameter_tuning.md"
    legacy_next_file = reports_root / "next_video.md"
    if legacy_hyper_file.exists():
        legacy_hyper_file.unlink()
    if legacy_next_file.exists():
        legacy_next_file.unlink()
    _build_hyperparameter_report(
        output_dir=reports_root / "hyperparameter_tuning",
        selections=report_bundle.selections,
        sweep_outcomes=report_bundle.sweep_outcomes,
        studies=report_bundle.studies,
        k_sweep=report_bundle.k_sweep,
        feature_spaces=feature_spaces,
        sentence_model=report_bundle.sentence_model,
    )
    _build_next_video_report(
        output_dir=reports_root / "next_video",
        metrics_by_feature=report_bundle.metrics_by_feature,
        studies=report_bundle.studies,
        feature_spaces=feature_spaces,
        loso_metrics=report_bundle.loso_metrics,
        allow_incomplete=allow_incomplete,
    )
    _build_opinion_report(
        output_path=reports_root / "opinion" / "README.md",
        metrics=report_bundle.opinion_metrics,
        studies=report_bundle.studies,
        allow_incomplete=allow_incomplete,
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
    configs = _build_sweep_configs(context)
    stage = getattr(args, "stage", "full")

    # Enumerate the full sweep plan once so it can be reused across stages.
    planned_tasks, cached_planned = _prepare_sweep_tasks(
        studies=studies,
        configs=configs,
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=context.sweep_dir,
        word2vec_model_base=context.word2vec_model_dir,
        reuse_existing=context.reuse_sweeps,
    )

    if stage == "plan":
        _log_dry_run(configs)
        LOGGER.info(
            "Planned %d sweep configurations (%d cached).",
            len(planned_tasks),
            len(cached_planned),
        )
        _emit_sweep_plan(planned_tasks)
        return

    if args.dry_run:
        _log_dry_run(configs)
        LOGGER.info(
            "Dry-run mode. Pending sweep tasks: %d (cached: %d).",
            len(planned_tasks),
            len(cached_planned),
        )
        return

    if stage == "sweeps":
        total_tasks = len(planned_tasks)
        if total_tasks == 0:
            LOGGER.info("No sweep tasks pending; existing metrics cover the grid.")
            return
        task_id = args.sweep_task_id
        if task_id is None:
            env_value = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env_value is None:
                raise RuntimeError(
                    "Sweep stage requires --sweep-task-id or the SLURM_ARRAY_TASK_ID environment variable."
                )
            try:
                task_id = int(env_value)
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid SLURM_ARRAY_TASK_ID '{env_value}'; expected an integer."
                ) from exc
        if args.sweep_task_count is not None and args.sweep_task_count != total_tasks:
            LOGGER.warning(
                "Sweep task count mismatch: expected=%d provided=%d.",
                total_tasks,
                args.sweep_task_count,
            )
        if task_id < 0 or task_id >= total_tasks:
            raise RuntimeError(
                f"Sweep task index {task_id} outside valid range 0..{total_tasks - 1}."
            )
        task = planned_tasks[task_id]
        if context.reuse_sweeps and task.metrics_path.exists():
            LOGGER.info(
                "Skipping sweep task %d (%s | %s | %s); metrics already present at %s.",
                task.index,
                task.study.key,
                task.config.feature_space,
                task.config.label,
                task.metrics_path,
            )
            return
        outcome = _execute_sweep_task(task)
        LOGGER.info(
            "Completed sweep task %d (%s | %s | %s). Metrics stored at %s.",
            outcome.order_index,
            task.study.key,
            task.config.feature_space,
            task.config.label,
            outcome.metrics_path,
        )
        return

    reuse_for_stage = context.reuse_sweeps
    if stage in {"finalize", "reports"}:
        reuse_for_stage = True

    pending_tasks, cached_outcomes = _prepare_sweep_tasks(
        studies=studies,
        configs=configs,
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=context.sweep_dir,
        word2vec_model_base=context.word2vec_model_dir,
        reuse_existing=reuse_for_stage,
    )

    if stage in {"finalize", "reports"} and pending_tasks:
        missing = ", ".join(_format_sweep_task_descriptor(task) for task in pending_tasks[:5])
        more = "" if len(pending_tasks) <= 5 else f", … ({len(pending_tasks)} total)"
        base_message = (
            "Sweep metrics missing for the following tasks: "
            f"{missing}{more}."
        )
        if context.allow_incomplete:
            LOGGER.warning(
                "%s Continuing with available metrics because allow-incomplete mode is enabled.",
                base_message,
            )
        else:
            raise RuntimeError(f"{base_message} Run --stage=sweeps to populate them.")

    executed_outcomes: List[SweepOutcome] = []
    if stage == "full":
        executed_outcomes = _execute_sweep_tasks(pending_tasks, jobs=context.jobs)

    sweep_outcomes = _merge_sweep_outcomes(cached_outcomes, executed_outcomes)
    selections = _select_best_configs(
        outcomes=sweep_outcomes,
        studies=studies,
        allow_incomplete=context.allow_incomplete,
    )

    if stage == "reports":
        slate_metrics = _load_final_metrics_from_disk(
            out_dir=context.out_dir,
            feature_spaces=context.feature_spaces,
            studies=studies,
        )
        if not slate_metrics:
            message = (
                f"No slate metrics found under {context.out_dir}. "
                "Run --stage=finalize before generating reports."
            )
            if context.allow_incomplete:
                LOGGER.warning("%s Continuing because allow-incomplete mode is enabled.", message)
            else:
                raise RuntimeError(message)
        opinion_metrics: Dict[str, Dict[str, Mapping[str, object]]] = {}
        for feature_space in context.feature_spaces:
            metrics = _load_opinion_metrics(context.out_dir, feature_space)
            if metrics:
                opinion_metrics[feature_space] = metrics
        loso_metrics = _load_loso_metrics_from_disk(
            out_dir=context.out_dir,
            feature_spaces=context.feature_spaces,
            studies=studies,
        )
        report_bundle = ReportBundle(
            selections=selections,
            sweep_outcomes=sweep_outcomes,
            studies=studies,
            metrics_by_feature=slate_metrics,
            opinion_metrics=opinion_metrics,
            k_sweep=context.k_sweep,
            loso_metrics=loso_metrics,
            feature_spaces=context.feature_spaces,
            sentence_model=(
                context.sentence_model if "sentence_transformer" in context.feature_spaces else None
            ),
            allow_incomplete=context.allow_incomplete,
        )
        _generate_reports(root, report_bundle)
        return

    slate_metrics = _run_final_evaluations(
        selections=selections,
        studies=studies,
        base_cli=base_cli,
        extra_cli=extra_cli,
        out_dir=context.out_dir,
        word2vec_model_dir=context.word2vec_model_dir,
        reuse_existing=context.reuse_final,
    )

    opinion_metrics = _run_opinion_evaluations(
        selections=selections,
        studies=studies,
        base_cli=base_cli,
        extra_cli=extra_cli,
        out_dir=context.out_dir,
        word2vec_model_dir=context.word2vec_model_dir,
        reuse_existing=context.reuse_final,
    )

    loso_metrics = _run_cross_study_evaluations(
        selections=selections,
        studies=studies,
        base_cli=base_cli,
        extra_cli=extra_cli,
        out_dir=context.out_dir,
        word2vec_model_dir=context.word2vec_model_dir,
        reuse_existing=context.reuse_final,
    )

    if stage == "finalize":
        return

    report_bundle = ReportBundle(
        selections=selections,
        sweep_outcomes=sweep_outcomes,
        studies=studies,
        metrics_by_feature=slate_metrics,
        opinion_metrics=opinion_metrics,
        k_sweep=context.k_sweep,
        loso_metrics=loso_metrics,
        feature_spaces=context.feature_spaces,
        sentence_model=context.sentence_model if "sentence_transformer" in context.feature_spaces else None,
        allow_incomplete=context.allow_incomplete,
    )
    _generate_reports(root, report_bundle)


if __name__ == "__main__":  # pragma: no cover
    main()
