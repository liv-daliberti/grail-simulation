"""High-level orchestration for the XGBoost baselines."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .cli import build_parser as build_xgb_parser
from .data import issues_in_dataset, load_dataset_source
from .evaluate import run_eval
from .model import XGBoostBoosterParams
from .opinion import DEFAULT_SPECS, OpinionEvalRequest, OpinionTrainConfig, run_opinion_eval

LOGGER = logging.getLogger("xgb.pipeline")


@dataclass(frozen=True)
class SweepConfig:
    """Hyper-parameter configuration evaluated during sweeps."""

    text_vectorizer: str
    vectorizer_tag: str
    learning_rate: float
    max_depth: int
    n_estimators: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float
    vectorizer_cli: Tuple[str, ...] = field(default_factory=tuple)

    def label(self) -> str:
        """Filesystem-friendly label."""

        tag = self.vectorizer_tag or self.text_vectorizer
        base = (
            f"lr{self.learning_rate:g}_depth{self.max_depth}_"
            f"estim{self.n_estimators}_sub{self.subsample:g}_"
            f"col{self.colsample_bytree:g}_l2{self.reg_lambda:g}_l1{self.reg_alpha:g}"
        ).replace(".", "p")
        return f"{tag}_{base}"

    def booster_params(self, tree_method: str) -> XGBoostBoosterParams:
        """Convert the sweep config into :class:`XGBoostBoosterParams`."""

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
        """Return CLI overrides encoding this configuration."""

        return [
            "--text_vectorizer",
            self.text_vectorizer,
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
        ] + list(self.vectorizer_cli)


@dataclass(frozen=True)
class StudySpec:
    """Descriptor for a participant study and its associated issue."""

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

    @property
    def evaluation_slug(self) -> str:
        """Return the slug used for evaluation artefacts."""

        return f"{self.issue_slug}_{self.study_slug}"


@dataclass
class SweepOutcome:
    """Metrics captured for a (study, config) combination."""

    order_index: int
    study: StudySpec
    config: SweepConfig
    accuracy: float
    coverage: float
    evaluated: int
    metrics_path: Path
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class SweepTask:
    """Container describing a single sweep execution request."""

    index: int
    study: StudySpec
    config: SweepConfig
    base_cli: Tuple[str, ...]
    extra_cli: Tuple[str, ...]
    run_root: Path
    tree_method: str
    metrics_path: Path


@dataclass
class StudySelection:
    """Selected configuration for the final evaluation of a participant study."""

    study: StudySpec
    outcome: SweepOutcome

    @property
    def config(self) -> SweepConfig:
        """Return the selected sweep configuration for this study."""

        return self.outcome.config

    @property
    def evaluation_slug(self) -> str:
        """Return the evaluation slug used for artefact directories."""

        return self.study.evaluation_slug


@dataclass(frozen=True)
class SweepRunContext:
    """CLI arguments shared across sweep invocations."""

    base_cli: Sequence[str]
    extra_cli: Sequence[str]
    sweep_dir: Path
    tree_method: str
    jobs: int


@dataclass(frozen=True)
class FinalEvalContext:
    """Runtime configuration for final slate evaluations."""

    base_cli: Sequence[str]
    extra_cli: Sequence[str]
    out_dir: Path
    tree_method: str
    save_model_dir: Path | None


@dataclass(frozen=True)
class OpinionStageConfig:
    """Inputs required to launch the opinion regression stage."""

    dataset: str
    cache_dir: str
    base_out_dir: Path
    extra_fields: Sequence[str]
    studies: Sequence[str]
    max_participants: int
    seed: int
    max_features: int | None
    tree_method: str
    overwrite: bool


def _parse_args(argv: Sequence[str] | None) -> Tuple[argparse.Namespace, List[str]]:
    """Return parsed arguments and passthrough CLI flags."""

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
        default="gpu_hist",
        help="Tree construction algorithm passed to XGBoost (default: gpu_hist).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum number of concurrent sweep workers (default: 1).",
    )
    parser.add_argument(
        "--learning-rate-grid",
        default="0.03,0.05,0.07,0.1,0.15,0.2",
        help="Comma-separated learning rates explored during sweeps.",
    )
    parser.add_argument(
        "--max-depth-grid",
        default="3,4,6,8",
        help="Comma-separated integer depths explored during sweeps.",
    )
    parser.add_argument(
        "--n-estimators-grid",
        default="150,250,350,450",
        help="Comma-separated boosting round counts explored during sweeps.",
    )
    parser.add_argument(
        "--subsample-grid",
        default="0.7,0.8,0.9,1.0",
        help="Comma-separated subsample ratios explored during sweeps.",
    )
    parser.add_argument(
        "--colsample-grid",
        default="0.6,0.8,1.0",
        help="Comma-separated column subsample ratios explored during sweeps.",
    )
    parser.add_argument(
        "--reg-lambda-grid",
        default="0.5,1.0,1.5",
        help="Comma-separated L2 regularisation weights explored during sweeps.",
    )
    parser.add_argument(
        "--reg-alpha-grid",
        default="0.0,0.1,0.5,1.0",
        help="Comma-separated L1 regularisation weights explored during sweeps.",
    )
    parser.add_argument(
        "--text-vectorizer-grid",
        default="tfidf",
        help="Comma-separated list of text vectorisers explored during sweeps.",
    )
    parser.add_argument(
        "--word2vec-size",
        type=int,
        default=256,
        help="Word2Vec vector size applied when evaluating the word2vec feature space.",
    )
    parser.add_argument(
        "--word2vec-window",
        type=int,
        default=5,
        help="Word2Vec context window size during training.",
    )
    parser.add_argument(
        "--word2vec_min_count",
        type=int,
        default=2,
        help="Minimum token frequency retained in the Word2Vec vocabulary.",
    )
    parser.add_argument(
        "--word2vec-epochs",
        type=int,
        default=10,
        help="Number of epochs used when training Word2Vec embeddings.",
    )
    parser.add_argument(
        "--word2vec-workers",
        type=int,
        default=1,
        help="Worker threads allocated to Word2Vec training.",
    )
    parser.add_argument(
        "--word2vec-model-dir",
        default="",
        help="Optional directory where Word2Vec models should be stored.",
    )
    parser.add_argument(
        "--sentence-transformer-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name evaluated when using the sentence_transformer feature space.",
    )
    parser.add_argument(
        "--sentence-transformer-device",
        default="",
        help="Optional device string (cpu/cuda) forwarded to SentenceTransformer.",
    )
    parser.add_argument(
        "--sentence-transformer-batch-size",
        type=int,
        default=32,
        help="Encoding batch size for sentence-transformer embeddings.",
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
    parser.add_argument(
        "--allow-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow finalize/report stages to proceed with partial sweeps or metrics (use --no-allow-incomplete to require completeness).",
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
        help="0-based sweep task index executed when --stage=sweeps is used.",
    )
    parser.add_argument(
        "--sweep-task-count",
        type=int,
        default=None,
        help="Expected number of sweep tasks (for validation when distributing sweeps).",
    )

    parsed, extra = parser.parse_known_args(argv)
    return parsed, list(extra)


def _repo_root() -> Path:
    """Return the repository root (two levels above this module)."""

    return Path(__file__).resolve().parents[2]


def _default_out_dir(root: Path) -> Path:
    """Return the default directory storing XGBoost artefacts."""

    return root / "models" / "xgb"


def _default_cache_dir(root: Path) -> Path:
    """Return the default HuggingFace cache directory."""

    return root / ".cache" / "huggingface" / "xgb"


def _default_reports_dir(root: Path) -> Path:
    """Return the directory receiving generated Markdown reports."""

    return root / "reports" / "xgb"


def _split_tokens(raw: str) -> List[str]:
    """Split comma-delimited CLI input into trimmed tokens."""

    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _sanitize_token(value: str) -> str:
    """Return a filesystem-friendly representation of ``value``."""

    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(".", "p")
        .replace(" ", "_")
    )


def _build_sweep_configs(args: argparse.Namespace) -> List[SweepConfig]:
    """Return the hyper-parameter sweep configurations resolved from CLI grids."""

    lr_values = [float(x) for x in _split_tokens(args.learning_rate_grid)]
    depth_values = [int(x) for x in _split_tokens(args.max_depth_grid)]
    estimator_values = [int(x) for x in _split_tokens(args.n_estimators_grid)]
    subsample_values = [float(x) for x in _split_tokens(args.subsample_grid)]
    colsample_values = [float(x) for x in _split_tokens(args.colsample_grid)]
    reg_lambda_values = [float(x) for x in _split_tokens(args.reg_lambda_grid)]
    reg_alpha_values = [float(x) for x in _split_tokens(args.reg_alpha_grid)]
    vectorizer_values = [token.lower() for token in _split_tokens(args.text_vectorizer_grid) or ["tfidf"]]

    def _vectorizer_cli(kind: str) -> Tuple[str, Tuple[str, ...]]:
        """
        Translate a vectorizer identifier into CLI overrides.

        Parameters
        ----------
        kind:
            Name of the vectorizer (``tfidf``, ``word2vec``, ``sentence_transformer``).

        Returns
        -------
        tuple[str, tuple[str, ...]]:
            Pair containing a short configuration tag and the command-line arguments.
        """

        if kind == "tfidf":
            return "tfidf", ()
        if kind == "word2vec":
            cli: List[str] = [
                "--word2vec_size",
                str(args.word2vec_size),
                "--word2vec_window",
                str(args.word2vec_window),
                "--word2vec_min_count",
                str(args.word2vec_min_count),
                "--word2vec_epochs",
                str(args.word2vec_epochs),
                "--word2vec_workers",
                str(args.word2vec_workers),
            ]
            if args.word2vec_model_dir:
                cli.extend(["--word2vec_model_dir", args.word2vec_model_dir])
            tag = f"w2v{args.word2vec_size}"
            return tag, tuple(cli)
        if kind == "sentence_transformer":
            cli = [
                "--sentence_transformer_model",
                args.sentence_transformer_model,
                "--sentence_transformer_batch_size",
                str(args.sentence_transformer_batch_size),
            ]
            if args.sentence_transformer_device:
                cli.extend(["--sentence_transformer_device", args.sentence_transformer_device])
            if args.sentence_transformer_normalize:
                cli.append("--sentence_transformer_normalize")
            else:
                cli.append("--sentence_transformer_no_normalize")
            model_name = args.sentence_transformer_model.split("/")[-1] if args.sentence_transformer_model else kind
            tag = f"st_{_sanitize_token(model_name)}"
            return tag, tuple(cli)
        raise ValueError(f"Unsupported text vectorizer '{kind}' in sweep grid.")

    configs: List[SweepConfig] = []
    for vectorizer in vectorizer_values:
        tag, vectorizer_cli = _vectorizer_cli(vectorizer)
        for values in product(
            lr_values,
            depth_values,
            estimator_values,
            subsample_values,
            colsample_values,
            reg_lambda_values,
            reg_alpha_values,
        ):
            (
                learning_rate,
                max_depth,
                n_estimators,
                subsample,
                colsample_bytree,
                reg_lambda,
                reg_alpha,
            ) = values
            configs.append(
                SweepConfig(
                    vectorizer,
                    tag,
                    learning_rate,
                    max_depth,
                    n_estimators,
                    subsample,
                    colsample_bytree,
                    reg_lambda,
                    reg_alpha,
                    vectorizer_cli,
                )
            )
    return configs


def _resolve_study_specs(
    *,
    dataset: str,
    cache_dir: str,
    requested_issues: Sequence[str],
    requested_studies: Sequence[str],
    allow_incomplete: bool,
) -> List[StudySpec]:
    """Return metadata describing the participant studies slated for evaluation."""

    ds = None
    available_issues: set[str]
    try:
        ds = load_dataset_source(dataset, cache_dir)
    except ImportError as exc:
        if allow_incomplete:
            LOGGER.warning(
                "Unable to load dataset '%s' (%s). Falling back to default study specs because allow-incomplete mode is enabled.",
                dataset,
                exc,
            )
        else:
            raise
    except FileNotFoundError as exc:
        if allow_incomplete:
            LOGGER.warning(
                "Dataset path '%s' missing (%s). Proceeding with default study specs because allow-incomplete mode is enabled.",
                dataset,
                exc,
            )
        else:
            raise

    if ds is not None:
        available_issues = set(issues_in_dataset(ds))
    else:
        available_issues = {spec.issue for spec in DEFAULT_SPECS}

    specs: List[StudySpec] = [
        StudySpec(key=spec.key, issue=spec.issue, label=spec.label)
        for spec in DEFAULT_SPECS
        if spec.issue in available_issues
    ]

    issue_filter = {token for token in requested_issues if token and token.lower() != "all"}
    if issue_filter:
        missing_issues = sorted(issue_filter - available_issues)
        if missing_issues:
            raise ValueError(f"Unknown issues requested: {', '.join(missing_issues)}")
        specs = [spec for spec in specs if spec.issue in issue_filter]

    study_filter = [token for token in requested_studies if token and token.lower() != "all"]
    if study_filter:
        key_map = {spec.key: spec for spec in specs}
        missing_studies = [token for token in study_filter if token not in key_map]
        if missing_studies:
            raise ValueError(f"Unknown studies requested: {', '.join(sorted(missing_studies))}")
        ordered: List[StudySpec] = []
        for token in study_filter:
            spec = key_map.get(token)
            if spec and spec not in ordered:
                ordered.append(spec)
        specs = ordered

    if not specs:
        raise ValueError("No studies selected for evaluation.")
    return specs


def _run_xgb_cli(args: Sequence[str]) -> None:
    """Execute the :mod:`xgb.cli` entry point with ``args``."""

    parser = build_xgb_parser()
    namespace = parser.parse_args(list(args))
    run_eval(namespace)


def _load_metrics(path: Path) -> Mapping[str, object]:
    """Return the metrics dictionary stored at ``path``."""

    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sweep_outcome_from_metrics(
    task: SweepTask,
    metrics: Mapping[str, object],
    metrics_path: Path,
) -> SweepOutcome:
    """Convert cached sweep metrics into an outcome instance."""

    return SweepOutcome(
        order_index=task.index,
        study=task.study,
        config=task.config,
        accuracy=float(metrics.get("accuracy", 0.0)),
        coverage=float(metrics.get("coverage", 0.0)),
        evaluated=int(metrics.get("evaluated", 0)),
        metrics_path=metrics_path,
        metrics=metrics,
    )


def _prepare_sweep_tasks(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: SweepRunContext,
    reuse_existing: bool,
) -> Tuple[List[SweepTask], List[SweepOutcome]]:
    """Return tasks requiring execution and any cached outcomes."""

    pending: List[SweepTask] = []
    cached: List[SweepOutcome] = []
    base_cli_tuple = tuple(context.base_cli)
    extra_cli_tuple = tuple(context.extra_cli)

    task_index = 0
    for config in configs:
        for study in studies:
            run_root = context.sweep_dir / study.issue_slug / study.study_slug / config.label()
            metrics_path = run_root / study.evaluation_slug / "metrics.json"
            task = SweepTask(
                index=task_index,
                study=study,
                config=config,
                base_cli=base_cli_tuple,
                extra_cli=extra_cli_tuple,
                run_root=run_root,
                tree_method=context.tree_method,
                metrics_path=metrics_path,
            )
            task_index += 1
            if reuse_existing and metrics_path.exists():
                LOGGER.info(
                    "[SWEEP][SKIP] issue=%s study=%s config=%s (cached).",
                    study.issue,
                    study.key,
                    config.label(),
                )
                metrics = _load_metrics(metrics_path)
                cached.append(_sweep_outcome_from_metrics(task, metrics, metrics_path))
                continue
            pending.append(task)
    return pending, cached


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
                "Duplicate sweep outcome for index=%d; replacing cached result.",
                outcome.order_index,
            )
        by_index[outcome.order_index] = outcome
    return [by_index[index] for index in sorted(by_index)]


def _execute_sweep_tasks(
    tasks: Sequence[SweepTask],
    *,
    jobs: int,
) -> List[SweepOutcome]:
    """Execute sweep tasks, optionally in parallel."""

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
    """Print a concise sweep plan listing."""

    print(f"TOTAL_TASKS={len(tasks)}")
    if not tasks:
        return
    print("INDEX\tSTUDY\tISSUE\tVECTORIZER\tLABEL")
    for display_index, task in enumerate(tasks):
        print(
            f"{display_index}\t{task.study.key}\t{task.study.issue}\t"
            f"{task.config.text_vectorizer}\t{task.config.label()}"
        )


def _format_sweep_task_descriptor(task: SweepTask) -> str:
    """Return a short descriptor for a sweep task."""

    return f"{task.study.key}:{task.study.issue}:{task.config.label()}"


def _load_final_metrics_from_disk(
    *,
    next_video_dir: Path,
    studies: Sequence[StudySpec],
) -> Dict[str, Mapping[str, object]]:
    """Load persisted final evaluation metrics per study."""

    metrics_by_study: Dict[str, Mapping[str, object]] = {}
    for spec in studies:
        metrics_path = next_video_dir / spec.evaluation_slug / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = dict(_load_metrics(metrics_path))
        metrics.setdefault("issue", spec.issue)
        metrics.setdefault("issue_label", spec.issue.replace("_", " ").title())
        metrics.setdefault("study", spec.key)
        metrics.setdefault("study_label", spec.label)
        metrics_by_study[spec.key] = metrics
    return metrics_by_study


def _load_opinion_metrics_from_disk(
    *,
    opinion_dir: Path,
    studies: Sequence[StudySpec],
) -> Dict[str, Dict[str, object]]:
    """Load opinion regression metrics saved by previous runs."""

    results: Dict[str, Dict[str, object]] = {}
    for spec in studies:
        metrics_path = (
            opinion_dir / spec.key / f"opinion_xgb_{spec.key}_validation_metrics.json"
        )
        if not metrics_path.exists():
            continue
        metrics = dict(_load_metrics(metrics_path))
        results[spec.key] = metrics
    return results


def _run_sweeps(
    *,
    studies: Sequence[StudySpec],
    configs: Sequence[SweepConfig],
    context: SweepRunContext,
) -> List[SweepOutcome]:
    """Execute hyper-parameter sweeps and collect outcome metadata."""

    pending_tasks, cached_outcomes = _prepare_sweep_tasks(
        studies=studies,
        configs=configs,
        context=context,
        reuse_existing=False,
    )
    executed_outcomes = _execute_sweep_tasks(pending_tasks, jobs=context.jobs)
    return _merge_sweep_outcomes(cached_outcomes, executed_outcomes)


def _execute_sweep_task(task: SweepTask) -> SweepOutcome:
    """Execute a single XGBoost sweep task and return the resulting metrics."""

    run_root = task.run_root
    run_root.mkdir(parents=True, exist_ok=True)

    cli_args: List[str] = list(task.base_cli)
    cli_args.extend(task.config.cli_args(task.tree_method))
    cli_args.extend(["--issues", task.study.issue])
    cli_args.extend(["--participant_studies", task.study.key])
    cli_args.extend(["--out_dir", str(run_root)])
    cli_args.extend(task.extra_cli)

    LOGGER.info(
        "[SWEEP] issue=%s study=%s config=%s",
        task.study.issue,
        task.study.key,
        task.config.label(),
    )
    _run_xgb_cli(cli_args)

    metrics = _load_metrics(task.metrics_path)
    return SweepOutcome(
        order_index=task.index,
        study=task.study,
        config=task.config,
        accuracy=float(metrics.get("accuracy", 0.0)),
        coverage=float(metrics.get("coverage", 0.0)),
        evaluated=int(metrics.get("evaluated", 0)),
        metrics_path=task.metrics_path,
        metrics=metrics,
    )


def _select_best_configs(outcomes: Sequence[SweepOutcome]) -> Dict[str, StudySelection]:
    """Pick the best configuration per study using accuracy, coverage, and support."""

    selections: Dict[str, StudySelection] = {}

    for outcome in outcomes:
        current = selections.get(outcome.study.key)
        if current is None:
            selections[outcome.study.key] = StudySelection(study=outcome.study, outcome=outcome)
            continue
        incumbent = current.outcome
        if outcome.accuracy > incumbent.accuracy + 1e-9:
            selections[outcome.study.key] = StudySelection(study=outcome.study, outcome=outcome)
            continue
        if incumbent.accuracy - outcome.accuracy <= 1e-9:
            if outcome.coverage > incumbent.coverage + 1e-9:
                selections[outcome.study.key] = StudySelection(
                    study=outcome.study,
                    outcome=outcome,
                )
            elif (
                abs(outcome.coverage - incumbent.coverage) <= 1e-9
                and outcome.evaluated > incumbent.evaluated
            ):
                selections[outcome.study.key] = StudySelection(
                    study=outcome.study,
                    outcome=outcome,
                )
    return selections


def _run_final_evaluations(
    *,
    selections: Mapping[str, StudySelection],
    context: FinalEvalContext,
) -> Dict[str, Mapping[str, object]]:
    """Run the final next-video evaluations for each selected configuration."""

    metrics_by_study: Dict[str, Mapping[str, object]] = {}
    context.out_dir.mkdir(parents=True, exist_ok=True)

    for study_key, selection in selections.items():
        cli_args: List[str] = []
        cli_args.extend(context.base_cli)
        cli_args.extend(selection.config.cli_args(context.tree_method))
        cli_args.extend(["--issues", selection.study.issue])
        cli_args.extend(["--participant_studies", selection.study.key])
        cli_args.extend(["--out_dir", str(context.out_dir)])
        if context.save_model_dir is not None:
            cli_args.extend(["--save_model", str(context.save_model_dir)])
        cli_args.extend(context.extra_cli)
        LOGGER.info(
            "[FINAL] issue=%s study=%s config=%s",
            selection.study.issue,
            selection.study.key,
            selection.config.label(),
        )
        _run_xgb_cli(cli_args)
        metrics_path = context.out_dir / selection.evaluation_slug / "metrics.json"
        metrics = dict(_load_metrics(metrics_path))
        metrics.setdefault("issue", selection.study.issue)
        metrics.setdefault("issue_label", selection.study.issue.replace("_", " ").title())
        metrics.setdefault("study", selection.study.key)
        metrics.setdefault("study_label", selection.study.label)
        metrics_by_study[study_key] = metrics
    return metrics_by_study


def _run_opinion_stage(
    *,
    selections: Mapping[str, StudySelection],
    config: OpinionStageConfig,
) -> Dict[str, Dict[str, object]]:
    """Execute the optional opinion regression stage for selected participant studies."""

    if not selections:
        LOGGER.warning("Skipping opinion stage; no selections available.")
        return {}

    opinion_out_dir = config.base_out_dir / "opinion"
    requested = [token for token in config.studies if token and token.lower() != "all"]
    if not requested:
        requested = [spec.key for spec in DEFAULT_SPECS]

    results: Dict[str, Dict[str, object]] = {}
    for study_key in requested:
        selection = selections.get(study_key)
        if selection is None:
            LOGGER.warning(
                "Skipping opinion study for study=%s (no selection available).",
                study_key,
            )
            continue
        opinion_config = OpinionTrainConfig(
            max_participants=config.max_participants,
            seed=config.seed,
            max_features=config.max_features,
            booster=selection.config.booster_params(config.tree_method),
        )
        payload = run_opinion_eval(
            request=OpinionEvalRequest(
                dataset=config.dataset,
                cache_dir=config.cache_dir,
                out_dir=opinion_out_dir,
                feature_space="tfidf",
                extra_fields=config.extra_fields,
                train_config=opinion_config,
                overwrite=config.overwrite,
            ),
            studies=[study_key],
        )
        results.update(payload)
    return results


def _format_float(value: float) -> str:
    """Format a floating-point metric with three decimal places."""

    return f"{value:.3f}"


def _write_reports(
    *,
    reports_dir: Path,
    outcomes: Sequence[SweepOutcome],
    selections: Mapping[str, StudySelection],
    final_metrics: Mapping[str, Mapping[str, object]],
    opinion_metrics: Mapping[str, Mapping[str, object]],
    allow_incomplete: bool,
) -> None:
    """Write the full report bundle capturing sweep and evaluation artefacts."""

    reports_dir.mkdir(parents=True, exist_ok=True)

    _write_catalog_report(reports_dir)
    _write_hyperparameter_report(
        reports_dir / "hyperparameter_tuning",
        outcomes,
        selections,
        allow_incomplete=allow_incomplete,
    )
    _write_next_video_report(
        reports_dir / "next_video",
        final_metrics,
        selections,
        allow_incomplete=allow_incomplete,
    )
    _write_opinion_report(
        reports_dir / "opinion",
        opinion_metrics,
        allow_incomplete=allow_incomplete,
    )


def _write_catalog_report(reports_dir: Path) -> None:
    """Create the catalog README summarising generated artefacts."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "README.md"
    lines: List[str] = []
    lines.append("# XGBoost Report Catalog")
    lines.append("")
    lines.append("Generated artefacts for the XGBoost baselines:")
    lines.append("")
    lines.append("- `next_video/` – slate-ranking accuracy for the selected configuration.")
    lines.append("- `opinion/` – post-study opinion regression results.")
    lines.append(
        "- `hyperparameter_tuning/` – notes from the sweep that selected the configuration."
    )
    lines.append("")
    lines.append("Model checkpoints and raw metrics live under `models/xgb/`.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_hyperparameter_report(
    directory: Path,
    outcomes: Sequence[SweepOutcome],
    selections: Mapping[str, StudySelection],
    *,
    allow_incomplete: bool,
) -> None:
    """Create the hyper-parameter sweep summary document."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "README.md"
    lines: List[str] = []
    lines.append("# Hyper-parameter Tuning")
    lines.append("")
    lines.append(
        "This summary lists the XGBoost configurations explored for each participant study. "
        "The selected configuration is highlighted in bold."
    )
    lines.append("")
    per_study: Dict[str, List[SweepOutcome]] = {}
    for outcome in outcomes:
        per_study.setdefault(outcome.study.key, []).append(outcome)

    if not per_study:
        lines.append("No sweep runs were available when this report was generated.")
        if allow_incomplete:
            lines.append(
                "Run the XGBoost pipeline with `--stage sweeps` or `--stage full` once artefacts are ready."
            )
        lines.append("")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    def _study_label(study_key: str) -> str:
        selection = selections.get(study_key)
        if selection is not None:
            return selection.study.label
        study_outcomes = per_study.get(study_key)
        if study_outcomes:
            return study_outcomes[0].study.label
        return study_key

    sorted_study_outcomes: Dict[str, List[SweepOutcome]] = {}
    for study_key in sorted(per_study.keys(), key=lambda key: _study_label(key).lower()):
        study_outcomes = per_study[study_key]
        selection = selections.get(study_key)
        study_label = _study_label(study_key)
        issue_label = study_outcomes[0].study.issue.replace("_", " ").title()
        lines.append(f"## {study_label}")
        lines.append("")
        lines.append(f"*Issue:* {issue_label}")
        lines.append("")
        lines.append("| Config | Accuracy ↑ | Coverage ↑ | Evaluated |")
        lines.append("| --- | ---: | ---: | ---: |")
        ordered = sorted(
            study_outcomes,
            key=lambda item: (item.accuracy, item.coverage, item.evaluated),
            reverse=True,
        )
        sorted_study_outcomes[study_key] = ordered
        for outcome in ordered:
            label = outcome.config.label()
            formatted = (
                f"**{label}**" if selection and outcome.config == selection.config else label
            )
            lines.append(
                f"| {formatted} | {_format_float(outcome.accuracy)} | "
                f"{_format_float(outcome.coverage)} | {outcome.evaluated} |"
            )
        lines.append("")

    if selections:
        lines.extend(_xgb_selection_summary_section(sorted_study_outcomes, selections))
        lines.extend(_xgb_parameter_frequency_section(selections))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _xgb_selection_summary_section(
    per_study_sorted: Mapping[str, Sequence[SweepOutcome]],
    selections: Mapping[str, StudySelection],
) -> List[str]:
    """Return a bullet summary comparing winning configurations to runner-ups."""

    lines: List[str] = ["### Selection Summary", ""]
    for study_key in sorted(per_study_sorted.keys(), key=lambda key: selections.get(key).study.label.lower() if selections.get(key) else per_study_sorted[key][0].study.label.lower()):
        selection = selections.get(study_key)
        ordered = per_study_sorted.get(study_key, [])
        if not ordered:
            continue
        descriptor = selection.study.label if selection is not None else ordered[0].study.label
        issue_label = ordered[0].study.issue.replace("_", " ").title()
        descriptor_full = f"{descriptor} (issue {issue_label})"
        if selection is None:
            top = ordered[0]
            lines.append(
                f"- **{descriptor_full}**: accuracy {_format_float(top.accuracy)} "
                f"with {_summarise_xgb_config(top.config)}."
            )
            continue
        best = selection.outcome
        summary = _summarise_xgb_config(selection.config)
        runner_up = ordered[1] if len(ordered) > 1 else None
        if runner_up is not None:
            delta_acc = best.accuracy - runner_up.accuracy
            delta_cov = best.coverage - runner_up.coverage
            lines.append(
                f"- **{descriptor_full}**: accuracy {_format_float(best.accuracy)} "
                f"(coverage {_format_float(best.coverage)}) using {summary}. "
                f"Δ accuracy vs. runner-up { _format_float(delta_acc) }, Δ coverage { _format_float(delta_cov) }."
            )
        else:
            lines.append(
                f"- **{descriptor_full}**: accuracy {_format_float(best.accuracy)} "
                f"(coverage {_format_float(best.coverage)}) using {summary}."
            )
    lines.append("")
    return lines


def _xgb_parameter_frequency_section(
    selections: Mapping[str, StudySelection],
) -> List[str]:
    """Summarise the hyper-parameter values chosen across all studies."""

    if not selections:
        return []

    param_counters = {
        "text_vectorizer": Counter(),
        "learning_rate": Counter(),
        "max_depth": Counter(),
        "n_estimators": Counter(),
        "subsample": Counter(),
        "colsample_bytree": Counter(),
        "reg_lambda": Counter(),
        "reg_alpha": Counter(),
    }

    for selection in selections.values():
        config = selection.config
        param_counters["text_vectorizer"][config.text_vectorizer] += 1
        param_counters["learning_rate"][config.learning_rate] += 1
        param_counters["max_depth"][config.max_depth] += 1
        param_counters["n_estimators"][config.n_estimators] += 1
        param_counters["subsample"][config.subsample] += 1
        param_counters["colsample_bytree"][config.colsample_bytree] += 1
        param_counters["reg_lambda"][config.reg_lambda] += 1
        param_counters["reg_alpha"][config.reg_alpha] += 1

    display_names = {
        "text_vectorizer": "Vectorizer",
        "learning_rate": "Learning rate",
        "max_depth": "Max depth",
        "n_estimators": "Estimators",
        "subsample": "Subsample",
        "colsample_bytree": "Column subsample",
        "reg_lambda": "L2 regularisation",
        "reg_alpha": "L1 regularisation",
    }

    lines: List[str] = ["### Parameter Frequency Across Selected Configurations", ""]
    lines.append("| Parameter | Preferred values (count) |")
    lines.append("| --- | --- |")
    for key in display_names:
        counter = param_counters.get(key, Counter())
        lines.append(
            f"| {display_names[key]} | {_format_param_counter(counter)} |"
        )
    lines.append("")
    return lines


def _summarise_xgb_config(config: SweepConfig) -> str:
    """Return a human-readable description of a sweep configuration."""

    vectorizer = config.text_vectorizer
    if config.vectorizer_tag and config.vectorizer_tag != config.text_vectorizer:
        vectorizer = f"{vectorizer} ({config.vectorizer_tag})"
    return (
        f"vectorizer={vectorizer}, lr={config.learning_rate:g}, depth={config.max_depth}, "
        f"estimators={config.n_estimators}, subsample={config.subsample:g}, "
        f"colsample={config.colsample_bytree:g}, λ={config.reg_lambda:g}, α={config.reg_alpha:g}"
    )


def _format_param_counter(counter: Counter) -> str:
    """Format a parameter usage counter for Markdown output."""

    if not counter:
        return "—"
    parts = []
    for value, count in counter.most_common():
        if isinstance(value, float):
            value_repr = f"{value:g}"
        else:
            value_repr = str(value)
        parts.append(f"{value_repr} ×{count}")
    return ", ".join(parts)


def _write_next_video_report(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
    selections: Mapping[str, StudySelection],
    *,
    allow_incomplete: bool,
) -> None:
    """Create the next-video evaluation summary document."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "README.md"
    lines: List[str] = []
    lines.append("# XGBoost Next-Video Baseline")
    lines.append("")
    lines.append("Accuracy on the validation split for the selected slate configuration.")
    lines.append("")

    if not metrics:
        lines.append("No finalized evaluation metrics were available when this report was generated.")
        if allow_incomplete:
            lines.append(
                "Run the pipeline with `--stage finalize` once sufficient artefacts exist to refresh this table."
            )
        lines.append("")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("| Study | Issue | Accuracy ↑ | Coverage ↑ | Evaluated |")
    lines.append("| --- | --- | ---: | ---: | ---: |")

    def _study_label(study_key: str) -> str:
        selection = selections.get(study_key)
        if selection is not None:
            return selection.study.label
        return study_key

    def _issue_label(study_key: str, fallback: str) -> str:
        selection = selections.get(study_key)
        if selection is not None:
            return selection.study.issue.replace("_", " ").title()
        if fallback:
            return str(fallback).replace("_", " ").title()
        return ""

    for study_key in sorted(metrics.keys(), key=lambda key: _study_label(key).lower()):
        summary = metrics[study_key]
        study_label = summary.get("study_label") or _study_label(study_key)
        issue_label = summary.get("issue_label") or _issue_label(
            study_key, summary.get("issue", study_key)
        )
        lines.append(
            f"| {study_label} | {issue_label or _issue_label(study_key, '')} | "
            f"{_format_float(float(summary.get('accuracy', 0.0)))} | "
            f"{_format_float(float(summary.get('coverage', 0.0)))} | "
            f"{int(summary.get('evaluated', 0))} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_opinion_report(
    directory: Path,
    metrics: Mapping[str, Mapping[str, object]],
    *,
    allow_incomplete: bool,
) -> None:
    """Create the opinion regression summary document."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "README.md"
    lines: List[str] = []
    lines.append("# XGBoost Opinion Regression")
    lines.append("")
    if not metrics:
        lines.append("No opinion runs were produced during this pipeline invocation.")
        if allow_incomplete:
            lines.append(
                "Rerun the pipeline with `--stage finalize` to populate this section once opinion metrics are available."
            )
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
    """Entry point orchestrating the full XGBoost workflow."""

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

    jobs_value = getattr(args, "jobs", 1) or 1
    env_jobs = os.environ.get("XGB_JOBS")
    if env_jobs:
        try:
            jobs_value = int(env_jobs)
        except ValueError:
            LOGGER.warning("Ignoring invalid XGB_JOBS value '%s'.", env_jobs)
    jobs = max(1, jobs_value)

    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    os.environ.setdefault("HF_HOME", cache_dir)

    allow_incomplete = getattr(args, "allow_incomplete", True)
    env_allow = os.environ.get("XGB_ALLOW_INCOMPLETE")
    if env_allow is not None:
        allow_incomplete = env_allow.lower() not in {"0", "false", "no"}

    issue_tokens = _split_tokens(args.issues)
    study_tokens = _split_tokens(args.studies)
    study_specs = _resolve_study_specs(
        dataset=dataset,
        cache_dir=cache_dir,
        requested_issues=issue_tokens,
        requested_studies=study_tokens,
        allow_incomplete=allow_incomplete,
    )
    extra_fields = _split_tokens(args.extra_text_fields)

    LOGGER.info("Parallel sweep jobs: %d", jobs)
    tree_method = args.tree_method or "gpu_hist"
    args.tree_method = tree_method
    LOGGER.info("Using XGBoost tree_method=%s.", tree_method)

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
    stage = getattr(args, "stage", "full")
    reuse_sweeps = not args.overwrite

    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_context = SweepRunContext(
        base_cli=base_cli,
        extra_cli=extra_cli,
        sweep_dir=sweep_dir,
        tree_method=args.tree_method,
        jobs=jobs,
    )

    planned_tasks, cached_planned = _prepare_sweep_tasks(
        studies=study_specs,
        configs=configs,
        context=sweep_context,
        reuse_existing=reuse_sweeps,
    )

    if stage == "plan":
        LOGGER.info(
            "Planned %d sweep configurations (%d cached).",
            len(planned_tasks),
            len(cached_planned),
        )
        _emit_sweep_plan(planned_tasks)
        return

    if args.dry_run:
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
                    "Sweep stage requires --sweep-task-id or SLURM_ARRAY_TASK_ID to be set."
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
        if reuse_sweeps and task.metrics_path.exists():
            LOGGER.info(
                "Skipping sweep task %d (%s | %s | %s); metrics already exist at %s.",
                task.index,
                task.study.key,
                task.study.issue,
                task.config.label(),
                task.metrics_path,
            )
            return
        outcome = _execute_sweep_task(task)
        LOGGER.info(
            "Completed sweep task %d (%s | %s | %s). Metrics stored at %s.",
            outcome.order_index,
            task.study.key,
            task.study.issue,
            task.config.label(),
            outcome.metrics_path,
        )
        return

    reuse_for_stage = reuse_sweeps
    if stage in {"finalize", "reports"}:
        reuse_for_stage = True
    pending_tasks, cached_outcomes = _prepare_sweep_tasks(
        studies=study_specs,
        configs=configs,
        context=sweep_context,
        reuse_existing=reuse_for_stage,
    )
    if stage in {"finalize", "reports"} and pending_tasks:
        missing = ", ".join(_format_sweep_task_descriptor(task) for task in pending_tasks[:5])
        more = "" if len(pending_tasks) <= 5 else f", … ({len(pending_tasks)} total)"
        message = (
            "Sweep metrics missing for the following tasks: "
            f"{missing}{more}. Run --stage=sweeps to populate them."
        )
        if allow_incomplete:
            LOGGER.warning("%s Continuing because allow-incomplete mode is enabled.", message)
        else:
            raise RuntimeError(message)

    executed_outcomes: List[SweepOutcome] = []
    if stage == "full":
        executed_outcomes = _execute_sweep_tasks(pending_tasks, jobs=jobs)

    outcomes = _merge_sweep_outcomes(cached_outcomes, executed_outcomes)
    if not outcomes:
        if allow_incomplete:
            LOGGER.warning(
                "No sweep outcomes available; reports will contain placeholders because allow-incomplete mode is enabled."
            )
        else:
            raise RuntimeError("No sweep outcomes available; ensure sweeps have completed.")

    selections = _select_best_configs(outcomes)
    if not selections:
        if allow_incomplete:
            LOGGER.warning(
                "Failed to select configurations for any study; downstream reports will rely on placeholders."
            )
        else:
            raise RuntimeError("Failed to select a configuration for any study.")
    else:
        LOGGER.info(
            "Selected configurations: %s",
            ", ".join(
                f"{selection.study.key} ({selection.study.issue})"
                for selection in selections.values()
            ),
        )

    final_eval_context = FinalEvalContext(
        base_cli=base_cli,
        extra_cli=extra_cli,
        out_dir=out_dir / "next_video",
        tree_method=args.tree_method,
        save_model_dir=Path(args.save_model_dir) if args.save_model_dir else None,
    )

    if stage == "reports":
        final_metrics = _load_final_metrics_from_disk(
            next_video_dir=final_eval_context.out_dir,
            studies=study_specs,
        )
        if not final_metrics:
            message = (
                f"No final metrics found under {final_eval_context.out_dir}. "
                "Run --stage=finalize before generating reports."
            )
            if allow_incomplete:
                LOGGER.warning("%s Continuing because allow-incomplete mode is enabled.", message)
            else:
                raise RuntimeError(message)
        opinion_metrics = _load_opinion_metrics_from_disk(
            opinion_dir=out_dir / "opinion",
            studies=study_specs,
        )
        _write_reports(
            reports_dir=reports_dir,
            outcomes=outcomes,
            selections=selections,
            final_metrics=final_metrics,
            opinion_metrics=opinion_metrics,
            allow_incomplete=allow_incomplete,
        )
        return

    final_metrics = _run_final_evaluations(selections=selections, context=final_eval_context)

    opinion_stage_config = OpinionStageConfig(
        dataset=dataset,
        cache_dir=cache_dir,
        base_out_dir=out_dir,
        extra_fields=extra_fields,
        studies=study_tokens,
        max_participants=args.opinion_max_participants,
        seed=args.seed,
        max_features=args.max_features if args.max_features > 0 else None,
        tree_method=args.tree_method,
        overwrite=args.overwrite,
    )
    opinion_metrics = _run_opinion_stage(selections=selections, config=opinion_stage_config)

    if stage == "finalize":
        return

    _write_reports(
        reports_dir=reports_dir,
        outcomes=outcomes,
        selections=selections,
        final_metrics=final_metrics,
        opinion_metrics=opinion_metrics,
        allow_incomplete=allow_incomplete,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
