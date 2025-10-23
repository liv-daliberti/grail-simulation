"""Argument parsing helpers for the modular KNN pipeline."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple

from common.cli_args import add_comma_separated_argument

from .cli_utils import add_sentence_transformer_normalize_flags
from common.cli_options import add_jobs_argument, add_stage_arguments

from .pipeline_context import PipelineContext

LOGGER = logging.getLogger("knn.pipeline.cli")


def parse_args(argv: Sequence[str] | None) -> Tuple[argparse.Namespace, List[str]]:
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
        help=(
            "SentenceTransformer model name used during sweeps "
            "(default: sentence-transformers/all-mpnet-base-v2)."
        ),
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
    add_sentence_transformer_normalize_flags(parser, help_prefix="Sweeps:")
    parser.add_argument(
        "--feature-spaces",
        default="",
        help=(
            "Comma-separated feature spaces to evaluate "
            "(default: tfidf,word2vec,sentence_transformer)."
        ),
    )
    parser.add_argument(
        "--issues",
        default="",
        help="Comma-separated list of issues to evaluate. Defaults to all issues in the dataset.",
    )
    add_comma_separated_argument(
        parser,
        flags="--studies",
        dest="studies",
        help_text=(
            "Comma-separated list of participant study keys (study1,study2,study3). "
            "Defaults to all studies."
        ),
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated subset of pipeline tasks to execute (next_video,opinion).",
    )
    parser.add_argument(
        "--reuse-sweeps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse existing sweep metrics when present instead of rerunning the full grid "
            "(use --no-reuse-sweeps to force a full rerun)."
        ),
    )
    parser.add_argument(
        "--reuse-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse cached finalize-stage artefacts when present instead of rerunning evaluations "
            "(use --no-reuse-final to force recomputation)."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow finalize/report stages to proceed with partial sweep data "
            "(use --no-allow-incomplete to require complete sweeps)."
        ),
    )
    add_jobs_argument(parser)
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
    add_stage_arguments(parser)
    parsed, extra = parser.parse_known_args(argv)
    return parsed, list(extra)


def repo_root() -> Path:
    """Return the repository root (two parents above this module)."""

    return Path(__file__).resolve().parents[2]


def default_dataset(root: Path) -> str:
    """Return the default dataset path rooted at ``root``."""

    return str(root / "data" / "cleaned_grail")


def default_cache_dir(root: Path) -> str:
    """Return the default Hugging Face cache directory under ``root``."""

    return str(root / ".cache" / "huggingface" / "knn")


def default_out_dir(root: Path) -> str:
    """Return the default KNN output directory rooted at ``root``."""

    return str(root / "models" / "knn")


def build_pipeline_context(args: argparse.Namespace, root: Path) -> PipelineContext:
    """Normalise CLI/environment options into a reusable context object."""

    dataset = args.dataset or os.environ.get("DATASET") or default_dataset(root)
    out_dir_value = args.out_dir or os.environ.get("OUT_DIR") or default_out_dir(root)
    out_dir = Path(out_dir_value)
    cache_dir_value = args.cache_dir or os.environ.get("CACHE_DIR") or default_cache_dir(root)
    sweep_dir = Path(args.sweep_dir or os.environ.get("KNN_SWEEP_DIR") or (out_dir / "sweeps"))
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
    task_tokens = _split_tokens(
        args.tasks
        or os.environ.get("KNN_PIPELINE_TASKS", "")
    )
    if not task_tokens:
        task_tokens = ["next_video", "opinion"]
    task_flags = {token.lower() for token in task_tokens}
    run_next_video = "next_video" in task_flags or "next" in task_flags
    run_opinion = "opinion" in task_flags

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
    if (
        "SENTENCE_TRANSFORMER_NORMALIZE" in os.environ
        and getattr(args, "sentence_transformer_normalize", None) is not False
    ):
        sentence_normalize_env = os.environ.get("SENTENCE_TRANSFORMER_NORMALIZE", "1")
        sentence_normalize = sentence_normalize_env.lower() not in {"0", "false", "no"}
    else:
        sentence_normalize = bool(getattr(args, "sentence_transformer_normalize", True))

    feature_spaces_tokens = (
        _split_tokens(getattr(args, "feature_spaces", ""))
        or _split_tokens(os.environ.get("KNN_FEATURE_SPACES", ""))
    )
    allowed_spaces = {"tfidf", "word2vec", "sentence_transformer"}
    resolved_feature_spaces = tuple(
        space
        for space in (token.lower() for token in feature_spaces_tokens)
        if space in allowed_spaces
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
        run_next_video=run_next_video,
        run_opinion=run_opinion,
    )


def build_base_cli(context: PipelineContext) -> List[str]:
    """Return the base CLI arguments shared across pipeline steps."""

    base_cli = [
        "--dataset",
        context.dataset,
        "--cache-dir",
        context.cache_dir,
    ]
    if context.run_next_video:
        base_cli.extend(["--fit-index"])
    base_cli.extend(["--overwrite"])
    if context.k_sweep:
        base_cli.extend(["--knn-k-sweep", context.k_sweep])
    return base_cli


def log_run_configuration(studies: Sequence["StudySpec"], context: PipelineContext) -> None:
    """Emit a concise summary of the resolved pipeline configuration."""

    task_tokens: List[str] = []
    if context.run_next_video:
        task_tokens.append("next-video")
    if context.run_opinion:
        task_tokens.append("opinion")
    LOGGER.info("Tasks: %s", ", ".join(task_tokens) if task_tokens else "none")
    LOGGER.info("Dataset: %s", context.dataset)
    LOGGER.info(
        "Studies: %s",
        ", ".join(f"{spec.key} ({spec.issue})" for spec in studies),
    )
    LOGGER.info("Output directory: %s", context.out_dir)
    LOGGER.info("Parallel jobs: %d", context.jobs)


def log_dry_run(configs: Sequence["SweepConfig"]) -> None:
    """Log the number of configurations planned during ``--dry-run``."""

    LOGGER.info("[DRY RUN] Planned %d sweep configurations.", len(configs))


def _split_tokens(raw: str | None) -> List[str]:
    """Split a comma-separated string into trimmed, non-empty tokens."""

    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _default_word2vec_workers() -> int:
    """Return the worker count for Word2Vec training based on environment hints."""

    env_value = os.environ.get("WORD2VEC_WORKERS")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            LOGGER.warning("Ignoring invalid WORD2VEC_WORKERS value '%s'.", env_value)
    max_workers = int(os.environ.get("MAX_WORD2VEC_WORKERS", "40"))
    n_cpus = os.cpu_count() or 1
    return max(1, min(n_cpus, max_workers))


__all__ = [
    "build_base_cli",
    "build_pipeline_context",
    "default_cache_dir",
    "default_dataset",
    "default_out_dir",
    "log_dry_run",
    "log_run_configuration",
    "parse_args",
    "repo_root",
]
