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

"""Command-line helpers for the Grail Simulation KNN pipeline.

The functions here construct the CLI parser, resolve default directories,
and translate parsed arguments into sweep configurations consumed by
``knn.pipeline`` and its orchestration helpers.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple

from common.cli.options import add_jobs_argument, add_stage_arguments, add_studies_argument

from ..cli.utils import add_sentence_transformer_normalize_flags
from .context import PipelineContext

LOGGER = logging.getLogger("knn.pipeline.cli")

def parse_args(argv: Sequence[str] | None) -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse high-level pipeline arguments and capture any extra CLI tokens.

    :param argv: Explicit argument vector used for parsing. When ``None``,
        the function consumes :data:`sys.argv`.
    :type argv: Sequence[str] | None
    :returns: Pair containing the parsed namespace and a list of passthrough arguments that should
        be forwarded to downstream CLI invocations.
    :rtype: Tuple[argparse.Namespace, List[str]]
    """
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
        help=(
            "Directory for persisted Word2Vec models "
            "(default: <out-dir>/next_video/word2vec_models)."
        ),
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
    add_studies_argument(
        parser,
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
        default=False,
        help=(
            "Reuse existing sweep metrics when present instead of rerunning the full grid "
            "(disabled by default; pass --reuse-sweeps to enable)."
        ),
    )
    parser.add_argument(
        "--reuse-final",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse cached finalize-stage artefacts when present instead of rerunning evaluations "
            "(disabled by default; pass --reuse-final to enable)."
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
        help="Directory for hyper-parameter sweeps (default: <out-dir>/next_video/sweeps).",
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
    """
    Compute the repository root by traversing upwards from this module.

    :returns: Absolute path to the project root (three directories above ``pipeline.cli``).
    :rtype: Path
    """
    return Path(__file__).resolve().parents[3]

def default_dataset(root: Path) -> str:
    """
    Resolve the default dataset location under the repository.

    :param root: Repository root from which data directories are derived.
    :type root: Path
    :returns: Filesystem path pointing to ``data/cleaned_grail`` beneath ``root``.
    :rtype: str
    """
    return str(root / "data" / "cleaned_grail")

def default_cache_dir(root: Path) -> str:
    """
    Resolve the datasets cache directory used by the pipeline.

    :param root: Repository root from which cache directories are derived.
    :type root: Path
    :returns: Path to ``.cache/huggingface/knn`` rooted under ``root``.
    :rtype: str
    """
    return str(root / ".cache" / "huggingface" / "knn")

def default_out_dir(root: Path) -> str:
    """
    Resolve the default directory for all pipeline outputs.

    :param root: Repository root from which output directories are derived.
    :type root: Path
    :returns: Path to ``models/knn`` rooted under ``root``.
    :rtype: str
    """
    return str(root / "models" / "knn")


def _bool_from_env(value: str | None, default: bool) -> bool:
    """Interpret ``value`` as a boolean, falling back to ``default`` when unset."""

    if value is None:
        return default
    return value.lower() not in {"0", "false", "no"}


def _resolve_reuse_flag(default: bool, env_key: str) -> bool:
    """Resolve a reuse flag by merging CLI, environment, and defaults."""

    return _bool_from_env(os.environ.get(env_key), default)


def _resolve_sentence_settings(
    args: argparse.Namespace,
) -> Tuple[str, str | None, int, bool]:
    """Normalise sentence-transformer configuration across CLI and environment."""

    model = (
        args.sentence_transformer_model
        or os.environ.get("SENTENCE_TRANSFORMER_MODEL")
        or "sentence-transformers/all-mpnet-base-v2"
    )
    device_raw = (
        args.sentence_transformer_device
        or os.environ.get("SENTENCE_TRANSFORMER_DEVICE")
        or ""
    )
    device = device_raw or None
    batch_size = int(
        args.sentence_transformer_batch_size
        or os.environ.get("SENTENCE_TRANSFORMER_BATCH_SIZE", "32")
    )
    normalize_env = os.environ.get("SENTENCE_TRANSFORMER_NORMALIZE")
    normalize_flag = getattr(args, "sentence_transformer_normalize", None)
    if normalize_env is not None and normalize_flag is not False:
        normalize = normalize_env.lower() not in {"0", "false", "no"}
    else:
        normalize = bool(getattr(args, "sentence_transformer_normalize", True))
    return model, device, batch_size, normalize


def _resolve_feature_spaces(args: argparse.Namespace) -> Tuple[str, ...]:
    """Resolve the feature spaces requested for KNN sweeps."""

    tokens = (
        _split_tokens(getattr(args, "feature_spaces", ""))
        or _split_tokens(os.environ.get("KNN_FEATURE_SPACES", ""))
    )
    allowed = {"tfidf", "word2vec", "sentence_transformer"}
    resolved = tuple(
        space
        for space in (token.lower() for token in tokens)
        if space in allowed
    )
    return resolved or ("tfidf", "word2vec", "sentence_transformer")


def _resolve_jobs(args: argparse.Namespace) -> int:
    """Resolve the parallel job count, respecting environment overrides."""

    jobs_value = getattr(args, "jobs", 1) or 1
    env_jobs = os.environ.get("KNN_JOBS")
    if env_jobs:
        try:
            jobs_value = int(env_jobs)
        except ValueError:
            LOGGER.warning("Ignoring invalid KNN_JOBS value '%s'.", env_jobs)
    return max(1, jobs_value)


def _resolve_task_flags(tokens: Sequence[str]) -> Tuple[bool, bool]:
    """Return the pipeline task toggles derived from ``tokens``."""

    next_aliases = {"next_video", "next", "next-video", "nextvideo", "slate"}
    opinion_aliases = {"opinion", "opinion_stage", "opinion-stage"}
    run_next = False
    run_opinion = False
    unknown: List[str] = []
    for token in tokens:
        normalised = token.strip().lower()
        if normalised in next_aliases:
            run_next = True
        elif normalised in opinion_aliases:
            run_opinion = True
        elif normalised:
            unknown.append(token.strip() or token)
    if unknown:
        LOGGER.warning(
            "Ignoring unknown pipeline task token(s): %s",
            ", ".join(sorted(set(unknown))),
        )
    if not run_next and not run_opinion:
        LOGGER.warning(
            "No recognised pipeline tasks provided; defaulting to next_video and opinion."
        )
        run_next = True
        run_opinion = True
    return run_next, run_opinion


def _resolve_task_configuration(args: argparse.Namespace) -> Tuple[bool, bool]:
    """Resolve the next-video/opinion task toggles for the run."""

    tokens = _split_tokens(
        args.tasks
        or os.environ.get("KNN_PIPELINE_TASKS", "")
    )
    if not tokens:
        tokens = ["next_video", "opinion"]
    return _resolve_task_flags(tokens)


def _resolve_allow_incomplete(args: argparse.Namespace) -> bool:
    """Resolve allow-incomplete behaviour from CLI and environment."""

    return _bool_from_env(
        os.environ.get("KNN_ALLOW_INCOMPLETE"),
        getattr(args, "allow_incomplete", True),
    )

def build_pipeline_context(args: argparse.Namespace, root: Path) -> PipelineContext:
    """
    Normalise CLI flags and environment overrides into a :class:`PipelineContext`.

    :param args: Parsed pipeline arguments that may omit optional values.
    :type args: argparse.Namespace
    :param root: Repository root, used to derive sensible defaults for directories.
    :type root: Path
    :returns: Fully-populated context capturing dataset paths, sweep configuration,
        Word2Vec parameters, SentenceTransformer settings, and task toggles.
    :rtype: ~knn.pipeline.context.PipelineContext
    """
    paths = _resolve_paths(args, root)
    settings = _resolve_settings(args)
    return _build_context_from(paths, settings)


def _resolve_paths(args: argparse.Namespace, root: Path) -> dict:
    """Resolve dataset and directory paths from CLI and environment."""
    out_dir_value = args.out_dir or os.environ.get("OUT_DIR") or default_out_dir(root)
    out_dir = Path(out_dir_value)
    next_video_dir = out_dir / "next_video"
    opinion_dir = out_dir / "opinions"
    return {
        "dataset": args.dataset or os.environ.get("DATASET") or default_dataset(root),
        "out_dir": out_dir,
        "cache_dir": args.cache_dir or os.environ.get("CACHE_DIR") or default_cache_dir(root),
        "sweep_dir": Path(
            getattr(args, "sweep_dir", None)
            or os.environ.get("KNN_SWEEP_DIR")
            or (next_video_dir / "sweeps")
        ),
        "opinion_sweep_dir": Path(
            os.environ.get("KNN_OPINION_SWEEP_DIR") or (opinion_dir / "sweeps")
        ),
        "word2vec_model_dir": Path(
            args.word2vec_model_dir
            or os.environ.get("WORD2VEC_MODEL_DIR")
            or (next_video_dir / "word2vec_models")
        ),
        "opinion_word2vec_dir": Path(
            os.environ.get("KNN_OPINION_WORD2VEC_DIR") or (opinion_dir / "word2vec_models")
        ),
        "next_video_dir": next_video_dir,
        "opinion_dir": opinion_dir,
    }


def _resolve_settings(args: argparse.Namespace) -> dict:
    """Resolve non-path settings and toggles from CLI and environment."""
    run_next_video, run_opinion = _resolve_task_configuration(args)
    sentence_model, sentence_device, sentence_batch_size, sentence_normalize = _resolve_sentence_settings(
        args
    )
    study_tokens = tuple(
        _split_tokens(getattr(args, "studies", ""))
        or _split_tokens(os.environ.get("KNN_STUDIES", ""))
        or _split_tokens(args.issues or "")
        or _split_tokens(os.environ.get("KNN_ISSUES", ""))
    )
    return {
        "k_sweep": (
            getattr(args, "k_sweep", None)
            or os.environ.get("KNN_K_SWEEP")
            or "1,2,3,4,5,10,15,20,25,50,75,100,125,150"
        ),
        "run_next_video": run_next_video,
        "run_opinion": run_opinion,
        "study_tokens": study_tokens,
        "word2vec_epochs": int(os.environ.get("WORD2VEC_EPOCHS", "10")),
        "word2vec_workers": _default_word2vec_workers(),
        "sentence_model": sentence_model,
        "sentence_device": sentence_device,
        "sentence_batch_size": sentence_batch_size,
        "sentence_normalize": sentence_normalize,
        "feature_spaces": _resolve_feature_spaces(args),
        "reuse_sweeps": _resolve_reuse_flag(
            getattr(args, "reuse_sweeps", False), "KNN_REUSE_SWEEPS"
        ),
        "reuse_final": _resolve_reuse_flag(
            getattr(args, "reuse_final", False), "KNN_REUSE_FINAL"
        ),
        "jobs": _resolve_jobs(args),
        "allow_incomplete": _resolve_allow_incomplete(args),
    }


def _build_context_from(paths: dict, settings: dict) -> PipelineContext:
    """Construct a PipelineContext from resolved paths and settings dictionaries."""
    return PipelineContext(
        dataset=paths["dataset"],
        out_dir=paths["out_dir"],
        cache_dir=str(paths["cache_dir"]),
        sweep_dir=paths["sweep_dir"],
        word2vec_model_dir=paths["word2vec_model_dir"],
        next_video_dir=paths["next_video_dir"],
        opinion_dir=paths["opinion_dir"],
        opinion_sweep_dir=paths["opinion_sweep_dir"],
        opinion_word2vec_dir=paths["opinion_word2vec_dir"],
        k_sweep=settings["k_sweep"],
        study_tokens=settings["study_tokens"],
        word2vec_epochs=settings["word2vec_epochs"],
        word2vec_workers=settings["word2vec_workers"],
        sentence_model=settings["sentence_model"],
        sentence_device=settings["sentence_device"],
        sentence_batch_size=settings["sentence_batch_size"],
        sentence_normalize=settings["sentence_normalize"],
        feature_spaces=settings["feature_spaces"],
        jobs=settings["jobs"],
        reuse_sweeps=settings["reuse_sweeps"],
        reuse_final=settings["reuse_final"],
        allow_incomplete=settings["allow_incomplete"],
        run_next_video=settings["run_next_video"],
        run_opinion=settings["run_opinion"],
    )

def build_base_cli(context: PipelineContext, extra_cli: Sequence[str] | None = None) -> List[str]:
    """
    Construct the CLI argument prefix reused across KNN pipeline invocations.

    :param context: Pipeline configuration describing dataset paths and sweep options.
    :type context: ~knn.pipeline.context.PipelineContext
    :returns: Argument list containing dataset/cache flags plus other shared options.
    :rtype: List[str]
    """
    base_cli: List[str] = [
        "--dataset",
        context.dataset,
        "--cache-dir",
        context.cache_dir,
    ]
    extra_cli = tuple(extra_cli or ())

    def _has_flag(flag: str) -> bool:
        flag_aliases = {flag}
        if "_" in flag:
            flag_aliases.add(flag.replace("_", "-"))
        else:
            flag_aliases.add(flag.replace("-", "_"))
        return any(
            token in flag_aliases
            or any(token.startswith(alias + "=") for alias in flag_aliases)
            for token in extra_cli
        )

    if context.run_next_video:
        suppress_fit = any(
            _has_flag(flag)
            for flag in (
                "--fit-index",
                "--fit_index",
                "--load-index",
                "--load_index",
                "--no-fit-index",
                "--no_fit_index",
            )
        )
        if not suppress_fit:
            base_cli.extend(["--fit-index"])
    base_cli.extend(["--overwrite"])
    if context.k_sweep:
        base_cli.extend(["--knn-k-sweep", context.k_sweep])
    return base_cli

def log_run_configuration(studies: Sequence["StudySpec"], context: PipelineContext) -> None:
    """
    Emit a concise summary of the resolved pipeline configuration.

    :param studies: Collection of study specifications that will be evaluated.
    :type studies: Sequence[~knn.pipeline.context.StudySpec]
    :param context: Pipeline context describing dataset paths, jobs, and task toggles.
    :type context: ~knn.pipeline.context.PipelineContext
    :returns: ``None``. The function writes human-readable configuration details to the logger.
    :rtype: None
    """
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
    """
    Log the number of sweep configurations that would have executed.

    :param configs: Planned sweep configuration objects assembled during a dry run.
    :type configs: Sequence[~knn.pipeline.context.SweepConfig]
    :returns: ``None``. The function reports the count via :mod:`logging`.
    :rtype: None
    """
    LOGGER.info("[DRY RUN] Planned %d sweep configurations.", len(configs))

def _split_tokens(raw: str | None) -> List[str]:
    """
    Split a comma-separated string into trimmed tokens, discarding empties.

    :param raw: Raw comma-separated value string or ``None``.
    :type raw: str | None
    :returns: List of individual tokens with whitespace removed.
    :rtype: List[str]
    """
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]

def _default_word2vec_workers() -> int:
    """
    Determine the Word2Vec worker count from environment hints and CPU availability.

    :returns: Positive worker count bounded by ``MAX_WORD2VEC_WORKERS`` and CPU cores.
    :rtype: int
    """
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
