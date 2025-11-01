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

"""Pipeline settings, path resolution, and vectorizer configuration helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from ..core.vectorizers import (
    SentenceTransformerVectorizerConfig,
    TfidfConfig,
    Word2VecVectorizerConfig,
    build_word2vec_config_from_args,
)
from .cli import (
    _default_cache_dir,
    _default_out_dir,
    _default_reports_dir,
    _repo_root,
)
from .sweeps.common import _gpu_tree_method_supported as _gpu_supported  # type: ignore

LOGGER = logging.getLogger("xgb.pipeline")


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved filesystem locations used by the pipeline run.

    Keep instance attributes under pylint's default limit by exposing
    derived directories as properties. ``next_video_dir`` and
    ``opinion_dir`` are computed from ``out_dir`` to preserve the public
    API without inflating the attribute count.
    """

    root: Path
    out_dir: Path
    sweep_dir: Path
    opinion_sweep_dir: Path
    reports_dir: Path
    dataset: str
    cache_dir: str

    @property
    def next_video_dir(self) -> Path:
        """Directory for next-video artefacts under ``out_dir``.

        Uses a fixed subfolder name to keep consumers decoupled from the
        on-disk layout and avoid duplicating path logic across modules.
        """
        return self.out_dir / "next_video"

    @property
    def opinion_dir(self) -> Path:
        """Directory for opinion-regression artefacts under ``out_dir``.

        Centralising this keeps report generation and evaluation code consistent
        and avoids scattering hard-coded folder names.
        """
        return self.out_dir / "opinions"


@dataclass(frozen=True)
class PipelineSettings:
    """Execution flags derived from CLI and environment variables."""

    jobs: int
    tree_method: str
    allow_incomplete: bool
    reuse_sweeps: bool
    reuse_final: bool


def determine_jobs(args) -> int:
    """Resolve the parallel job count from CLI and environment."""

    jobs_value = getattr(args, "jobs", 1) or 1
    env_jobs = os.environ.get("XGB_JOBS")
    if env_jobs:
        try:
            jobs_value = int(env_jobs)
        except ValueError:
            LOGGER.warning("Ignoring invalid XGB_JOBS value '%s'.", env_jobs)
    return max(1, jobs_value)


def determine_tree_method(args) -> str:
    """Resolve the XGBoost ``tree_method`` to use for this run."""

    method = args.tree_method or "gpu_hist"
    if method == "gpu_hist" and not _gpu_supported():
        LOGGER.warning(
            "Requested tree_method=gpu_hist but the installed XGBoost build lacks GPU support. "
            "Falling back to tree_method=hist."
        )
        method = "hist"
    return method


def build_paths(args) -> PipelinePaths:
    """Compute canonical directories and dataset/cache locations for a run."""

    root = _repo_root()
    dataset = args.dataset or str(root / "data" / "cleaned_grail")
    cache_dir = args.cache_dir or str(_default_cache_dir(root))
    out_dir = Path(args.out_dir or _default_out_dir(root))
    sweep_dir = Path(args.sweep_dir or ((out_dir / "next_video") / "sweeps"))
    opinion_sweep_dir = Path(
        os.environ.get("XGB_OPINION_SWEEP_DIR") or ((out_dir / "opinions") / "sweeps")
    )
    reports_dir = Path(args.reports_dir or _default_reports_dir(root))
    return PipelinePaths(
        root=root,
        out_dir=out_dir,
        sweep_dir=sweep_dir,
        opinion_sweep_dir=opinion_sweep_dir,
        reports_dir=reports_dir,
        dataset=str(dataset),
        cache_dir=str(cache_dir),
    )


def resolve_reuse_flags(args) -> Tuple[bool, bool, str | None, str | None]:
    """Resolve sweep/final reuse booleans and their sources for logging."""

    reuse_sweeps = bool(getattr(args, "reuse_sweeps", False))
    reuse_sweeps_source: str | None = "--reuse-sweeps" if reuse_sweeps else None
    reuse_sweeps_env = os.environ.get("XGB_REUSE_SWEEPS")
    if reuse_sweeps_env is not None:
        reuse_sweeps = reuse_sweeps_env.lower() not in {"0", "false", "no"}
        reuse_sweeps_source = "XGB_REUSE_SWEEPS"

    reuse_final = reuse_sweeps
    reuse_final_source: str | None = "sweep reuse default" if reuse_final else None
    if args.reuse_final is not None:
        reuse_final = args.reuse_final
        reuse_final_source = "--reuse-final"
    reuse_final_env = os.environ.get("XGB_REUSE_FINAL")
    if reuse_final_env is not None:
        reuse_final = reuse_final_env.lower() not in {"0", "false", "no"}
        reuse_final_source = "XGB_REUSE_FINAL"
    return reuse_sweeps, reuse_final, reuse_sweeps_source, reuse_final_source


@dataclass(frozen=True)
class VectorizerConfigs:
    """Bundle of vectorizer configuration objects for convenience."""

    max_features_value: int | None
    tfidf_config: TfidfConfig
    word2vec_model_base: Path | None
    word2vec_config: Word2VecVectorizerConfig
    sentence_transformer_config: SentenceTransformerVectorizerConfig


def build_vectorizer_configs(args) -> VectorizerConfigs:
    """Construct vectorizer configuration objects from CLI args."""

    max_features_value = args.max_features if args.max_features > 0 else None
    tfidf_config = TfidfConfig(max_features=max_features_value)
    word2vec_model_base = (
        Path(args.word2vec_model_dir).resolve()
        if args.word2vec_model_dir
        else None
    )
    word2vec_config = build_word2vec_config_from_args(args, model_dir=None)
    sentence_transformer_config = SentenceTransformerVectorizerConfig(
        model_name=args.sentence_transformer_model,
        device=args.sentence_transformer_device or None,
        batch_size=args.sentence_transformer_batch_size,
        normalize=args.sentence_transformer_normalize,
    )
    return VectorizerConfigs(
        max_features_value=max_features_value,
        tfidf_config=tfidf_config,
        word2vec_model_base=word2vec_model_base,
        word2vec_config=word2vec_config,
        sentence_transformer_config=sentence_transformer_config,
    )


def build_base_cli(args, paths: PipelinePaths, extra_fields: Sequence[str]) -> List[str]:
    """Compose the base CLI shared by downstream training/eval stages."""

    base_cli: List[str] = [
        "--fit_model",
        "--dataset",
        paths.dataset,
        "--cache_dir",
        paths.cache_dir,
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
    return base_cli


def log_reuse_flags(
    reuse_sweeps: bool,
    reuse_final: bool,
    reuse_sweeps_source: str | None,
    reuse_final_source: str | None,
) -> None:
    """Emit informative logs summarising reuse behaviour for this run."""

    if reuse_sweeps:
        detail = f" ({reuse_sweeps_source})" if reuse_sweeps_source else ""
        LOGGER.warning(
            "Cached sweep metrics reuse enabled%s; stale artefacts will be used when present.",
            detail,
        )
    else:
        LOGGER.info("Cached sweep metrics reuse is off; sweeps will recompute results.")
    if reuse_final:
        detail = f" ({reuse_final_source})" if reuse_final_source else ""
        LOGGER.warning(
            "Finalize-stage reuse enabled%s; cached evaluation artefacts may be consumed.",
            detail,
        )


__all__ = [
    "PipelinePaths",
    "PipelineSettings",
    "determine_jobs",
    "determine_tree_method",
    "build_paths",
    "resolve_reuse_flags",
    "build_vectorizer_configs",
    "VectorizerConfigs",
    "build_base_cli",
    "log_reuse_flags",
]
