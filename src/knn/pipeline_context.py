"""Shared data classes for the modular KNN pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from common.pipeline_types import (
    OpinionStudySelection as BaseOpinionStudySelection,
    StudySelection as BaseStudySelection,
    StudySpec,
)


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
            cleaned = "".join(token for token in model_name if token.isalnum())
            parts.append(f"model-{cleaned or 'st'}")
        return "_".join(parts)

    def cli_args(self, *, word2vec_model_dir: Path | None) -> list[str]:
        """Return CLI overrides implementing this configuration."""

        args: list[str] = [
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
    """Capture metrics for a configuration/issue pair."""

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



@dataclass
class StudySelection(BaseStudySelection[SweepOutcome]):
    """Selected configuration for a specific study within a feature space."""

    @property
    def accuracy(self) -> float:
        """Return the held-out accuracy achieved by the selection."""

        return self.outcome.accuracy

    @property
    def best_k(self) -> int:
        """Return the optimal ``k`` discovered during sweeps."""

        return self.outcome.best_k


@dataclass
class OpinionSweepOutcome:
    """Metrics captured for a configuration/study pair during opinion sweeps."""

    order_index: int
    study: StudySpec
    config: SweepConfig
    feature_space: str
    mae: float
    rmse: float
    r2: float
    best_k: int
    participants: int
    metrics_path: Path
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class OpinionSweepTask:
    """Description of an opinion-sweep execution request."""

    index: int
    study: StudySpec
    config: SweepConfig
    base_cli: Tuple[str, ...]
    extra_cli: Tuple[str, ...]
    run_root: Path
    word2vec_model_dir: Path | None
    metrics_path: Path


class OpinionStudySelection(BaseOpinionStudySelection[OpinionSweepOutcome]):
    """Selected configuration for the final opinion evaluation."""

    @property
    def best_k(self) -> int:
        """Return the selected ``k`` for the study."""

        return self.outcome.best_k


@dataclass(frozen=True)
class PipelineContext:
    """Normalised configuration for a pipeline run."""

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
    run_next_video: bool = True
    run_opinion: bool = True


@dataclass(frozen=True)
class ReportBundle:
    """Inputs required to render the Markdown summaries."""

    selections: Mapping[str, Mapping[str, StudySelection]]
    sweep_outcomes: Sequence[SweepOutcome]
    opinion_selections: Mapping[str, Mapping[str, OpinionStudySelection]] = field(
        default_factory=dict
    )
    opinion_sweep_outcomes: Sequence[OpinionSweepOutcome] = field(default_factory=tuple)
    studies: Sequence[StudySpec] = field(default_factory=tuple)
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]] = field(
        default_factory=dict
    )
    opinion_metrics: Mapping[str, Mapping[str, Mapping[str, object]]] = field(
        default_factory=dict
    )
    k_sweep: str = ""
    loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None
    feature_spaces: Tuple[str, ...] = ("tfidf", "word2vec", "sentence_transformer")
    sentence_model: Optional[str] = None
    allow_incomplete: bool = False
    include_next_video: bool = True
    include_opinion: bool = True


@dataclass(frozen=True)
class MetricSummary:
    """Normalised slice of common slate metrics."""

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
    """Normalised view of opinion-regression metrics."""

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


__all__ = [
    "MetricSummary",
    "OpinionStudySelection",
    "OpinionSummary",
    "OpinionSweepOutcome",
    "OpinionSweepTask",
    "PipelineContext",
    "ReportBundle",
    "StudySelection",
    "StudySpec",
    "SweepConfig",
    "SweepOutcome",
    "SweepTask",
]
