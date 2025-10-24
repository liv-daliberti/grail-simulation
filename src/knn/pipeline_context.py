"""Shared data classes for the modular KNN pipeline."""
# pylint: disable=line-too-long
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple

from common.pipeline_types import (
    OpinionStudySelection as BaseOpinionStudySelection,
    StudySelection as BaseStudySelection,
    StudySpec,
)

@dataclass(frozen=True)
class SweepConfig:  # pylint: disable=too-many-instance-attributes
    """
    Describe a single hyper-parameter configuration scheduled for execution.

    :ivar feature_space: Feature space identifier (``tfidf``, ``word2vec``, or ``sentence_transformer``).
    :vartype feature_space: str
    :ivar metric: Distance metric passed to the KNN scorer (``l2`` or ``cosine``).
    :vartype metric: str
    :ivar text_fields: Additional text columns merged into the viewer prompt.
    :vartype text_fields: Tuple[str, ...]
    :ivar word2vec_size: Word2Vec embedding dimensionality when ``feature_space`` is ``word2vec``.
    :vartype word2vec_size: int | None
    :ivar word2vec_window: Word2Vec context window size for training.
    :vartype word2vec_window: int | None
    :ivar word2vec_min_count: Minimum token frequency retained in the Word2Vec vocabulary.
    :vartype word2vec_min_count: int | None
    :ivar word2vec_epochs: Number of epochs used when (re)training the Word2Vec model.
    :vartype word2vec_epochs: int | None
    :ivar word2vec_workers: Worker count for Word2Vec training/encoding.
    :vartype word2vec_workers: int | None
    :ivar sentence_transformer_model: SentenceTransformer model identifier evaluated for this sweep.
    :vartype sentence_transformer_model: str | None
    :ivar sentence_transformer_device: Device override applied when encoding sentence embeddings.
    :vartype sentence_transformer_device: str | None
    :ivar sentence_transformer_batch_size: Batch size used when generating sentence embeddings.
    :vartype sentence_transformer_batch_size: int | None
    :ivar sentence_transformer_normalize: Whether embeddings are L2-normalised prior to similarity scoring.
    :vartype sentence_transformer_normalize: bool | None
    """
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
        """
        Create a filesystem-friendly identifier summarising the configuration.

        :returns: Underscore-delimited label that highlights metric, text fields, and model specifics.
        :rtype: str
        """
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
        """
        Translate the configuration into CLI arguments understood by :mod:`knn.cli`.

        :param word2vec_model_dir: Directory housing cached Word2Vec models to reuse during sweeps.
        :type word2vec_model_dir: Path | None
        :returns: Argument vector providing the minimal overrides required to reproduce the configuration.
        :rtype: list[str]
        :raises ValueError: If mandatory Word2Vec parameters are omitted for a Word2Vec sweep.
        """
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
class SweepOutcome:  # pylint: disable=too-many-instance-attributes
    """
    Persisted metrics for evaluating a configuration against a single study.

    :ivar order_index: Stable position reflecting the original sweep submission order.
    :vartype order_index: int
    :ivar study: Study metadata describing the evaluated dataset slice.
    :vartype study: StudySpec
    :ivar feature_space: Feature space evaluated by the sweep run.
    :vartype feature_space: str
    :ivar config: Hyper-parameter configuration that produced the metrics.
    :vartype config: SweepConfig
    :ivar accuracy: Held-out accuracy achieved on the validation split.
    :vartype accuracy: float
    :ivar best_k: Optimal neighbour count determined for the study.
    :vartype best_k: int
    :ivar eligible: Number of evaluation rows contributing to the metrics.
    :vartype eligible: int
    :ivar metrics_path: Filesystem path to the JSON metrics artefact.
    :vartype metrics_path: Path
    :ivar metrics: Raw metrics dictionary loaded from :attr:`metrics_path`.
    :vartype metrics: Mapping[str, object]
    """
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
class SweepTask:  # pylint: disable=too-many-instance-attributes
    """
    Describe an executable sweep job with resolved CLI arguments and paths.

    :ivar index: Stable ordinal used to preserve scheduling order.
    :vartype index: int
    :ivar study: Study specification that the sweep job evaluates.
    :vartype study: StudySpec
    :ivar config: Hyper-parameter configuration to realise in the job.
    :vartype config: SweepConfig
    :ivar base_cli: Shared CLI arguments applied to every sweep invocation.
    :vartype base_cli: Tuple[str, ...]
    :ivar extra_cli: User-specified CLI arguments appended to the invocation.
    :vartype extra_cli: Tuple[str, ...]
    :ivar run_root: Directory where intermediate sweep artefacts are written.
    :vartype run_root: Path
    :ivar word2vec_model_dir: Optional directory providing cached Word2Vec models.
    :vartype word2vec_model_dir: Path | None
    :ivar issue: Human-readable issue label aligned with the study.
    :vartype issue: str
    :ivar issue_slug: Normalised slug used for filesystem naming.
    :vartype issue_slug: str
    :ivar metrics_path: Expected location of the metrics JSON produced by the run.
    :vartype metrics_path: Path
    """
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
class StudySelection(BaseStudySelection[SweepOutcome]):  # pylint: disable=too-many-instance-attributes
    """
    Selected configuration for a specific study within a feature space.

    :ivar study: Study metadata chosen for final evaluation.
    :vartype study: StudySpec
    :ivar outcome: Winning sweep outcome promoted for the study.
    :vartype outcome: SweepOutcome
    """
    @property
    def accuracy(self) -> float:
        """
        Return the held-out accuracy achieved by the selection.

        :returns: the held-out accuracy achieved by the selection

        :rtype: float

        """
        return self.outcome.accuracy

    @property
    def best_k(self) -> int:
        """
        Return the optimal ``k`` discovered during sweeps.

        :returns: the optimal ``k`` discovered during sweeps

        :rtype: int

        """
        return self.outcome.best_k

@dataclass
class OpinionSweepOutcome:  # pylint: disable=too-many-instance-attributes
    """
    Metrics captured for a configuration/study pair during opinion sweeps.

    :ivar order_index: Stable order matching how sweep jobs were enqueued.
    :vartype order_index: int
    :ivar study: Opinion study associated with the outcome.
    :vartype study: StudySpec
    :ivar config: Hyper-parameter configuration that produced the outcome.
    :vartype config: SweepConfig
    :ivar feature_space: Feature space evaluated by the opinion sweep.
    :vartype feature_space: str
    :ivar mae: Mean absolute error achieved on the validation split.
    :vartype mae: float
    :ivar rmse: Root-mean-square error achieved on the validation split.
    :vartype rmse: float
    :ivar r2: Coefficient of determination for the opinion regression.
    :vartype r2: float
    :ivar best_k: Optimal neighbour count determined for the study.
    :vartype best_k: int
    :ivar participants: Number of participants contributing to the metrics.
    :vartype participants: int
    :ivar metrics_path: Filesystem path to the metrics JSON artefact.
    :vartype metrics_path: Path
    :ivar metrics: Raw metrics payload loaded from :attr:`metrics_path`.
    :vartype metrics: Mapping[str, object]
    """
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
class SweepTaskContext:
    """
    Shared CLI/runtime parameters required to materialise sweep tasks.

    :ivar base_cli: Baseline CLI arguments applied to each sweep invocation.
    :vartype base_cli: Sequence[str]
    :ivar extra_cli: Additional CLI arguments appended to the baseline invocation.
    :vartype extra_cli: Sequence[str]
    :ivar sweep_dir: Root directory where sweep artefacts are written.
    :vartype sweep_dir: Path
    :ivar word2vec_model_base: Directory storing cached Word2Vec artefacts.
    :vartype word2vec_model_base: Path
    """

    base_cli: Sequence[str]
    extra_cli: Sequence[str]
    sweep_dir: Path
    word2vec_model_base: Path


@dataclass(frozen=True)
class EvaluationContext:
    """Shared CLI/runtime parameters for final evaluation stages."""

    base_cli: Sequence[str]
    extra_cli: Sequence[str]
    out_dir: Path
    word2vec_model_dir: Path
    reuse_existing: bool

@dataclass(frozen=True)
class OpinionSweepTask:  # pylint: disable=too-many-instance-attributes
    """
    Describe an opinion-sweep job paired with its execution context.

    :ivar index: Stable ordinal matching the submission order.
    :vartype index: int
    :ivar study: Opinion study that the sweep job targets.
    :vartype study: StudySpec
    :ivar config: Hyper-parameter configuration under evaluation.
    :vartype config: SweepConfig
    :ivar base_cli: Baseline CLI arguments reused across tasks.
    :vartype base_cli: Tuple[str, ...]
    :ivar extra_cli: Additional passthrough CLI arguments for the job.
    :vartype extra_cli: Tuple[str, ...]
    :ivar run_root: Directory where opinion sweep outputs are written.
    :vartype run_root: Path
    :ivar word2vec_model_dir: Optional directory providing cached Word2Vec models.
    :vartype word2vec_model_dir: Path | None
    :ivar metrics_path: Expected location of the metrics JSON produced by the run.
    :vartype metrics_path: Path
    """
    index: int
    study: StudySpec
    config: SweepConfig
    base_cli: Tuple[str, ...]
    extra_cli: Tuple[str, ...]
    run_root: Path
    word2vec_model_dir: Path | None
    metrics_path: Path

# ``typing`` evaluates generic subscripts at runtime, so avoid instantiating the
# generic base while the module loads to prevent ``TypeError`` when Python
# validates the number of parameters. Static analyzers still see the narrowed
# generic thanks to the TYPE_CHECKING branch.
if TYPE_CHECKING:
    OpinionSelectionBase = BaseOpinionStudySelection[OpinionSweepOutcome]
else:
    OpinionSelectionBase = BaseOpinionStudySelection

class OpinionStudySelection(OpinionSelectionBase):  # pylint: disable=too-many-instance-attributes
    """
    Selected configuration for the final opinion evaluation.

    :ivar study: Opinion study metadata chosen for final evaluation.
    :vartype study: StudySpec
    :ivar outcome: Winning opinion sweep outcome promoted for the study.
    :vartype outcome: OpinionSweepOutcome
    """
    @property
    def best_k(self) -> int:
        """
        Return the selected ``k`` for the study.

        :returns: the selected ``k`` for the study

        :rtype: int

        """
        return self.outcome.best_k

@dataclass(frozen=True)
class PipelineContext:  # pylint: disable=too-many-instance-attributes
    """
    Normalised configuration for a pipeline run.

    :ivar dataset: Dataset path or HuggingFace identifier used for all workloads.
    :vartype dataset: str
    :ivar out_dir: Output directory where sweeps, reports, and metrics are written.
    :vartype out_dir: Path
    :ivar cache_dir: Hugging Face datasets cache directory.
    :vartype cache_dir: str
    :ivar sweep_dir: Directory that stores hyper-parameter sweep outputs.
    :vartype sweep_dir: Path
    :ivar word2vec_model_dir: Location used to persist or read Word2Vec models.
    :vartype word2vec_model_dir: Path
    :ivar k_sweep: Comma-separated list of ``k`` values evaluated during sweeps.
    :vartype k_sweep: str
    :ivar study_tokens: Study identifiers supplied via CLI or environment overrides.
    :vartype study_tokens: Tuple[str, ...]
    :ivar word2vec_epochs: Number of epochs to use when training Word2Vec embeddings.
    :vartype word2vec_epochs: int
    :ivar word2vec_workers: Number of parallel workers for Word2Vec processing.
    :vartype word2vec_workers: int
    :ivar sentence_model: SentenceTransformer model identifier to encode viewer prompts.
    :vartype sentence_model: str
    :ivar sentence_device: Device hint (``cpu``/``cuda``) for SentenceTransformer, if provided.
    :vartype sentence_device: str | None
    :ivar sentence_batch_size: Batch size used during SentenceTransformer encoding.
    :vartype sentence_batch_size: int
    :ivar sentence_normalize: Flag indicating whether embeddings are L2-normalised.
    :vartype sentence_normalize: bool
    :ivar feature_spaces: Feature spaces that should be evaluated during the run.
    :vartype feature_spaces: Tuple[str, ...]
    :ivar jobs: Level of parallelism when scheduling sweep or evaluation tasks.
    :vartype jobs: int
    :ivar reuse_sweeps: Whether cached sweep artefacts can be reused instead of re-running.
    :vartype reuse_sweeps: bool
    :ivar reuse_final: Whether cached final evaluation artefacts can be reused.
    :vartype reuse_final: bool
    :ivar allow_incomplete: Permit finalize/report stages to run with partial sweep coverage.
    :vartype allow_incomplete: bool
    :ivar run_next_video: Toggle controlling whether slate evaluation is executed.
    :vartype run_next_video: bool
    :ivar run_opinion: Toggle controlling whether opinion evaluation is executed.
    :vartype run_opinion: bool
    """
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
class ReportBundle:  # pylint: disable=too-many-instance-attributes
    """
    Aggregated artefacts required to render Markdown reports for the pipeline run.

    :ivar selections: Winning slate selections keyed by feature space and study slug.
    :vartype selections: Mapping[str, Mapping[str, StudySelection]]
    :ivar sweep_outcomes: Chronological list of all slate sweep outcomes.
    :vartype sweep_outcomes: Sequence[SweepOutcome]
    :ivar opinion_selections: Winning opinion selections keyed by feature space and study slug.
    :vartype opinion_selections: Mapping[str, Mapping[str, OpinionStudySelection]]
    :ivar opinion_sweep_outcomes: Chronological list of all opinion sweep outcomes.
    :vartype opinion_sweep_outcomes: Sequence[OpinionSweepOutcome]
    :ivar studies: Study descriptors used when rendering friendly labels.
    :vartype studies: Sequence[StudySpec]
    :ivar metrics_by_feature: Cached final slate metrics grouped by feature space and study.
    :vartype metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    :ivar opinion_metrics: Cached final opinion metrics grouped by feature space and study.
    :vartype opinion_metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :ivar k_sweep: Textual representation of the ``k`` sweep grid.
    :vartype k_sweep: str
    :ivar loso_metrics: Optional leave-one-study-out metrics aggregated by feature/study.
    :vartype loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]] | None
    :ivar feature_spaces: Ordered set of feature spaces included in the bundle.
    :vartype feature_spaces: Tuple[str, ...]
    :ivar sentence_model: SentenceTransformer model name when reports include that feature space.
    :vartype sentence_model: Optional[str]
    :ivar allow_incomplete: Whether missing sweeps are tolerated when rendering summaries.
    :vartype allow_incomplete: bool
    :ivar include_next_video: Flag indicating whether slate sections should be generated.
    :vartype include_next_video: bool
    :ivar include_opinion: Flag indicating whether opinion sections should be generated.
    :vartype include_opinion: bool
    """
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
class MetricSummary:  # pylint: disable=too-many-instance-attributes
    """
    Normalised slice of slate evaluation metrics used across reports.

    :ivar accuracy: Validation accuracy for the selected configuration.
    :vartype accuracy: Optional[float]
    :ivar accuracy_ci: 95% confidence interval for :attr:`accuracy`.
    :vartype accuracy_ci: Optional[Tuple[float, float]]
    :ivar baseline: Baseline accuracy from the most-frequent-gold comparator.
    :vartype baseline: Optional[float]
    :ivar baseline_ci: 95% confidence interval for :attr:`baseline`.
    :vartype baseline_ci: Optional[Tuple[float, float]]
    :ivar random_baseline: Expected accuracy for a random slate selection baseline.
    :vartype random_baseline: Optional[float]
    :ivar best_k: ``k`` value delivering the best validation accuracy.
    :vartype best_k: Optional[int]
    :ivar n_total: Total number of evaluation rows considered.
    :vartype n_total: Optional[int]
    :ivar n_eligible: Number of rows eligible for the final metric.
    :vartype n_eligible: Optional[int]
    """
    accuracy: Optional[float] = None
    accuracy_ci: Optional[Tuple[float, float]] = None
    baseline: Optional[float] = None
    baseline_ci: Optional[Tuple[float, float]] = None
    random_baseline: Optional[float] = None
    best_k: Optional[int] = None
    n_total: Optional[int] = None
    n_eligible: Optional[int] = None

@dataclass(frozen=True)
class OpinionSummary:  # pylint: disable=too-many-instance-attributes
    """
    Normalised view of opinion-regression metrics.

    :ivar mae: Mean absolute error for the selected configuration.
    :vartype mae: Optional[float]
    :ivar rmse: Root-mean-square error for the selected configuration.
    :vartype rmse: Optional[float]
    :ivar r2: Coefficient of determination capturing explained variance.
    :vartype r2: Optional[float]
    :ivar mae_change: Normalised change in MAE relative to the baseline.
    :vartype mae_change: Optional[float]
    :ivar baseline_mae: Baseline MAE measured using pre-study opinions.
    :vartype baseline_mae: Optional[float]
    :ivar mae_delta: Absolute delta between :attr:`mae` and :attr:`baseline_mae`.
    :vartype mae_delta: Optional[float]
    :ivar best_k: Neighbourhood size delivering the final metrics.
    :vartype best_k: Optional[int]
    :ivar participants: Number of participants included in the evaluation split.
    :vartype participants: Optional[int]
    :ivar dataset: Name of the dataset used to compute the metrics.
    :vartype dataset: Optional[str]
    :ivar split: Dataset split powering the evaluation (e.g. ``train``, ``validation``).
    :vartype split: Optional[str]
    """
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
    "SweepTaskContext",
    "EvaluationContext",
]
