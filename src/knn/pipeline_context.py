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

"""Data structures describing the Grail Simulation KNN pipeline state.

This module defines the sweep configurations, task descriptors, execution
contexts, and report bundles shared by the pipeline orchestration code.
"""

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

    :param feature_space: Feature space identifier (``tfidf``, ``word2vec``,
        or ``sentence_transformer``).
    :type feature_space: str
    :param metric: Distance metric passed to the KNN scorer (``l2`` or ``cosine``).
    :type metric: str
    :param text_fields: Additional text columns merged into the viewer prompt.
    :type text_fields: Tuple[str, ...]
    :param word2vec_size: Word2Vec embedding dimensionality when the feature space is ``word2vec``.
    :type word2vec_size: int | None
    :param word2vec_window: Word2Vec context window size for training.
    :type word2vec_window: int | None
    :param word2vec_min_count: Minimum token frequency retained in the Word2Vec vocabulary.
    :type word2vec_min_count: int | None
    :param word2vec_epochs: Number of epochs used when (re)training the Word2Vec model.
    :type word2vec_epochs: int | None
    :param word2vec_workers: Worker count for Word2Vec training/encoding.
    :type word2vec_workers: int | None
    :param sentence_transformer_model: SentenceTransformer model identifier evaluated for this sweep.
    :type sentence_transformer_model: str | None
    :param sentence_transformer_device: Device override applied when encoding sentence embeddings.
    :type sentence_transformer_device: str | None
    :param sentence_transformer_batch_size: Batch size used when generating sentence embeddings.
    :type sentence_transformer_batch_size: int | None
    :param sentence_transformer_normalize: Whether embeddings are L2-normalised prior to similarity scoring.
    :type sentence_transformer_normalize: bool | None
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

        :param self: Configuration instance being labelled.
        :type self: SweepConfig
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

    :param order_index: Stable position reflecting the original sweep submission order.
    :type order_index: int
    :param study: Study metadata describing the evaluated dataset slice.
    :type study: StudySpec
    :param feature_space: Feature space evaluated by the sweep run.
    :type feature_space: str
    :param config: Hyper-parameter configuration that produced the metrics.
    :type config: SweepConfig
    :param accuracy: Held-out accuracy achieved on the validation split.
    :type accuracy: float
    :param best_k: Optimal neighbour count determined for the study.
    :type best_k: int
    :param eligible: Number of evaluation rows contributing to the metrics.
    :type eligible: int
    :param metrics_path: Filesystem path to the JSON metrics artefact.
    :type metrics_path: Path
    :param metrics: Raw metrics dictionary loaded from :attr:`metrics_path`.
    :type metrics: Mapping[str, object]
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

    :param index: Stable ordinal used to preserve scheduling order.
    :type index: int
    :param study: Study specification that the sweep job evaluates.
    :type study: StudySpec
    :param config: Hyper-parameter configuration to realise in the job.
    :type config: SweepConfig
    :param base_cli: Shared CLI arguments applied to every sweep invocation.
    :type base_cli: Tuple[str, ...]
    :param extra_cli: User-specified CLI arguments appended to the invocation.
    :type extra_cli: Tuple[str, ...]
    :param run_root: Directory where intermediate sweep artefacts are written.
    :type run_root: Path
    :param word2vec_model_dir: Optional directory providing cached Word2Vec models.
    :type word2vec_model_dir: Path | None
    :param issue: Human-readable issue label aligned with the study.
    :type issue: str
    :param issue_slug: Normalised slug used for filesystem naming.
    :type issue_slug: str
    :param metrics_path: Expected location of the metrics JSON produced by the run.
    :type metrics_path: Path
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

    :param study: Study metadata chosen for final evaluation.
    :type study: StudySpec
    :param outcome: Winning sweep outcome promoted for the study.
    :type outcome: SweepOutcome
    """
    @property
    def accuracy(self) -> float:
        """
        Return the held-out accuracy achieved by the selection.

        :param self: Selection exposing the chosen sweep outcome.
        :type self: StudySelection
        :returns: the held-out accuracy achieved by the selection

        :rtype: float

        """
        return self.outcome.accuracy

    @property
    def best_k(self) -> int:
        """
        Return the optimal ``k`` discovered during sweeps.

        :param self: Selection exposing the chosen sweep outcome.
        :type self: StudySelection
        :returns: the optimal ``k`` discovered during sweeps

        :rtype: int

        """
        return self.outcome.best_k

@dataclass
class OpinionSweepOutcome:  # pylint: disable=too-many-instance-attributes
    """
    Metrics captured for a configuration/study pair during opinion sweeps.

    :param order_index: Stable order matching how sweep jobs were enqueued.
    :type order_index: int
    :param study: Opinion study associated with the outcome.
    :type study: StudySpec
    :param config: Hyper-parameter configuration that produced the outcome.
    :type config: SweepConfig
    :param feature_space: Feature space evaluated by the opinion sweep.
    :type feature_space: str
    :param mae: Mean absolute error achieved on the validation split.
    :type mae: float
    :param rmse: Root-mean-square error achieved on the validation split.
    :type rmse: float
    :param r2_score: Coefficient of determination for the opinion regression.
    :type r2_score: float
    :param baseline_mae: Baseline MAE measured using before-study opinions (if any).
    :type baseline_mae: Optional[float]
    :param mae_delta: Absolute delta between :attr:`mae` and :attr:`baseline_mae`.
    :type mae_delta: Optional[float]
    :param accuracy: Directional accuracy achieved by the configuration.
    :type accuracy: Optional[float]
    :param baseline_accuracy: Directional accuracy achieved by the baseline.
    :type baseline_accuracy: Optional[float]
    :param accuracy_delta: Improvement in accuracy over the baseline.
    :type accuracy_delta: Optional[float]
    :param best_k: Optimal neighbour count determined for the study.
    :type best_k: int
    :param participants: Number of participants contributing to the metrics.
    :type participants: int
    :param eligible: Count of evaluation examples used for accuracy metrics.
    :type eligible: Optional[int]
    :param metrics_path: Filesystem path to the metrics JSON artefact.
    :type metrics_path: Path
    :param metrics: Raw metrics payload loaded from :attr:`metrics_path`.
    :type metrics: Mapping[str, object]
    """
    order_index: int
    study: StudySpec
    config: SweepConfig
    feature_space: str
    mae: float
    rmse: float
    r2_score: float
    baseline_mae: Optional[float]
    mae_delta: Optional[float]
    accuracy: Optional[float]
    baseline_accuracy: Optional[float]
    accuracy_delta: Optional[float]
    best_k: int
    participants: int
    eligible: Optional[int]
    metrics_path: Path
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class SweepTaskContext:
    """
    Shared CLI/runtime parameters required to materialise sweep tasks.

    :param base_cli: Baseline CLI arguments applied to each sweep invocation.
    :type base_cli: Sequence[str]
    :param extra_cli: Additional CLI arguments appended to the baseline invocation.
    :type extra_cli: Sequence[str]
    :param sweep_dir: Root directory where sweep artefacts are written.
    :type sweep_dir: Path
    :param word2vec_model_base: Directory storing cached Word2Vec artefacts.
    :type word2vec_model_base: Path
    """

    base_cli: Sequence[str]
    extra_cli: Sequence[str]
    sweep_dir: Path
    word2vec_model_base: Path


@dataclass(frozen=True)
class EvaluationOutputs:
    """
    Normalised directory layout for next-video and opinion evaluation artefacts.

    :param next_video: Directory where next-video artefacts are written.
    :type next_video: Path
    :param opinion: Directory where opinion artefacts are written.
    :type opinion: Path
    :param shared: Shared directory preserved for backwards compatibility.
    :type shared: Path
    """

    next_video: Path
    opinion: Path
    shared: Path

    @classmethod
    def from_keywords(
        cls,
        *,
        out_dir: Path | None,
        next_video_out_dir: Path | None,
        opinion_out_dir: Path | None,
    ) -> "EvaluationOutputs":
        """
        Materialise the directory structure while supporting legacy ``out_dir`` overrides.

        :param out_dir: Legacy base directory used for both tasks.
        :type out_dir: Path | None
        :param next_video_out_dir: Task-specific next-video output directory.
        :type next_video_out_dir: Path | None
        :param opinion_out_dir: Task-specific opinion output directory.
        :type opinion_out_dir: Path | None
        :returns: ``EvaluationOutputs`` capturing the resolved directories.
        :rtype: EvaluationOutputs
        :raises TypeError: When neither legacy nor task-specific directories are supplied.
        """

        resolved_opinion = (
            opinion_out_dir
            if opinion_out_dir is not None
            else out_dir
            if out_dir is not None
            else next_video_out_dir
        )
        resolved_next = (
            next_video_out_dir
            if next_video_out_dir is not None
            else out_dir
            if out_dir is not None
            else resolved_opinion
        )
        if resolved_opinion is None or resolved_next is None:
            raise TypeError(
                "EvaluationContext requires out_dir, or explicit next/opinion output directories."
            )
        resolved_shared = out_dir if out_dir is not None else resolved_opinion
        return cls(
            next_video=resolved_next,
            opinion=resolved_opinion,
            shared=resolved_shared,
        )


@dataclass(frozen=True)
class EvaluationWord2VecPaths:
    """
    Normalised Word2Vec cache layout for next-video and opinion runs.

    :param next_video: Directory housing Word2Vec caches for next-video runs.
    :type next_video: Path
    :param opinion: Directory housing Word2Vec caches for opinion runs.
    :type opinion: Path
    :param shared: Shared Word2Vec cache directory retained for backwards compatibility.
    :type shared: Path
    """

    next_video: Path
    opinion: Path
    shared: Path

    @classmethod
    def from_keywords(
        cls,
        *,
        word2vec_model_dir: Path | None,
        next_video_word2vec_dir: Path | None,
        opinion_word2vec_dir: Path | None,
        fallback_parent: Path,
    ) -> "EvaluationWord2VecPaths":
        """
        Materialise the Word2Vec cache directories while supporting legacy overrides.

        :param word2vec_model_dir: Legacy shared Word2Vec cache directory.
        :type word2vec_model_dir: Path | None
        :param next_video_word2vec_dir: Next-video specific Word2Vec cache directory.
        :type next_video_word2vec_dir: Path | None
        :param opinion_word2vec_dir: Opinion specific Word2Vec cache directory.
        :type opinion_word2vec_dir: Path | None
        :param fallback_parent: Directory used to derive a cache path when none are supplied.
        :type fallback_parent: Path
        :returns: ``EvaluationWord2VecPaths`` capturing the resolved cache directories.
        :rtype: EvaluationWord2VecPaths
        """

        resolved_shared = (
            word2vec_model_dir
            if word2vec_model_dir is not None
            else next_video_word2vec_dir
            if next_video_word2vec_dir is not None
            else opinion_word2vec_dir
            if opinion_word2vec_dir is not None
            else fallback_parent / "word2vec_models"
        )
        resolved_next = (
            next_video_word2vec_dir if next_video_word2vec_dir is not None else resolved_shared
        )
        resolved_opinion = (
            opinion_word2vec_dir if opinion_word2vec_dir is not None else resolved_shared
        )
        return cls(
            next_video=resolved_next,
            opinion=resolved_opinion,
            shared=resolved_shared,
        )


@dataclass(frozen=True)
class EvaluationContext:
    """
    Shared CLI/runtime parameters for final evaluation stages.

    :param base_cli: Baseline CLI arguments reused for every invocation.
    :type base_cli: Sequence[str]
    :param extra_cli: Additional CLI arguments appended to each invocation.
    :type extra_cli: Sequence[str]
    :param next_video_out_dir: Output directory where next-video artefacts are written.
    :type next_video_out_dir: Path
    :param opinion_out_dir: Output directory where opinion artefacts are written.
    :type opinion_out_dir: Path
    :param next_video_word2vec_dir: Location of cached Word2Vec models for next-video runs.
    :type next_video_word2vec_dir: Path
    :param opinion_word2vec_dir: Location of cached Word2Vec models for opinion runs.
    :type opinion_word2vec_dir: Path
    :param reuse_existing: Flag controlling whether cached artefacts may be reused.
    :type reuse_existing: bool
    :param out_dir: Legacy base directory used for both tasks (fallback when task-specific
        directories are omitted).
    :type out_dir: Path | None
    :param word2vec_model_dir: Legacy Word2Vec cache directory shared across tasks.
    :type word2vec_model_dir: Path | None
    """

    base_cli: Sequence[str]
    extra_cli: Sequence[str]
    reuse_existing: bool
    outputs: EvaluationOutputs
    word2vec_models: EvaluationWord2VecPaths

    @classmethod
    def from_args(
        cls,
        *,
        base_cli: Sequence[str],
        extra_cli: Sequence[str],
        reuse_existing: bool,
        **overrides: object,
    ) -> "EvaluationContext":
        """
        Build an :class:`EvaluationContext` from legacy or task-specific directory overrides.

        :returns: Result produced by ``EvaluationContext``.
        :rtype: EvaluationContext
        """

        valid_keys = {
            "out_dir",
            "word2vec_model_dir",
            "next_video_out_dir",
            "opinion_out_dir",
            "next_video_word2vec_dir",
            "opinion_word2vec_dir",
        }
        unexpected = set(overrides) - valid_keys
        if unexpected:
            formatted = ", ".join(sorted(unexpected))
            raise TypeError(f"EvaluationContext received unexpected keyword(s): {formatted}")

        outputs = EvaluationOutputs.from_keywords(
            out_dir=overrides.get("out_dir"),
            next_video_out_dir=overrides.get("next_video_out_dir"),
            opinion_out_dir=overrides.get("opinion_out_dir"),
        )
        word2vec_paths = EvaluationWord2VecPaths.from_keywords(
            word2vec_model_dir=overrides.get("word2vec_model_dir"),
            next_video_word2vec_dir=overrides.get("next_video_word2vec_dir"),
            opinion_word2vec_dir=overrides.get("opinion_word2vec_dir"),
            fallback_parent=outputs.next_video,
        )
        return cls(
            base_cli=base_cli,
            extra_cli=extra_cli,
            reuse_existing=reuse_existing,
            outputs=outputs,
            word2vec_models=word2vec_paths,
        )

    @property
    def next_video_out_dir(self) -> Path:
        """
        Directory where next-video evaluation artefacts are written.

        :rtype: Path
        """

        return self.outputs.next_video

    @property
    def opinion_out_dir(self) -> Path:
        """
        Directory where opinion evaluation artefacts are written.

        :rtype: Path
        """

        return self.outputs.opinion

    @property
    def out_dir(self) -> Path:
        """
        Legacy evaluation output directory preserved for backwards compatibility.

        :rtype: Path
        """

        return self.outputs.shared

    @property
    def next_video_word2vec_dir(self) -> Path:
        """
        Directory containing Word2Vec caches for next-video runs.

        :rtype: Path
        """

        return self.word2vec_models.next_video

    @property
    def opinion_word2vec_dir(self) -> Path:
        """
        Directory containing Word2Vec caches for opinion runs.

        :rtype: Path
        """

        return self.word2vec_models.opinion

    @property
    def word2vec_model_dir(self) -> Path:
        """
        Legacy shared Word2Vec cache directory preserved for backwards compatibility.

        :rtype: Path
        """

        return self.word2vec_models.shared

@dataclass(frozen=True)
class OpinionSweepTask:  # pylint: disable=too-many-instance-attributes
    """
    Describe an opinion-sweep job paired with its execution context.

    :param index: Stable ordinal matching the submission order.
    :type index: int
    :param study: Opinion study that the sweep job targets.
    :type study: StudySpec
    :param config: Hyper-parameter configuration under evaluation.
    :type config: SweepConfig
    :param base_cli: Baseline CLI arguments reused across tasks.
    :type base_cli: Tuple[str, ...]
    :param extra_cli: Additional passthrough CLI arguments for the job.
    :type extra_cli: Tuple[str, ...]
    :param run_root: Directory where opinion sweep outputs are written.
    :type run_root: Path
    :param word2vec_model_dir: Optional directory providing cached Word2Vec models.
    :type word2vec_model_dir: Path | None
    :param metrics_path: Expected location of the metrics JSON produced by the run.
    :type metrics_path: Path
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

    :param study: Opinion study metadata chosen for final evaluation.
    :type study: StudySpec
    :param outcome: Winning opinion sweep outcome promoted for the study.
    :type outcome: OpinionSweepOutcome
    """
    @property
    def best_k(self) -> int:
        """
        Return the selected ``k`` for the study.

        :param self: Opinion selection exposing the chosen sweep outcome.
        :type self: OpinionStudySelection
        :returns: the selected ``k`` for the study

        :rtype: int

        """
        return self.outcome.best_k

@dataclass(frozen=True)
class PipelineContext:  # pylint: disable=too-many-instance-attributes
    """
    Normalised configuration for a pipeline run.

    :param dataset: Dataset path or HuggingFace identifier used for all workloads.
    :type dataset: str
    :param out_dir: Output directory where sweeps, reports, and metrics are written.
    :type out_dir: Path
    :param cache_dir: Hugging Face datasets cache directory.
    :type cache_dir: str
    :param sweep_dir: Directory that stores next-video hyper-parameter sweep outputs.
    :type sweep_dir: Path
    :param word2vec_model_dir: Location used to persist or read Word2Vec models.
    :type word2vec_model_dir: Path
    :param next_video_dir: Root directory for next-video evaluation artefacts.
    :type next_video_dir: Path
    :param opinion_dir: Root directory for opinion evaluation artefacts.
    :type opinion_dir: Path
    :param opinion_sweep_dir: Directory storing opinion hyper-parameter sweep outputs.
    :type opinion_sweep_dir: Path
    :param opinion_word2vec_dir: Location used to persist or read opinion Word2Vec models.
    :type opinion_word2vec_dir: Path
    :param k_sweep: Comma-separated list of ``k`` values evaluated during sweeps.
    :type k_sweep: str
    :param study_tokens: Study identifiers supplied via CLI or environment overrides.
    :type study_tokens: Tuple[str, ...]
    :param word2vec_epochs: Number of epochs to use when training Word2Vec embeddings.
    :type word2vec_epochs: int
    :param word2vec_workers: Number of parallel workers for Word2Vec processing.
    :type word2vec_workers: int
    :param sentence_model: SentenceTransformer model identifier to encode viewer prompts.
    :type sentence_model: str
    :param sentence_device: Device hint (``cpu``/``cuda``) for SentenceTransformer, if provided.
    :type sentence_device: str | None
    :param sentence_batch_size: Batch size used during SentenceTransformer encoding.
    :type sentence_batch_size: int
    :param sentence_normalize: Flag indicating whether embeddings are L2-normalised.
    :type sentence_normalize: bool
    :param feature_spaces: Feature spaces that should be evaluated during the run.
    :type feature_spaces: Tuple[str, ...]
    :param jobs: Level of parallelism when scheduling sweep or evaluation tasks.
    :type jobs: int
    :param reuse_sweeps: Whether cached sweep artefacts can be reused instead of re-running.
    :type reuse_sweeps: bool
    :param reuse_final: Whether cached final evaluation artefacts can be reused.
    :type reuse_final: bool
    :param allow_incomplete: Permit finalize/report stages to run with partial sweep coverage.
    :type allow_incomplete: bool
    :param run_next_video: Toggle controlling whether slate evaluation is executed.
    :type run_next_video: bool
    :param run_opinion: Toggle controlling whether opinion evaluation is executed.
    :type run_opinion: bool
    """
    dataset: str
    out_dir: Path
    cache_dir: str
    sweep_dir: Path
    word2vec_model_dir: Path
    next_video_dir: Path
    opinion_dir: Path
    opinion_sweep_dir: Path
    opinion_word2vec_dir: Path
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

    :param selections: Winning slate selections keyed by feature space and study slug.
    :type selections: Mapping[str, Mapping[str, StudySelection]]
    :param sweep_outcomes: Chronological list of all slate sweep outcomes.
    :type sweep_outcomes: Sequence[SweepOutcome]
    :param opinion_selections: Winning opinion selections keyed by feature space and study slug.
    :type opinion_selections: Mapping[str, Mapping[str, OpinionStudySelection]]
    :param opinion_sweep_outcomes: Chronological list of all opinion sweep outcomes.
    :type opinion_sweep_outcomes: Sequence[OpinionSweepOutcome]
    :param studies: Study descriptors used when rendering friendly labels.
    :type studies: Sequence[StudySpec]
    :param metrics_by_feature: Cached final slate metrics grouped by feature space and study.
    :type metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param opinion_metrics: Cached final opinion metrics grouped by feature space and study.
    :type opinion_metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
    :param k_sweep: Textual representation of the ``k`` sweep grid.
    :type k_sweep: str
    :param loso_metrics: Optional leave-one-study-out metrics aggregated by feature/study.
    :type loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]] | None
    :param feature_spaces: Ordered set of feature spaces included in the bundle.
    :type feature_spaces: Tuple[str, ...]
    :param sentence_model: SentenceTransformer model name when reports include that feature space.
    :type sentence_model: Optional[str]
    :param allow_incomplete: Whether missing sweeps are tolerated when rendering summaries.
    :type allow_incomplete: bool
    :param include_next_video: Flag indicating whether slate sections should be generated.
    :type include_next_video: bool
    :param include_opinion: Flag indicating whether opinion sections should be generated.
    :type include_opinion: bool
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

    :param accuracy: Validation accuracy for the selected configuration.
    :type accuracy: Optional[float]
    :param accuracy_ci: 95% confidence interval for :attr:`accuracy`.
    :type accuracy_ci: Optional[Tuple[float, float]]
    :param baseline: Baseline accuracy from the most-frequent-gold comparator.
    :type baseline: Optional[float]
    :param baseline_ci: 95% confidence interval for :attr:`baseline`.
    :type baseline_ci: Optional[Tuple[float, float]]
    :param random_baseline: Expected accuracy for a random slate selection baseline.
    :type random_baseline: Optional[float]
    :param best_k: ``k`` value delivering the best validation accuracy.
    :type best_k: Optional[int]
    :param n_total: Total number of evaluation rows considered.
    :type n_total: Optional[int]
    :param n_eligible: Number of rows eligible for the final metric.
    :type n_eligible: Optional[int]
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

    :param mae: Mean absolute error for the selected configuration.
    :type mae: Optional[float]
    :param rmse: Root-mean-square error for the selected configuration.
    :type rmse: Optional[float]
    :param r2_score: Coefficient of determination capturing explained variance.
    :type r2_score: Optional[float]
    :param mae_change: Normalised change in MAE relative to the baseline.
    :type mae_change: Optional[float]
    :param baseline_mae: Baseline MAE measured using pre-study opinions.
    :type baseline_mae: Optional[float]
    :param mae_delta: Absolute delta between :attr:`mae` and :attr:`baseline_mae`.
    :type mae_delta: Optional[float]
    :param accuracy: Directional accuracy comparing predicted opinion shifts.
    :type accuracy: Optional[float]
    :param baseline_accuracy: Directional accuracy achieved by the no-change baseline.
    :type baseline_accuracy: Optional[float]
    :param accuracy_delta: Improvement in directional accuracy over the baseline.
    :type accuracy_delta: Optional[float]
    :param best_k: Neighbourhood size delivering the final metrics.
    :type best_k: Optional[int]
    :param participants: Number of participants included in the evaluation split.
    :type participants: Optional[int]
    :param eligible: Count of evaluation examples used to compute accuracy metrics.
    :type eligible: Optional[int]
    :param dataset: Name of the dataset used to compute the metrics.
    :type dataset: Optional[str]
    :param split: Dataset split powering the evaluation (e.g. ``train``, ``validation``).
    :type split: Optional[str]
    """
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    mae_change: Optional[float] = None
    baseline_mae: Optional[float] = None
    mae_delta: Optional[float] = None
    accuracy: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    accuracy_delta: Optional[float] = None
    best_k: Optional[int] = None
    participants: Optional[int] = None
    eligible: Optional[int] = None
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
