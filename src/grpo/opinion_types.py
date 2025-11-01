#!/usr/bin/env python
"""Types and data models for GRPO opinion evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Sequence

from common.opinion import (
    OpinionArtifacts,
    OpinionStudyResult as _BaseOpinionStudyResult,
    OpinionSpec,
)

# Public alias used in this package
OpinionStudyFiles = OpinionArtifacts


@dataclass(frozen=True)
class OpinionDatasetSpec:
    """Dataset selection applied for opinion evaluation."""

    name: str
    split: str
    cache_dir: str | None


@dataclass(frozen=True)
class OpinionPromptSettings:
    """Prompt configuration used when preparing model inputs."""

    system: str
    opinion: str
    solution_key: str | None
    max_history: int


@dataclass(frozen=True)
class OpinionEvaluationControls:
    """Execution controls covering participant limits and caching."""

    max_participants: int
    direction_tolerance: float
    overwrite: bool
    # Periodic flush interval (participants). When >0, artefacts are persisted
    # incrementally during evaluation to improve resilience.
    flush_every: int = 0


@dataclass(frozen=True)
class OpinionEvaluationSettings:
    """Configuration for GRPO opinion evaluation."""

    dataset: OpinionDatasetSpec
    prompts: OpinionPromptSettings
    controls: OpinionEvaluationControls
    include_studies: Sequence[str] | None


@dataclass(frozen=True)
class OpinionStudySummary:
    """Bundle capturing evaluation metrics and participant counts for a study."""

    metrics: Mapping[str, object]
    baseline: Mapping[str, object]
    participants: int
    eligible: int


class OpinionStudyResult(_BaseOpinionStudyResult):
    """Per-study artefacts returned by the evaluation runner."""

    def __init__(
        self,
        *,
        study: OpinionSpec,
        files: OpinionStudyFiles,
        summary: OpinionStudySummary | None = None,
        **legacy_kwargs,
    ) -> None:
        if summary is None:
            try:
                metrics = legacy_kwargs.pop("metrics")
                baseline = legacy_kwargs.pop("baseline")
                participants = legacy_kwargs.pop("participants")
                eligible = legacy_kwargs.pop("eligible")
            except KeyError as exc:  # pragma: no cover - defensive guard
                raise TypeError(
                    "OpinionStudyResult requires either 'summary' or the legacy "
                    "metrics/baseline/participants/eligible arguments."
                ) from exc
            summary = OpinionStudySummary(
                metrics=metrics,
                baseline=baseline,
                participants=int(participants),
                eligible=int(eligible),
            )
        if legacy_kwargs:
            raise TypeError(
                f"Unexpected arguments for OpinionStudyResult: {sorted(legacy_kwargs)}"
            )
        super().__init__(
            study_key=study.key,
            study_label=study.label,
            issue=study.issue,
            participants=summary.participants,
            eligible=summary.eligible,
            metrics=summary.metrics,
            baseline=summary.baseline,
            artifacts=files,
            spec=study,
        )

    @property
    def study(self) -> OpinionSpec:
        """Return the opinion study specification."""

        spec = self.spec
        if spec is None:  # pragma: no cover - defensive guard
            raise RuntimeError("OpinionStudyResult missing spec reference")
        return spec

    @property
    def files(self) -> OpinionStudyFiles:
        """Expose the filesystem artefacts associated with the study."""

        return self.artifacts


@dataclass(frozen=True)
class OpinionEvaluationResult:
    """Aggregate payload containing per-study and combined metrics."""

    studies: Sequence[OpinionStudyResult]
    combined_metrics: Mapping[str, object]


# Backwards compatibility: retain OpinionArtifacts alias.
OpinionArtifacts = OpinionStudyFiles


@dataclass
class _StudyAccumulator:
    """Capture the rolling state for a single study evaluation."""

    predictions: List[Mapping[str, object]] = field(default_factory=list)
    qa_entries: List[str] = field(default_factory=list)
    truth_before: List[float] = field(default_factory=list)
    truth_after: List[float] = field(default_factory=list)
    pred_after: List[float] = field(default_factory=list)

    @property
    def participants(self) -> int:
        """Return the number of recorded participants."""
        return len(self.predictions)

    def record(self, artefact: "_OpinionPredictionArtefact") -> None:
        """Append a single prediction artefact into the accumulator.

        Updates predictions, QA log entries, and each truth/pred vector.
        """
        self.predictions.append(artefact.payload)
        self.qa_entries.append(artefact.qa_entry)
        self.truth_before.append(float(artefact.before))
        self.truth_after.append(float(artefact.after))
        self.pred_after.append(float(artefact.prediction))


@dataclass
class _CombinedVectors:
    """Aggregate opinion vectors across every evaluated study."""

    truth_before: List[float] = field(default_factory=list)
    truth_after: List[float] = field(default_factory=list)
    pred_after: List[float] = field(default_factory=list)

    def extend(self, accumulator: _StudyAccumulator) -> None:
        """Concatenate vectors from a per-study accumulator."""
        self.truth_before.extend(accumulator.truth_before)
        self.truth_after.extend(accumulator.truth_after)
        self.pred_after.extend(accumulator.pred_after)


@dataclass(frozen=True)
class OpinionInferenceContext:
    """Bundle the runtime inference dependencies used across studies."""

    tokenizer: object
    model: object
    generation: object


@dataclass(frozen=True)
class OpinionStudyContext:
    """Convenience bundle combining prompts, tolerance, and inference context."""

    prompts: OpinionPromptSettings
    direction_tolerance: float
    inference: OpinionInferenceContext
    overwrite: bool
    flush_every: int


@dataclass(frozen=True)
class _OpinionPredictionArtefact:
    """Artefacts generated when scoring a single participant example."""

    payload: Mapping[str, object]
    qa_entry: str
    before: float
    after: float
    prediction: float


__all__ = [
    "OpinionArtifacts",
    "OpinionDatasetSpec",
    "OpinionEvaluationControls",
    "OpinionEvaluationResult",
    "OpinionEvaluationSettings",
    "OpinionInferenceContext",
    "OpinionPromptSettings",
    "OpinionStudyContext",
    "OpinionStudyFiles",
    "OpinionStudyResult",
    "OpinionStudySummary",
]
