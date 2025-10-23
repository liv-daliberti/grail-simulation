"""Shared pipeline dataclasses reused across baseline implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .pipeline_models import StudySpec


OutcomeT = TypeVar("OutcomeT")


@dataclass
class StudySelection(Generic[OutcomeT]):
    """Container describing the selected outcome for a study."""

    study: StudySpec
    outcome: OutcomeT

    @property
    def config(self):  # type: ignore[override]
        return self.outcome.config

    @property
    def evaluation_slug(self) -> str:
        return self.study.evaluation_slug


@dataclass
class OpinionStudySelection(Generic[OutcomeT]):
    """Container describing the selected opinion outcome for a study."""

    study: StudySpec
    outcome: OutcomeT

    @property
    def config(self):  # type: ignore[override]
        return self.outcome.config


__all__ = ["OpinionStudySelection", "StudySelection", "StudySpec"]
