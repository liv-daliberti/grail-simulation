"""Shared pipeline dataclasses reused across baseline implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .pipeline_models import StudySpec


OutcomeT = TypeVar("OutcomeT")


@dataclass
class StudySelection(Generic[OutcomeT]):
    """

    Container describing the selected outcome for a study.



    :ivar study: Attribute ``study``.

    :vartype study: StudySpec

    :ivar outcome: Attribute ``outcome``.

    :vartype outcome: OutcomeT

    """


    study: StudySpec
    outcome: OutcomeT

    @property
    def config(self):  # type: ignore[override]
        """
        Expose the configuration associated with the selected outcome.

        :returns: Configuration object produced by the underlying outcome.
        :rtype: Any
        """
        return self.outcome.config

    @property
    def evaluation_slug(self) -> str:
        """
        Return the evaluation slug derived from the selected study.

        :returns: Evaluation slug used for downstream artefact paths.
        :rtype: str
        """
        return self.study.evaluation_slug


@dataclass
class OpinionStudySelection(Generic[OutcomeT]):
    """

    Container describing the selected opinion outcome for a study.



    :ivar study: Attribute ``study``.

    :vartype study: StudySpec

    :ivar outcome: Attribute ``outcome``.

    :vartype outcome: OutcomeT

    """


    study: StudySpec
    outcome: OutcomeT

    @property
    def config(self):  # type: ignore[override]
        """
        Expose the configuration associated with the selected opinion outcome.

        :returns: Configuration object produced by the underlying opinion outcome.
        :rtype: Any
        """
        return self.outcome.config


__all__ = ["OpinionStudySelection", "StudySelection", "StudySpec"]
