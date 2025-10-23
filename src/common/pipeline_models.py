"""Shared dataclasses used across KNN and XGBoost pipelines."""

from __future__ import annotations

from dataclasses import dataclass


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

