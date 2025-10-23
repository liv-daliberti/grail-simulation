"""Shared dataclasses used across KNN and XGBoost pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StudySpec:
    """

    Descriptor for a participant study and its associated issue.



    :ivar key: Attribute ``key``.

    :vartype key: str

    :ivar issue: Attribute ``issue``.

    :vartype issue: str

    :ivar label: Attribute ``label``.

    :vartype label: str

    """


    key: str
    issue: str
    label: str

    @property
    def study_slug(self) -> str:
        """

        Return a filesystem-safe slug for the study key.



        :returns: Result produced by ``study_slug``.

        :rtype: str

        """


        return self.key.replace(" ", "_")

    @property
    def issue_slug(self) -> str:
        """

        Return a filesystem-safe slug for the associated issue.



        :returns: Result produced by ``issue_slug``.

        :rtype: str

        """


        return self.issue.replace(" ", "_")

    @property
    def evaluation_slug(self) -> str:
        """

        Return the slug used for evaluation artefacts.



        :returns: Result produced by ``evaluation_slug``.

        :rtype: str

        """


        return f"{self.issue_slug}_{self.study_slug}"
