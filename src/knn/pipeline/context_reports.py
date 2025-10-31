#!/usr/bin/env python
"""Report bundle structures for the KNN pipeline.

Split out from ``context.py`` to simplify that module and improve Sphinx
autodoc. These helpers aggregate selections, outcomes, and metrics used
by Markdown report generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class _ReportSelections:
    selections: Mapping[str, Mapping[str, "StudySelection"]]
    opinion_selections: Mapping[str, Mapping[str, "OpinionStudySelection"]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _ReportOutcomes:
    sweep_outcomes: Sequence["SweepOutcome"] = field(default_factory=tuple)
    opinion_sweep_outcomes: Sequence["OpinionSweepOutcome"] = field(default_factory=tuple)


@dataclass(frozen=True)
class _ReportMetrics:
    metrics_by_feature: Mapping[str, Mapping[str, Mapping[str, object]]] = field(
        default_factory=dict
    )
    opinion_metrics: Mapping[str, Mapping[str, Mapping[str, object]]] = field(
        default_factory=dict
    )
    opinion_from_next_metrics: Mapping[str, Mapping[str, Mapping[str, object]]] = field(
        default_factory=dict
    )
    loso_metrics: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None


@dataclass(frozen=True)
class _PresentationFlags:
    """Presentation toggles grouped to keep attribute counts low."""

    allow_incomplete: bool = False
    include_next_video: bool = True
    include_opinion: bool = True
    include_opinion_from_next: bool = False


@dataclass(frozen=True)
class _PredictionRoots:
    """Grouped locations for prediction artefacts used by report generators."""

    opinion_predictions_root: Path | None = None
    opinion_from_next_predictions_root: Path | None = None


@dataclass(frozen=True)
class _ReportPresentation:
    feature_spaces: Tuple[str, ...] = ("tfidf", "word2vec", "sentence_transformer")
    sentence_model: Optional[str] = None
    k_sweep: str = ""
    studies: Sequence["StudySpec"] = field(default_factory=tuple)
    flags: _PresentationFlags = field(default_factory=_PresentationFlags)
    predictions: _PredictionRoots = field(default_factory=_PredictionRoots)


class ReportBundle:
    """
    Aggregated artefacts required to render Markdown reports for the pipeline run.

    :param selections: Winning selections for slate and opinion tasks.
    :type selections: _ReportSelections
    :param outcomes: Chronological lists of executed/cached sweep outcomes.
    :type outcomes: _ReportOutcomes
    :param metrics: Final and LOSO metrics grouped by feature and study.
    :type metrics: _ReportMetrics
    :param presentation: Report presentation options and auxiliary paths.
    :type presentation: _ReportPresentation
    """

    def __init__(
        self,
        *,
        selections: _ReportSelections,
        outcomes: _ReportOutcomes,
        metrics: _ReportMetrics,
        presentation: _ReportPresentation,
    ) -> None:
        self._selections = selections
        self._outcomes = outcomes
        self._metrics = metrics
        self._presentation = presentation

    # Backwards-compatible read-only properties
    @property
    def selections(self) -> Mapping[str, Mapping[str, "StudySelection"]]:  # pragma: no cover
        """Winning slate selections keyed by feature and study.

        :returns: Nested mapping of selections per feature space and study.
        :rtype: Mapping[str, Mapping[str, ~knn.pipeline.context.StudySelection]]
        """
        return self._selections.selections

    @property
    def sweep_outcomes(self) -> Sequence["SweepOutcome"]:  # pragma: no cover
        """Chronological list of all slate sweep outcomes.
        
        :returns: Slate outcomes in chronological order.
        :rtype: Sequence[~knn.pipeline.context.SweepOutcome]
        """
        return self._outcomes.sweep_outcomes

    @property
    def opinion_selections(self):  # pragma: no cover
        """Mapping from feature space to per-study opinion selections.

        :returns: Nested mapping of opinion selections per feature and study.
        :rtype: Mapping[str, Mapping[str, ~knn.pipeline.context.OpinionStudySelection]]
        """
        return self._selections.opinion_selections

    @property
    def opinion_sweep_outcomes(self):  # pragma: no cover
        """Sequence of cached and executed opinion sweep outcomes.

        :returns: Opinion outcomes in chronological order.
        :rtype: Sequence[~knn.pipeline.context.OpinionSweepOutcome]
        """
        return self._outcomes.opinion_sweep_outcomes

    @property
    def studies(self) -> Sequence["StudySpec"]:  # pragma: no cover
        """Study descriptors used when rendering friendly labels.
        
        :returns: Study descriptors used to render friendly labels.
        :rtype: Sequence[~knn.pipeline.context.StudySpec]
        """
        return self._presentation.studies

    @property
    def metrics_by_feature(self):  # pragma: no cover
        """Per-feature mapping of metrics dictionaries keyed by study.

        :returns: Metrics snapshots per feature space and study.
        :rtype: Mapping[str, Mapping[str, Mapping[str, object]]]
        """
        return self._metrics.metrics_by_feature

    @property
    def opinion_metrics(self):  # pragma: no cover
        """Per-feature mapping of opinion metrics keyed by study.

        :returns: Opinion metrics per feature space and study.
        :rtype: Mapping[str, Mapping[str, Mapping[str, object]]]
        """
        return self._metrics.opinion_metrics

    @property
    def opinion_from_next_metrics(self):  # pragma: no cover
        """Per-feature mapping of 'opinion-from-next' metrics keyed by study.

        :returns: Opinion-from-next metrics per feature space and study.
        :rtype: Mapping[str, Mapping[str, Mapping[str, object]]]
        """
        return self._metrics.opinion_from_next_metrics

    @property
    def k_sweep(self) -> str:  # pragma: no cover
        """Textual representation of the ``k`` sweep grid.
        
        :returns: Human-readable summary of the k grid.
        :rtype: str
        """
        return self._presentation.k_sweep

    @property
    def loso_metrics(self):  # pragma: no cover
        """Leave-one-study-out metrics per feature space (optional).

        :returns: LOSO metrics per feature space, if available.
        :rtype: Mapping[str, Mapping[str, Mapping[str, object]]] | None
        """
        return self._metrics.loso_metrics

    @property
    def feature_spaces(self) -> Tuple[str, ...]:  # pragma: no cover
        """Ordered set of feature spaces included in the bundle.

        :returns: Tuple of feature space identifiers.
        :rtype: Tuple[str, ...]
        """
        return self._presentation.feature_spaces

    @property
    def sentence_model(self) -> Optional[str]:  # pragma: no cover
        """SentenceTransformer model name if applicable.

        :returns: Fully qualified model identifier or ``None``.
        :rtype: Optional[str]
        """
        return self._presentation.sentence_model

    @property
    def allow_incomplete(self) -> bool:  # pragma: no cover
        """Whether missing sweeps are tolerated when rendering summaries.

        :returns: True if report generation may continue with missing artefacts.
        :rtype: bool
        """
        return self._presentation.flags.allow_incomplete

    @property
    def include_next_video(self) -> bool:  # pragma: no cover
        """Flag indicating whether slate sections should be generated.

        :returns: True if slate sections should be generated.
        :rtype: bool
        """
        return self._presentation.flags.include_next_video

    @property
    def include_opinion(self) -> bool:  # pragma: no cover
        """Flag indicating whether opinion sections should be generated.

        :returns: True if opinion sections should be generated.
        :rtype: bool
        """
        return self._presentation.flags.include_opinion

    @property
    def include_opinion_from_next(self) -> bool:  # pragma: no cover
        """Flag for rendering opinion-from-next sections in reports.

        :returns: True if opinion-from-next sections should be generated.
        :rtype: bool
        """
        return self._presentation.flags.include_opinion_from_next

    @property
    def opinion_predictions_root(self) -> Path | None:  # pragma: no cover
        """Root directory containing opinion prediction artefacts.

        :returns: Path to opinion predictions or ``None``.
        :rtype: Optional[pathlib.Path]
        """
        return self._presentation.predictions.opinion_predictions_root

    @property
    def opinion_from_next_predictions_root(self) -> Path | None:  # pragma: no cover
        """Root directory for opinion-from-next prediction artefacts.

        :returns: Path to opinion-from-next predictions or ``None``.
        :rtype: Optional[pathlib.Path]
        """
        return self._presentation.predictions.opinion_from_next_predictions_root
