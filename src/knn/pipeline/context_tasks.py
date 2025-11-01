#!/usr/bin/env python
"""Task descriptors for KNN sweeps.

Split from ``context.py`` to reduce module size and clarify responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

from common.pipeline.types import StudySpec
from common.opinion.sweep_types import BaseOpinionSweepTask
from common.opinion.sweep_helpers import ExtrasSweepTask

from .context_config import SweepConfig


@dataclass(frozen=True)
class SweepTaskContext:
    """Shared CLI/runtime parameters required to materialise sweep tasks."""

    base_cli: Sequence[str]
    extra_cli: Sequence[str]
    sweep_dir: Path
    word2vec_model_base: Path


@dataclass(frozen=True)
class _SweepTaskExtras:
    """Additional KNN-specific task metadata grouped to reduce attributes."""

    word2vec_model_dir: Path | None
    issue: str
    issue_slug: str


class SweepTask(ExtrasSweepTask["SweepConfig"]):
    """Extend :class:`common.opinion.sweep_types.BaseSweepTask` with KNN metadata."""

    def __init__(
        self,
        *,
        index: int,
        study: StudySpec,
        config: "SweepConfig",
        base_cli: Tuple[str, ...],
        extra_cli: Tuple[str, ...],
        run_root: Path,
        metrics_path: Path,
        word2vec_model_dir: Path | None,
        issue: str,
        issue_slug: str,
        train_participant_studies: Tuple[str, ...] = (),
    ) -> None:
        extras = _SweepTaskExtras(
            word2vec_model_dir=word2vec_model_dir,
            issue=issue,
            issue_slug=issue_slug,
        )
        # Use the shared initialiser from ExtrasSweepTask to avoid duplicate
        # forwarding boilerplate and keep logic in one place.
        self._init_shared(
            index=index,
            study=study,
            config=config,
            base_cli=base_cli,
            extra_cli=extra_cli,
            run_root=run_root,
            metrics_path=metrics_path,
            train_participant_studies=train_participant_studies,
            extras=extras,
        )

    _extras: "_SweepTaskExtras"

    @property
    def word2vec_model_dir(self) -> Path | None:  # pragma: no cover - simple forwarding
        """Optional path to cached Word2Vec models used by this task."""
        return self._extras.word2vec_model_dir

    @property
    def issue(self) -> str:  # pragma: no cover - simple forwarding
        """Target issue identifier associated with the task/study."""
        return self._extras.issue

    @property
    def issue_slug(self) -> str:  # pragma: no cover - simple forwarding
        """Filesystem-friendly slug for the task's target issue."""
        return self._extras.issue_slug

    # train_participant_studies provided by BaseSweepTask


@dataclass(frozen=True)
class _OpinionTaskExtras:
    """Additional opinion task execution metadata grouped for compactness."""

    base_cli: Tuple[str, ...]
    extra_cli: Tuple[str, ...]
    run_root: Path
    word2vec_model_dir: Path | None


class OpinionSweepTask(BaseOpinionSweepTask[SweepConfig]):
    """Extend :class:`common.opinion.sweep_types.BaseOpinionSweepTask` with CLI context."""

    def __init__(
        self,
        *,
        index: int,
        study: StudySpec,
        config: SweepConfig,
        metrics_path: Path,
        base_cli: Tuple[str, ...],
        extra_cli: Tuple[str, ...],
        run_root: Path,
        word2vec_model_dir: Path | None,
    ) -> None:
        super().__init__(index=index, study=study, config=config, metrics_path=metrics_path)
        object.__setattr__(
            self,
            "_extras",
            _OpinionTaskExtras(
                base_cli=tuple(base_cli),
                extra_cli=tuple(extra_cli),
                run_root=run_root,
                word2vec_model_dir=word2vec_model_dir,
            ),
        )

    _extras: "_OpinionTaskExtras"

    @property
    def base_cli(self) -> Tuple[str, ...]:  # pragma: no cover - simple forwarding
        """Baseline CLI arguments reused across tasks."""
        return self._extras.base_cli

    @property
    def extra_cli(self) -> Tuple[str, ...]:  # pragma: no cover - simple forwarding
        """Additional passthrough CLI arguments for the job."""
        return self._extras.extra_cli

    @property
    def run_root(self) -> Path:  # pragma: no cover - simple forwarding
        """Directory where opinion sweep outputs are written."""
        return self._extras.run_root

    @property
    def word2vec_model_dir(self) -> Path | None:  # pragma: no cover - forwarding
        """Optional path to cached Word2Vec models used by this task."""
        return self._extras.word2vec_model_dir
