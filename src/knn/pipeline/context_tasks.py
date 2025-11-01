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


@dataclass(frozen=True)
class SweepTaskBase:
    """Grouped base fields required to construct a :class:`SweepTask`."""

    index: int
    study: StudySpec
    config: "SweepConfig"
    base_cli: Tuple[str, ...]
    extra_cli: Tuple[str, ...]
    run_root: Path
    metrics_path: Path
    train_participant_studies: Tuple[str, ...] = ()


class SweepTask(ExtrasSweepTask["SweepConfig"]):
    """Extend :class:`common.opinion.sweep_types.BaseSweepTask` with KNN metadata."""

    def __init__(
        self,
        *,
        base: SweepTaskBase | None = None,
        extras: _SweepTaskExtras | None = None,
        **legacy_kwargs: object,
    ) -> None:
        """Construct a sweep task from grouped ``base`` fields and ``extras``.

        Grouping reduces argument count and clarifies call sites while keeping
        the underlying :class:`ExtrasSweepTask` initialisation intact.
        """
        # Backwards compatibility: accept legacy flat kwargs used in tests.
        if base is None or extras is None:
            required = [
                "index",
                "study",
                "config",
                "base_cli",
                "extra_cli",
                "run_root",
                "metrics_path",
                "word2vec_model_dir",
                "issue",
                "issue_slug",
            ]
            for key in required:
                assert key in legacy_kwargs and legacy_kwargs[key] is not None
            base = SweepTaskBase(
                index=legacy_kwargs["index"],
                study=legacy_kwargs["study"],
                config=legacy_kwargs["config"],
                base_cli=tuple(legacy_kwargs["base_cli"]),  # type: ignore[arg-type]
                extra_cli=tuple(legacy_kwargs["extra_cli"]),  # type: ignore[arg-type]
                run_root=legacy_kwargs["run_root"],
                metrics_path=legacy_kwargs["metrics_path"],
                train_participant_studies=tuple(
                    legacy_kwargs.get("train_participant_studies", ())  # type: ignore[arg-type]
                ),
            )
            extras = _SweepTaskExtras(
                word2vec_model_dir=legacy_kwargs["word2vec_model_dir"],
                issue=legacy_kwargs["issue"],
                issue_slug=legacy_kwargs["issue_slug"],
            )

        # Use the shared initialiser from ExtrasSweepTask to avoid duplicate
        # forwarding boilerplate and keep logic in one place.
        self._init_shared(
            index=base.index,
            study=base.study,
            config=base.config,
            base_cli=base.base_cli,
            extra_cli=base.extra_cli,
            run_root=base.run_root,
            metrics_path=base.metrics_path,
            train_participant_studies=base.train_participant_studies,
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


@dataclass(frozen=True)
class OpinionTaskBase:
    """Grouped base fields required to construct an :class:`OpinionSweepTask`."""

    index: int
    study: StudySpec
    config: SweepConfig
    metrics_path: Path


class OpinionSweepTask(BaseOpinionSweepTask[SweepConfig]):
    """Extend :class:`common.opinion.sweep_types.BaseOpinionSweepTask` with CLI context."""

    def __init__(
        self,
        *,
        base: OpinionTaskBase | None = None,
        extras: _OpinionTaskExtras | None = None,
        **legacy_kwargs: object,
    ) -> None:
        if base is None or extras is None:
            required = ["index", "study", "config", "metrics_path", "base_cli", "extra_cli", "run_root"]
            for key in required:
                assert key in legacy_kwargs and legacy_kwargs[key] is not None
            base = OpinionTaskBase(
                index=legacy_kwargs["index"],
                study=legacy_kwargs["study"],
                config=legacy_kwargs["config"],
                metrics_path=legacy_kwargs["metrics_path"],
            )
            extras = _OpinionTaskExtras(
                base_cli=tuple(legacy_kwargs["base_cli"]),  # type: ignore[arg-type]
                extra_cli=tuple(legacy_kwargs["extra_cli"]),  # type: ignore[arg-type]
                run_root=legacy_kwargs["run_root"],
                word2vec_model_dir=legacy_kwargs.get("word2vec_model_dir"),
            )
        super().__init__(
            index=base.index,
            study=base.study,
            config=base.config,
            metrics_path=base.metrics_path,
        )
        object.__setattr__(self, "_extras", extras)

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
