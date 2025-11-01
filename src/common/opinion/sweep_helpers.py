#!/usr/bin/env python
"""Helpers to construct sweep task base fields without duplication."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple, TypeVar

from .sweep_types import BaseSweepTask

ConfigT = TypeVar("ConfigT")


def base_task_kwargs(
    *,
    index: int,
    study: Any,
    config: Any,
    base_cli: Tuple[str, ...] | tuple[str, ...] | list[str],
    extra_cli: Tuple[str, ...] | tuple[str, ...] | list[str],
    run_root: Path,
    metrics_path: Path,
    train_participant_studies: tuple[str, ...] | Tuple[str, ...] | list[str] | None = None,
) -> Mapping[str, object]:
    """Return keyword arguments for :class:`common.opinion.BaseSweepTask`.

    This small helper reduces repeated boilerplate when initialising subclasses
    that simply forward the same base fields to ``BaseSweepTask.__init__``.
    """

    return {
        "index": int(index),
        "study": study,
        "config": config,
        "base_cli": tuple(base_cli),
        "extra_cli": tuple(extra_cli),
        "run_root": run_root,
        "metrics_path": metrics_path,
        "train_participant_studies": tuple(train_participant_studies or ()),
    }


__all__ = ["base_task_kwargs"]


class ExtrasSweepTask(BaseSweepTask[ConfigT]):
    """Base sweep task that initialises shared fields and stores extras.

    Subclasses can avoid duplicated ``__init__`` boilerplate by delegating
    construction to this class and providing a pipeline-specific ``_extras``
    dataclass instance.
    """

    def __init__(
        self,
        *,
        index: int,
        study: Any,
        config: Any,
        base_cli: Tuple[str, ...] | tuple[str, ...] | list[str],
        extra_cli: Tuple[str, ...] | tuple[str, ...] | list[str],
        run_root: Path,
        metrics_path: Path,
        train_participant_studies: tuple[str, ...] | Tuple[str, ...] | list[str] | None = None,
        extras: Any,
    ) -> None:
        # Delegate to the shared initialiser to avoid duplication across subclasses.
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

    def _init_shared(
        self,
        *,
        index: int,
        study: Any,
        config: Any,
        base_cli: Tuple[str, ...] | tuple[str, ...] | list[str],
        extra_cli: Tuple[str, ...] | tuple[str, ...] | list[str],
        run_root: Path,
        metrics_path: Path,
        train_participant_studies: tuple[str, ...] | Tuple[str, ...] | list[str] | None = None,
        extras: Any,
    ) -> None:
        """Initialise base sweep task fields and attach extras.

        This method is provided to let subclasses avoid repeating the same
        argument forwarding pattern in their ``__init__`` methods.
        """
        super().__init__(
            **base_task_kwargs(
                index=index,
                study=study,
                config=config,
                base_cli=tuple(base_cli),
                extra_cli=tuple(extra_cli),
                run_root=run_root,
                metrics_path=metrics_path,
                train_participant_studies=train_participant_studies,
            )
        )
        object.__setattr__(self, "_extras", extras)

    _extras: Any

    @property
    def extras(self) -> Any:
        """Return the pipeline-specific extras payload attached to this task."""
        return self._extras

    def with_extras(self, extras: Any) -> "ExtrasSweepTask[ConfigT]":
        """Return a copy of this task with a different extras payload.

        Useful for callers that want to vary auxiliary metadata while keeping
        the base sweep task fields identical.
        """
        return ExtrasSweepTask(
            index=self.index,
            study=self.study,
            config=self.config,
            base_cli=self.base_cli,
            extra_cli=self.extra_cli,
            run_root=self.run_root,
            metrics_path=self.metrics_path,
            train_participant_studies=self.train_participant_studies,
            extras=extras,
        )


__all__.append("ExtrasSweepTask")
