#!/usr/bin/env python
"""Evaluation directory and cache path helpers for KNN pipeline.

Split from ``context.py`` to keep that module compact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class EvaluationOutputs:
    """Normalised directory layout for next-video and opinion evaluation artefacts."""

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
        """Materialise outputs while supporting legacy ``out_dir`` overrides.

        :param out_dir: Shared root directory used when specific outputs are absent.
        :param next_video_out_dir: Explicit directory for next-video evaluation artefacts.
        :param opinion_out_dir: Explicit directory for opinion evaluation artefacts.
        :returns: Normalised output directory structure for the evaluation run.
        """

        resolved_opinion = (
            opinion_out_dir
            if opinion_out_dir is not None
            else (out_dir or next_video_out_dir)
        )
        resolved_next = (
            next_video_out_dir
            if next_video_out_dir is not None
            else (out_dir or resolved_opinion)
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
class EvaluationOverrides:
    """Grouped directory overrides accepted by :meth:`EvaluationContext.from_args`.

    All attributes are optional and default to ``None``; when omitted, the
    builder falls back to the legacy keyword arguments or reasonable defaults.
    """

    out_dir: Path | None = None
    next_video_out_dir: Path | None = None
    opinion_out_dir: Path | None = None
    word2vec_model_dir: Path | None = None
    next_video_word2vec_dir: Path | None = None
    opinion_word2vec_dir: Path | None = None


@dataclass(frozen=True)
class EvaluationWord2VecPaths:
    """Normalised Word2Vec cache layout for next-video and opinion runs."""

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
        """Materialise Word2Vec cache directories while supporting legacy overrides.

        :param word2vec_model_dir: Shared Word2Vec cache root for both tasks.
        :param next_video_word2vec_dir: Cache directory for next-video Word2Vec artefacts.
        :param opinion_word2vec_dir: Cache directory for opinion Word2Vec artefacts.
        :param fallback_parent: Parent used to construct a default cache directory.
        :returns: Normalised Word2Vec cache directory structure.
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
        resolved_next = next_video_word2vec_dir or resolved_shared
        resolved_opinion = opinion_word2vec_dir or resolved_shared
        return cls(
            next_video=resolved_next,
            opinion=resolved_opinion,
            shared=resolved_shared,
        )


@dataclass(frozen=True)
class EvaluationContext:
    """Shared CLI/runtime parameters for final evaluation stages."""

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
        dirs: "EvaluationOverrides | None" = None,
        **overrides: Mapping[str, object],
    ) -> "EvaluationContext":
        """Build an :class:`EvaluationContext` from legacy or task-specific overrides.

        :param base_cli: Base CLI arguments used by evaluation commands.
        :param extra_cli: Additional CLI flags toggled by the caller.
        :param next_video_out_dir: Default next-video output directory.
        :param opinion_out_dir: Default opinion output directory.
        :param next_video_word2vec_dir: Default next-video Word2Vec cache directory.
        :param opinion_word2vec_dir: Default opinion Word2Vec cache directory.
        :param reuse_existing: Whether to reuse existing artefacts on disk.
        :param overrides: Optional keywords to override directory defaults.
        :returns: Fully constructed evaluation context with resolved paths.
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
        # Allow callers to pass grouped overrides via ``dirs`` while
        # maintaining backwards compatibility with legacy flat kwargs.
        _out_dir = (overrides.get("out_dir") if overrides.get("out_dir") is not None else None)
        _next_out = (
            overrides.get("next_video_out_dir")  # type: ignore[arg-type]
            if overrides.get("next_video_out_dir") is not None
            else (dirs.next_video_out_dir if dirs is not None else None)
        )
        _opinion_out = (
            overrides.get("opinion_out_dir")  # type: ignore[arg-type]
            if overrides.get("opinion_out_dir") is not None
            else (dirs.opinion_out_dir if dirs is not None else None)
        )

        outputs = EvaluationOutputs.from_keywords(
            out_dir=_out_dir,
            next_video_out_dir=_next_out,
            opinion_out_dir=_opinion_out,
        )

        _w2v_root = (
            overrides.get("word2vec_model_dir")  # type: ignore[arg-type]
            if overrides.get("word2vec_model_dir") is not None
            else (dirs.word2vec_model_dir if dirs is not None else None)
        )
        _w2v_next = (
            overrides.get("next_video_word2vec_dir")  # type: ignore[arg-type]
            if overrides.get("next_video_word2vec_dir") is not None
            else (dirs.next_video_word2vec_dir if dirs is not None else None)
        )
        _w2v_opinion = (
            overrides.get("opinion_word2vec_dir")  # type: ignore[arg-type]
            if overrides.get("opinion_word2vec_dir") is not None
            else (dirs.opinion_word2vec_dir if dirs is not None else None)
        )

        word2vec_paths = EvaluationWord2VecPaths.from_keywords(
            word2vec_model_dir=_w2v_root,
            next_video_word2vec_dir=_w2v_next,
            opinion_word2vec_dir=_w2v_opinion,
            fallback_parent=outputs.next_video,
        )
        return cls(
            base_cli=base_cli,
            extra_cli=extra_cli,
            reuse_existing=reuse_existing,
            outputs=outputs,
            word2vec_models=word2vec_paths,
        )

    # Backwards-compatible properties
    @property
    def next_video_out_dir(self) -> Path:  # pragma: no cover
        """Resolved next-video output directory.

        :rtype: pathlib.Path
        """
        return self.outputs.next_video

    @property
    def opinion_out_dir(self) -> Path:  # pragma: no cover
        """Resolved opinion output directory.

        :rtype: pathlib.Path
        """
        return self.outputs.opinion

    @property
    def out_dir(self) -> Path:  # pragma: no cover
        """Shared root output directory used by evaluations.

        :rtype: pathlib.Path
        """
        return self.outputs.shared

    @property
    def next_video_word2vec_dir(self) -> Path:  # pragma: no cover
        """Resolved Word2Vec cache directory for next-video runs.

        :rtype: pathlib.Path
        """
        return self.word2vec_models.next_video

    @property
    def opinion_word2vec_dir(self) -> Path:  # pragma: no cover
        """Resolved Word2Vec cache directory for opinion runs.

        :rtype: pathlib.Path
        """
        return self.word2vec_models.opinion

    @property
    def word2vec_model_dir(self) -> Path:  # pragma: no cover
        """Shared Word2Vec cache root directory.

        :rtype: pathlib.Path
        """
        return self.word2vec_models.shared
