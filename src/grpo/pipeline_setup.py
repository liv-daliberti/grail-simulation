#!/usr/bin/env python
"""Setup utilities for :mod:`grpo.pipeline`.

Contains helpers to build the runtime context, resolve prompt text, and
summarize stage selections derived from CLI arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from common.utils.repo import resolve_repo_root_from_monkeypatch
from .config import DEFAULT_SYSTEM_PROMPT, OPINION_SYSTEM_PROMPT, repo_root as _cfg_repo_root


@dataclass(frozen=True)
class PipelineContext:
    """Filesystem locations reused across pipeline stages."""

    repo_root: Path
    out_dir: Path
    next_video_root: Path
    opinion_root: Path
    label: str

    @property
    def next_video_run_dir(self) -> Path:
        """Directory for this run's Next-Video stage artifacts.

        Returns:
            Path: ``next_video_root / label`` where stage outputs are written.
        """

        return self.next_video_root / self.label

    @property
    def opinion_run_dir(self) -> Path:
        """Directory for this run's Opinion stage artifacts.

        Returns:
            Path: ``opinion_root / label`` where stage outputs are written.
        """

        return self.opinion_root / self.label


@dataclass(frozen=True)
class PipelinePrompts:
    """Resolved system prompts used across evaluation tasks."""

    system: str
    opinion: str


@dataclass
class PipelineResults:
    """Container for evaluation results across stages."""

    next_video: object | None = None
    opinion: object | None = None


@dataclass(frozen=True)
class StageSelection:
    """Flags that control which parts of the pipeline to run."""

    stage: str
    run_next_video: bool
    run_opinion: bool

    @property
    def run_evaluations(self) -> bool:
        """Whether evaluation stages should run.

        Returns:
            bool: ``True`` when ``stage`` is ``"full"`` or ``"evaluate"``.
        """

        return self.stage in {"full", "evaluate"}

    @property
    def run_reports(self) -> bool:
        """Whether report-generation stages should run.

        Returns:
            bool: ``True`` when ``stage`` is ``"full"`` or ``"reports"``.
        """

        return self.stage in {"full", "reports"}


def _load_prompt_from_file(path: str | None, *, fallback: str) -> str:
    """Load prompt text from a file path with a safe fallback.

    Args:
        path: Optional path to a UTF-8 encoded prompt file. When ``None`` or
            empty, the ``fallback`` is returned instead of reading from disk.
        fallback: Default prompt text used when no file is provided.

    Returns:
        str: The prompt contents either from the given file or the fallback.
    """

    if not path:
        return fallback
    prompt_path = Path(path)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _build_context(args) -> PipelineContext:
    """Construct the shared filesystem context from CLI-like arguments.

    The returned paths are used consistently across pipeline stages so that
    intermediate products and reports land under predictable locations.

    Args:
        args: Namespace or object exposing ``out_dir`` (optional), ``model``
            (optional), and ``label`` (optional) attributes.

    Returns:
        PipelineContext: Immutable container with resolved directories.
    """

    out_dir = _resolve_out_dir(args)
    return PipelineContext(
        repo_root=_resolve_repo_root(),
        out_dir=out_dir,
        next_video_root=out_dir / "next_video",
        opinion_root=out_dir / "opinion",
        label=_derive_label(args),
    )


def _load_prompts(args) -> PipelinePrompts:
    """Resolve system prompts from CLI arguments or built-in defaults.

    Args:
        args: Namespace or object exposing ``system_prompt_file`` and
            ``opinion_prompt_file`` attributes (both optional file paths).

    Returns:
        PipelinePrompts: The base system prompt and the opinion system prompt.

    See Also:
        - :data:`grpo.config.DEFAULT_SYSTEM_PROMPT`
        - :data:`grpo.config.OPINION_SYSTEM_PROMPT`
    """

    return PipelinePrompts(
        system=_load_prompt_from_file(args.system_prompt_file, fallback=DEFAULT_SYSTEM_PROMPT),
        opinion=_load_prompt_from_file(
            args.opinion_prompt_file,
            fallback=OPINION_SYSTEM_PROMPT,
        ),
    )


def _resolve_stage_selection(args) -> StageSelection:
    """Summarize which pipeline components to run based on flags.

    Args:
        args: Namespace or object exposing ``stage`` (``"full"``,
            ``"evaluate"``, or ``"reports"``), and negated booleans
            ``no_next_video`` and ``no_opinion``.

    Returns:
        StageSelection: Normalized selection including convenience booleans.
    """

    return StageSelection(
        stage=args.stage,
        run_next_video=not args.no_next_video,
        run_opinion=not args.no_opinion,
    )


def _derive_label(args) -> str:
    """Derive a concise label identifying the evaluation run.

    Resolution order is:
    1) An explicit ``label`` argument when present and non-empty
    2) The basename of ``model`` with path separators and spaces normalized
    3) The default label ``"grpo"``

    Args:
        args: Namespace or object optionally exposing ``label`` and ``model``.

    Returns:
        str: A safe label suitable for directory names.
    """

    if getattr(args, "label", None):
        return args.label.strip()
    if getattr(args, "model", None):
        model_name = Path(args.model).name
        return model_name.replace("/", "_").replace(" ", "_")
    return "grpo"


def _resolve_out_dir(args) -> Path:
    """Resolve the base output directory for evaluation artifacts.

    Args:
        args: Namespace or object optionally exposing ``out_dir``.

    Returns:
        Path: The provided ``out_dir`` or ``repo_root()/models/grpo``.
    """
    if getattr(args, "out_dir", None):
        return Path(args.out_dir)
    return _resolve_repo_root() / "models" / "grpo"


def _resolve_repo_root() -> Path:
    """Resolve the repository root directory.

    Prefers a monkey‑patched ``grpo.pipeline._repo_root`` when present; otherwise
    falls back to :func:`grpo.config.repo_root`.
    """
    return resolve_repo_root_from_monkeypatch("grpo.pipeline", Path(_cfg_repo_root()))
