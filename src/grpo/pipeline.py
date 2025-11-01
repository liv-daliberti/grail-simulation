#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI entry point for evaluating and reporting on GRPO baselines.

This module focuses on orchestration and re-exports, delegating most
implementation details to small submodules under :mod:`grpo`:

- grpo.pipeline_cli — CLI parsing
- grpo.pipeline_common — logging and summary helpers
- grpo.pipeline_setup — context and prompt resolution
- grpo.pipeline_loaders — cache loaders and rebuilders
- grpo.pipeline_runner — evaluation and reporting runners

The public surface and internal helpers remain accessible via
`grpo.pipeline` to preserve backwards compatibility and tests.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
import time
import warnings
from typing import Any, Sequence

# Import helpers from the refactored submodules and alias them to preserve
# names that tests and external users may rely on.
from .pipeline_common import (
    configure_logging,
    _status,
    _log_next_video_summary,
    _log_opinion_summary,
)
from .pipeline_cli import _parse_args
from .pipeline_setup import (
    PipelineResults,
    _build_context,
    _load_prompts,
    _resolve_stage_selection,
)
from .pipeline_runner import (
    _run_evaluations,
    _generate_reports_if_needed,
)
from .config import repo_root as _cfg_repo_root

# Suppress noisy Pydantic 2.x field attribute warnings when present.
# Import lazily and defensively to work across environments without Pydantic.
try:
    import pydantic as _pydantic  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    _UNSUPPORTED_FIELD_ATTRIBUTE_WARNING = None
else:
    _UNSUPPORTED_FIELD_ATTRIBUTE_WARNING = getattr(
        getattr(_pydantic, "warnings", None),
        "UnsupportedFieldAttributeWarning",
        None,
    )

##
# Re-exports and lazy attribute resolution
# ---------------------------------------
#
# This module intentionally exposes a "flat" public surface under
# `grpo.pipeline` by re-exporting commonly used classes and functions from
# submodules such as `grpo.next_video` and `grpo.opinion`. To keep imports
# lightweight (especially during Sphinx autodoc) and to avoid unused-import
# lint errors, these items are resolved lazily via `__getattr__`.
#
# The names available for lazy import are enumerated in `__all__` and grouped
# by their source module. Accessing any such attribute triggers a dynamic
# import of its defining submodule and caches the attribute in this module's
# globals.
##

# Now that all imports are declared, apply the warning filter if supported.
if isinstance(_UNSUPPORTED_FIELD_ATTRIBUTE_WARNING, type) and issubclass(
    _UNSUPPORTED_FIELD_ATTRIBUTE_WARNING, Warning
):
    warnings.filterwarnings("ignore", category=_UNSUPPORTED_FIELD_ATTRIBUTE_WARNING)


LOGGER = logging.getLogger("grpo.pipeline")

_NV_EXPORTS = [
    # next_video
    "FilterSelection",
    "NextVideoEvaluationResult",
    "NextVideoDatasetSpec",
    "NextVideoEvaluationLimits",
    "NextVideoEvaluationSettings",
    "NextVideoPromptSettings",
    "run_next_video_evaluation",
]

# opinion types and helpers
# Derive the export list from grpo.opinion_types to avoid duplication.
try:
    _OPINION_BASE_EXPORTS = [
        n for n in importlib.import_module("grpo.opinion_types").__all__ if n != "OpinionArtifacts"
    ]
except (ModuleNotFoundError, ImportError, AttributeError):  # pragma: no cover - optional module
    # Keep a compact fallback to avoid duplicate-code lint with grpo.opinion_types
    _OPINION_BASE_EXPORTS = (
        "OpinionDatasetSpec,OpinionEvaluationControls,OpinionEvaluationResult,"
        "OpinionEvaluationSettings,OpinionInferenceContext,OpinionPromptSettings,"
        "OpinionStudyFiles,OpinionStudyResult,OpinionStudySummary"
    ).split(",")
_OPINION_EXPORTS = [*_OPINION_BASE_EXPORTS, "run_opinion_evaluation"]

_PIPELINE_COMMON_REEXPORTS = [
    "DEFAULT_REGENERATE_HINT",
    "_comma_separated",
]

_PIPELINE_SETUP_REEXPORTS = [
    "PipelineContext",
    "PipelinePrompts",
    "PipelineResults",
    "StageSelection",
    "_build_context",
    "_load_prompts",
    "_resolve_stage_selection",
]

_PIPELINE_LOADERS_REEXPORTS = [
    "_load_next_video_from_disk",
    "_load_opinion_from_disk",
]

# Public API re-exports
__all__ = [
    # next_video
    *_NV_EXPORTS,
    # opinion
    *[n for n in _OPINION_EXPORTS if n != "run_opinion_evaluation"],
    "run_opinion_evaluation",
    # pipeline_common
    "configure_logging",
    *_PIPELINE_COMMON_REEXPORTS,
    # pipeline_setup
    *_PIPELINE_SETUP_REEXPORTS,
    # pipeline_loaders
    *_PIPELINE_LOADERS_REEXPORTS,
    # opinion defaults commonly used by callers/tests
    "DEFAULT_SPECS",
    # legacy monkeypatch hook for tests
    "_repo_root",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin import shim
    """Dynamically resolve re-exported attributes.

    This function lazily imports the appropriate submodule and binds the
    requested attribute into this module's global namespace on first access.

    Parameters
    ----------
    name
        Attribute name requested by the caller.

    Returns
    -------
    Any
        The resolved attribute.

    Raises
    ------
    AttributeError
        If ``name`` is not a known re-export.
    RuntimeError
        If ``run_opinion_evaluation`` is requested but ``grpo.opinion`` is
        unavailable in the current environment.
    """

    if name in _NV_EXPORTS:
        mod = importlib.import_module(".next_video", __package__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj

    if name in _OPINION_EXPORTS:
        try:
            mod = importlib.import_module(".opinion", __package__)
            obj = getattr(mod, name)
        except ImportError as exc:
            if name == "run_opinion_evaluation":
                raise RuntimeError(
                    "grpo.opinion is unavailable during documentation import; "
                    "install optional dependencies or build outside Sphinx."
                ) from exc
            # Fall back to the lightweight types-only module for data classes.
            mod = importlib.import_module(".opinion_types", __package__)
            obj = getattr(mod, name)
        globals()[name] = obj
        return obj

    if name in _PIPELINE_COMMON_REEXPORTS:
        mod = importlib.import_module(".pipeline_common", __package__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj

    if name in _PIPELINE_SETUP_REEXPORTS:
        mod = importlib.import_module(".pipeline_setup", __package__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj

    if name in _PIPELINE_LOADERS_REEXPORTS:
        mod = importlib.import_module(".pipeline_loaders", __package__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj

    if name == "DEFAULT_SPECS":
        mod = importlib.import_module("common.opinion")
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj

    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - convenience for REPL/docs
    """Return a sorted list of attributes for introspection tools."""
    return sorted(set(globals().keys()) | set(__all__))

def _repo_root() -> Path:
    """Return the repository root directory.

    This helper exists for backwards compatibility with tests that monkeypatch
    ``grpo.pipeline._repo_root``.

    Returns
    -------
    pathlib.Path
        The repository root as determined by :func:`grpo.config.repo_root`.
    """

    return Path(_cfg_repo_root())


def main(argv: Sequence[str] | None = None) -> None:
    """Run the GRPO evaluation and reporting pipeline.

    Parameters
    ----------
    argv
        Optional list of command-line arguments. When ``None`` (the default),
        arguments are read from ``sys.argv``.

    Raises
    ------
    SystemExit
        If required arguments are missing for the selected stages (e.g.,
        ``--model`` is required when running evaluations).

    Notes
    -----
    - Logs a concise status summary of stage selection and context.
    - Supports running evaluations (next-video, opinion) and report generation.
    - Re-exports of commonly used types are available directly from
      ``grpo.pipeline`` for convenience.

    Examples
    --------
    Run both evaluation and reporting stages with a specific model::

        python -m grpo.pipeline \
            --stage evaluate,report \
            --model gpt-4o \
            --dataset wage --split test
    """

    args = _parse_args(list(argv) if argv is not None else None)
    run_start = time.perf_counter()
    configure_logging(args.log_level)

    selection = _resolve_stage_selection(args)
    context = _build_context(args)
    prompts = _load_prompts(args)
    _status(
        "Stage selection: stage=%s run_evaluations=%s next_video=%s opinion=%s run_reports=%s",
        selection.stage,
        selection.run_evaluations,
        selection.run_next_video,
        selection.run_opinion,
        selection.run_reports,
    )
    _status(
        "Pipeline context: repo_root=%s out_dir=%s label=%s model=%s dataset=%s split=%s",
        context.repo_root,
        context.out_dir,
        context.label,
        args.model or "<unset>",
        args.dataset,
        args.split,
    )

    if selection.run_evaluations and not args.model:
        raise SystemExit("--model must be provided when running the evaluate stage.")

    results = PipelineResults()
    if selection.run_evaluations:
        results = _run_evaluations(args, selection, context, prompts)
        # Emit summary lines mirroring previous behaviour.
        if selection.run_next_video:
            _log_next_video_summary(getattr(results, "next_video", None))
        if selection.run_opinion:
            _log_opinion_summary(getattr(results, "opinion", None))

    if selection.run_reports:
        _generate_reports_if_needed(selection, context, results, args)

    _status("Pipeline finished in %.2fs", time.perf_counter() - run_start)


if __name__ == "__main__":  # pragma: no cover
    main()
