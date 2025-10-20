"""Backward-compatibility shim for :mod:`prompt_builder`.

The original implementation lived in this single module.  The logic now
resides in the ``prompt_builder`` package under ``src/prompt_builder/``.
Importing from here continues to work while downstream code migrates to
the new module layout.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'prompt_builder.py' is deprecated; import from the "
    "'prompt_builder' package (e.g. 'from prompt_builder.prompt import build_user_prompt') instead.",
    DeprecationWarning,
    stacklevel=2,
)

from prompt_builder import (  # noqa: F401 - re-exported for legacy imports
    as_list_json,
    build_user_prompt,
    clean_text,
    is_nanlike,
    render_profile,
    secs,
    synthesize_viewer_sentence,
    truthy,
)

__all__ = [
    "as_list_json",
    "build_user_prompt",
    "clean_text",
    "is_nanlike",
    "render_profile",
    "secs",
    "synthesize_viewer_sentence",
    "truthy",
]
