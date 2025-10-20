"""Public API for the prompt_builder package.

This package exposes the same entry points that callers previously
imported from the single ``src/prompt_builder.py`` module while
organising the implementation across smaller, testable modules.
"""

from __future__ import annotations

from .formatters import clean_text
from .parsers import as_list_json, is_nanlike, secs, truthy
from .profiles import render_profile, synthesize_viewer_sentence
from .prompt import build_user_prompt
from .samples import PromptSample, generate_prompt_samples, write_samples_markdown

__all__ = [
    "as_list_json",
    "build_user_prompt",
    "clean_text",
    "is_nanlike",
    "generate_prompt_samples",
    "render_profile",
    "secs",
    "PromptSample",
    "synthesize_viewer_sentence",
    "truthy",
    "write_samples_markdown",
]
