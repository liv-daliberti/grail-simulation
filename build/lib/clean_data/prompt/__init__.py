"""Namespace exposing the prompt analytics CLI and programmatic API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["generate_prompt_feature_report", "main"]


def _resolve_cli_attribute(name: str) -> Any:
    """Lazy-load the CLI module to avoid duplicate imports during ``python -m``."""
    module = import_module("clean_data.prompt.cli")
    return getattr(module, name)


def generate_prompt_feature_report(*args: Any, **kwargs: Any) -> Any:
    """Delegate to :func:`clean_data.prompt.cli.generate_prompt_feature_report`."""

    return _resolve_cli_attribute("generate_prompt_feature_report")(*args, **kwargs)


def main(*args: Any, **kwargs: Any) -> Any:
    """Delegate to :func:`clean_data.prompt.cli.main`."""

    return _resolve_cli_attribute("main")(*args, **kwargs)
