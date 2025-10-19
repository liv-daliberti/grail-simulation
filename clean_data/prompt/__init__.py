"""Namespace exposing the prompt analytics CLI and programmatic API."""

from __future__ import annotations

from .cli import generate_prompt_feature_report, main

# pylint: disable=duplicate-code

__all__ = ["generate_prompt_feature_report", "main"]
