#!/usr/bin/env python
"""Convenience re-export for the GPT-4o pipeline entry point."""

from __future__ import annotations

from common.pipeline.gpt4o_models import PipelinePaths, SweepConfig, SweepOutcome

from .main import main

__all__ = ["main", "SweepConfig", "SweepOutcome", "PipelinePaths"]
