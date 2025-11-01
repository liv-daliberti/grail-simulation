#!/usr/bin/env python
"""Convenience wrappers for common CLI argument patterns."""

from __future__ import annotations

from pathlib import Path
from argparse import ArgumentParser

from .options import add_standard_eval_arguments


def add_shared_eval_arguments(parser: ArgumentParser, *, default_out_dir: str | Path) -> None:
    """Add standard evaluation arguments with project-wide defaults.

    This centralises repeated invocations of ``add_standard_eval_arguments``
    across baseline CLIs (KNN, XGB, etc.).

    Args:
        parser: ArgumentParser to extend in-place.
        default_out_dir: Base output directory for model artefacts.
    """

    add_standard_eval_arguments(
        parser,
        default_out_dir=str(default_out_dir),
        include_llm_args=False,
        include_opinion_args=False,
        include_studies_filter=False,
        dataset_default="data/cleaned_grail",
        issues_default="",
        include_legacy_aliases=True,
    )


__all__ = ["add_shared_eval_arguments"]
