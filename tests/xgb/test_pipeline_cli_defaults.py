"""Tests for xgb.pipeline_cli default flags used by training scripts.

Ensures that extra text fields default to the extended set so features
show up by default when running the shell wrappers.
"""

from __future__ import annotations

from xgb.pipeline_cli import _parse_args
from xgb.cli import DEFAULT_XGB_TEXT_FIELDS


def test_pipeline_cli_uses_extended_extra_text_fields_by_default() -> None:
    args, extra = _parse_args([])
    assert not extra
    # The pipeline should default to the same extended fields as the xgb CLI
    assert args.extra_text_fields == DEFAULT_XGB_TEXT_FIELDS

