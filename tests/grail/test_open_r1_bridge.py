#!/usr/bin/env python
"""Smoke tests covering the GRAIL trainer's linkage to shared Open-R1 helpers."""

from __future__ import annotations

import pytest

from common.open_r1 import shared as shared_module

_GRAIL_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - optional dependency guard
    from grail import grail as grail_module
except ImportError as exc:  # pragma: no cover - optional dependency guard
    grail_module = None  # type: ignore[assignment]
    _GRAIL_IMPORT_ERROR = exc


def _require_grail() -> None:
    """Skip tests when optional GRAIL dependencies are not installed."""

    if _GRAIL_IMPORT_ERROR is not None:
        pytest.skip(f"Skipping grail/Open-R1 bridge tests: {_GRAIL_IMPORT_ERROR}")


def test_passthrough_fields_match_shared_definition() -> None:
    """GRAIL should inherit the shared passthrough metadata."""

    _require_grail()
    assert grail_module.PASSTHROUGH_FIELDS == shared_module.PASSTHROUGH_FIELDS


def test_train_columns_extend_shared_base() -> None:
    """GRAIL keeps the shared base columns and adds discriminator artefacts."""

    _require_grail()
    extra_columns = grail_module.TRAIN_KEEP_COLUMNS - shared_module.BASE_TRAIN_KEEP_COLUMNS
    assert shared_module.BASE_TRAIN_KEEP_COLUMNS.issubset(grail_module.TRAIN_KEEP_COLUMNS)
    assert extra_columns == {"slate_items_with_meta"}
