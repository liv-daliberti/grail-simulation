#!/usr/bin/env python
"""Ensure the GRPO trainer still depends on the shared Open-R1 helpers."""

from __future__ import annotations

from common.open_r1 import shared as shared_module
from grpo import grpo as grpo_module


def test_keep_columns_match_shared_base() -> None:
    """GRPO should use the shared base training columns without modification."""

    assert grpo_module.KEEP_COLUMNS == shared_module.BASE_TRAIN_KEEP_COLUMNS


def test_passthrough_keys_match_shared_fields() -> None:
    """Pass-through metadata must stay aligned with the shared helper definitions."""

    assert grpo_module.PASSTHROUGH_KEYS == shared_module.PASSTHROUGH_FIELDS
