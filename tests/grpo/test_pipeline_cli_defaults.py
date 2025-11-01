#!/usr/bin/env python
"""Lock GRPO pipeline CLI defaults used by scripts and docs.

Ensures core flags remain stable after refactors so wrappers like
training scripts and README examples continue to work.
"""

from __future__ import annotations

import grpo
from grpo.pipeline_cli import _parse_args


def test_pipeline_cli_uses_expected_defaults() -> None:
    args = _parse_args([])

    # Dataset defaults should mirror the public constants.
    assert args.dataset == grpo.DEFAULT_DATASET_PATH
    assert args.split == grpo.DEFAULT_EVAL_SPLIT

    # Core runtime and generation defaults.
    assert args.dtype == "auto"
    assert args.max_history == 12
    assert args.temperature == 0.0
    assert args.top_p is None
    assert args.max_new_tokens == 128
    assert args.flush_interval == 25
    assert args.eval_max == 0

    # Stage and reporting defaults.
    assert args.stage == "full"
    assert args.no_next_video is False
    assert args.no_opinion is False
    assert args.reports_subdir == "grpo"
    assert args.baseline_label == "GRPO"
    assert args.direction_tolerance == 1e-6

