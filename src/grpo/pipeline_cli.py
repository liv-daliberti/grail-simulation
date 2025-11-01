#!/usr/bin/env python
"""CLI argument parsing for :mod:`grpo.pipeline`.

Isolated to keep the main entrypoint small and to speed up import time
for modules that only need types or helpers.
"""

from __future__ import annotations

import argparse

from . import DEFAULT_DATASET_PATH, DEFAULT_EVAL_SPLIT
from .pipeline_common import DEFAULT_REGENERATE_HINT


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments into an :class:`argparse.Namespace`."""

    parser = argparse.ArgumentParser(
        description="Evaluate GRPO checkpoints on next-video and opinion tasks."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Dataset path or HF id.")
    parser.add_argument("--split", default=DEFAULT_EVAL_SPLIT, help="Evaluation split name.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Finetuned GRPO model path or hub identifier (required for evaluation stage).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision/tag when loading from the hub.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Torch dtype for model loading (e.g. bfloat16, float16, auto).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Human-readable label for report directories (defaults to model name).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory receiving evaluation artifacts (defaults to <repo>/models/grpo).",
    )
    parser.add_argument(
        "--reports-subdir",
        default="grpo",
        help="Subdirectory under reports/ to store Markdown summaries.",
    )
    parser.add_argument(
        "--baseline-label",
        default="GRPO",
        help="Display name for the baseline used in report headings.",
    )
    parser.add_argument(
        "--regenerate-hint",
        default=DEFAULT_REGENERATE_HINT,
        help=(
            "Sentence appended to the catalog README describing how to refresh artefacts. "
            "Pass an empty string to omit."
        ),
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Optional path containing the GRPO system prompt.",
    )
    parser.add_argument(
        "--opinion-prompt-file",
        default=None,
        help="Optional path containing the opinion evaluation system prompt.",
    )
    parser.add_argument(
        "--solution-key",
        default="next_video_id",
        help="Dataset column containing the gold next-video identifier.",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=12,
        help="Maximum watch-history depth forwarded to prompt_builder.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature used during generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus sampling top-p parameter.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens generated per completion.",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=25,
        help=(
            "Interval (in examples) to flush intermediate artefacts to disk. "
            "Use 0 to disable periodic flushing."
        ),
    )
    parser.add_argument(
        "--eval-max",
        type=int,
        default=0,
        help="Limit next-video evaluation rows (0 keeps all).",
    )
    parser.add_argument(
        "--issues",
        default="",
        help="Comma-separated issue filters for next-video evaluation.",
    )
    parser.add_argument(
        "--studies",
        default="",
        help="Comma-separated participant-study filters for next-video evaluation.",
    )
    parser.add_argument(
        "--opinion-studies",
        default="",
        help="Comma-separated opinion study keys to evaluate.",
    )
    parser.add_argument(
        "--opinion-max-participants",
        type=int,
        default=0,
        help="Optional cap on participants per opinion study.",
    )
    parser.add_argument(
        "--direction-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for treating opinion deltas as no-change.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation artifacts.",
    )
    parser.add_argument(
        "--no-next-video",
        action="store_true",
        help="Skip next-video evaluation and reporting.",
    )
    parser.add_argument(
        "--no-opinion",
        action="store_true",
        help="Skip opinion evaluation and reporting.",
    )
    parser.add_argument(
        "--stage",
        choices=["full", "evaluate", "reports"],
        default="full",
        help="Select which stage(s) to run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, etc.).",
    )
    return parser.parse_args(argv)
