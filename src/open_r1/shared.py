"""Shared helpers used by the Open-R1 GRPO training entrypoints."""

from __future__ import annotations

import os

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from transformers.trainer_utils import IntervalStrategy, get_last_checkpoint
from trl import TrlParser

# Common prompt and column definitions -----------------------------------------------------------

PASSTHROUGH_FIELDS = {
    "issue",
    "session_id",
    "step_index",
    "display_step",
    "display_order_key",
    "issue_source",
    "issue_detail",
    "slate_source",
    "next_video_id",
    "next_video_title",
    "next_video_channel",
    "next_video_channel_id",
    "urlid",
    "topic_id",
}

BASE_TRAIN_KEEP_COLUMNS = {
    "prompt",
    "answer",
    "gold_index",
    "gold_id",
    "n_options",
    "viewer_profile",
    "state_text",
    "slate_items",
    "slate_text",
    "watched_detailed_json",
    "watched_vids_json",
    "current_video_id",
    "current_video_title",
    "task",
    "is_replay",
    "accuracy",
    "mix_group_id",
    "mix_copy_idx",
}


def collect_passthrough_fields(example: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a mapping containing the shared passthrough metadata."""

    return {field: example[field] for field in PASSTHROUGH_FIELDS if field in example}


def build_training_example(  # pylint: disable=too-many-arguments
    *,
    system_prompt: str,
    user_prompt: str,
    gold_index: int,
    gold_id: str,
    n_options: int,
    viewer_profile: str,
    slate_items: Sequence[Mapping[str, Any]],
    slate_text: str,
    watched_detailed_json: Any,
    watched_vids_json: Any,
    current_video_id: str,
    current_video_title: str,
    extra_fields: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble the common training example payload used by GRPO variants."""

    example: Dict[str, Any] = {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "answer": str(gold_index),
        "gold_index": gold_index,
        "gold_id": gold_id,
        "n_options": n_options,
        "viewer_profile": viewer_profile,
        "state_text": user_prompt,
        "slate_items": list(slate_items),
        "slate_text": slate_text,
        "watched_detailed_json": watched_detailed_json,
        "watched_vids_json": watched_vids_json,
        "current_video_id": current_video_id,
        "current_video_title": current_video_title,
        "task": "GRAIL",
        "is_replay": False,
        "accuracy": 0.0,
        "mix_group_id": -1,
        "mix_copy_idx": -1,
    }

    if extra_fields:
        example.update(extra_fields)
    return example


def prepare_eval_dataset(
    dataset: Mapping[str, Any],
    script_args: Any,
    training_args: Any,
    *,
    logger: Any,
    prefix: str,
) -> Optional[Any]:
    """Return the evaluation dataset subset when ``do_eval`` is enabled."""

    if not getattr(training_args, "do_eval", False):
        return None
    test_split = getattr(script_args, "dataset_test_split", None)
    if not (test_split and test_split in dataset):
        logger.warning(
            "[%s] do_eval enabled but test split '%s' missing; disabling eval",
            prefix,
            test_split,
        )
        training_args.do_eval = False
        return None
    eval_ds = dataset[test_split]
    max_eval = getattr(training_args, "max_eval_samples", None)
    if isinstance(max_eval, int) and 0 < max_eval < len(eval_ds):
        return eval_ds.shuffle(seed=training_args.seed).select(range(max_eval))
    return eval_ds


def configure_eval(
    training_args: Any,
    eval_ds: Optional[Any],
    *,
    logger: Any,
    prefix: str,
) -> None:
    """Adjust evaluation scheduling to ensure periodic evaluation runs."""

    if not getattr(training_args, "do_eval", False) or eval_ds is None:
        return
    strategy = getattr(training_args, "evaluation_strategy", IntervalStrategy.NO)
    if strategy == IntervalStrategy.NO:
        logger.info("[%s] forcing evaluation_strategy='steps' because do_eval is enabled", prefix)
        training_args.evaluation_strategy = IntervalStrategy.STEPS
    eval_steps = getattr(training_args, "eval_steps", None)
    if eval_steps is None or int(eval_steps) <= 0:
        raise ValueError(
            "eval_steps must be > 0 when do_eval is enabled. Set a positive value in the config."
        )


def resolve_checkpoint(training_args: Any) -> Optional[str]:
    """Return the checkpoint path used to resume GRPO training, if available."""

    resume = getattr(training_args, "resume_from_checkpoint", None)
    if resume:
        return resume
    output_dir = getattr(training_args, "output_dir", None)
    if output_dir:
        path = os.fspath(output_dir)
        if path and os.path.isdir(path):
            return get_last_checkpoint(path)
    return None


def parse_and_run(
    main_fn: Callable[[Any, Any, Any], None],
    argument_classes: Tuple[type, type, type],
) -> None:
    """Parse CLI arguments using ``argument_classes`` and execute ``main_fn``."""

    parser = TrlParser(argument_classes)
    main_fn(*parser.parse_args_and_config())
