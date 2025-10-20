#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standard GRPO training entrypoint for the GRAIL simulation dataset.

This variant performs vanilla GRPO without any discriminator shaping. It relies
solely on reward functions specified in the recipe YAML and uses the shared
prompt-builder utilities to render user messages from dataset columns.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

import datasets  # pylint: disable=unused-import  # ensures HF dataset scripts are registered
import torch  # pylint: disable=unused-import
import transformers  # pylint: disable=unused-import
from datasets import DatasetDict
from transformers import set_seed
from transformers.trainer_utils import IntervalStrategy, get_last_checkpoint
from trl import ModelConfig, TrlParser, get_peft_config
from trl.trainer.grpo_trainer import GRPOTrainer

from prompt_builder.formatters import clean_text
from prompt_builder.parsers import as_list_json, is_nanlike
from prompt_builder.prompt import build_user_prompt
from prompt_builder.profiles import synthesize_viewer_sentence

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer

DEFAULT_SYSTEM_PROMPT = (
    "You are choosing EXACTLY ONE item from a short slate for a specific viewer.\n"
    "Think briefly in <think>…</think>, then output ONLY the option NUMBER "
    "(1..N) inside <answer>…</answer>.\n"
    "Format (STRICT): <think>…</think><answer>3</answer>"
)

PASSTHROUGH_KEYS = {
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

KEEP_COLUMNS = {
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

logger = logging.getLogger(__name__)


def _canon(value: str) -> str:
    """Normalize a string for forgiving comparisons (lowercase and strip punctuation)."""
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower().strip())


def _load_slate_items(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract a pruned slate with consistent keys and sanitized text."""
    arr = as_list_json(ex.get("slate_items_json"))
    keep_keys = {
        "title",
        "id",
        "raw_id",
        "video_id",
        "video_title",
        "channel",
        "channel_title",
        "channel_name",
        "channel_id",
        "length_seconds",
        "duration_seconds",
        "duration",
        "watch_seconds",
        "score",
        "rank",
        "position",
        "reason",
        "source",
    }
    out: List[Dict[str, Any]] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        cleaned: Dict[str, Any] = {}
        for key, value in it.items():
            if key in keep_keys and not is_nanlike(value):
                cleaned[key] = value
        title = clean_text(
            it.get("title") or it.get("name") or it.get("video_title") or cleaned.get("title"),
            limit=160,
        )
        vid = clean_text(
            it.get("id") or it.get("raw_id") or it.get("video_id") or cleaned.get("id")
        )
        channel = clean_text(
            it.get("channel_title")
            or it.get("channel_name")
            or it.get("channel")
            or cleaned.get("channel_title"),
            limit=120,
        )
        if title:
            cleaned["title"] = title
        if vid:
            cleaned["id"] = vid
        if channel:
            cleaned["channel_title"] = channel
        if cleaned.get("id") or cleaned.get("title"):
            out.append(cleaned)
    return out


def _gold_index_from_items(gold: str, items: List[Dict[str, Any]]) -> int:
    """Return the 1-based index of the gold answer within the candidate slate."""
    gold = (gold or "").strip()
    if not gold or not items:
        return -1
    for idx, it in enumerate(items, 1):
        if gold == (it.get("id") or ""):
            return idx
    canon = _canon(gold)
    if canon:
        for idx, it in enumerate(items, 1):
            if canon == _canon(it.get("title", "")):
                return idx
    return -1


def _derive_next_from_history(ex: Dict[str, Any], current_id: str) -> str:
    """Infer the clicked id from watch history when no explicit label exists."""
    vids = as_list_json(ex.get("watched_vids_json"))
    if current_id and isinstance(vids, list) and vids:
        try:
            i = vids.index(current_id)
            if i + 1 < len(vids):
                nxt = vids[i + 1]
                if isinstance(nxt, str) and nxt.strip():
                    return nxt.strip()
        except ValueError:
            pass
    det = as_list_json(ex.get("watched_detailed_json"))
    if current_id and isinstance(det, list) and det:
        for j, r in enumerate(det):
            if isinstance(r, dict) and (r.get("id") or "").strip() == current_id:
                if j + 1 < len(det):
                    nxt = (det[j + 1].get("id") or "").strip()
                    if nxt:
                        return nxt
                break
    return ""


def _get_gold_next_id(ex: Dict[str, Any], sol_key: Optional[str]) -> str:
    """Pick the best available gold id from the row using preferred columns and fallbacks."""
    if sol_key and sol_key not in {"current_video_id", "current_id"}:
        value = ex.get(sol_key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ["next_video_id", "clicked_id", "video_id", "label", "answer"]:
        value = ex.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    current = (ex.get("current_video_id") or "").strip()
    return _derive_next_from_history(ex, current)


def _row_to_example(
    ex: Dict[str, Any],
    system_prompt: Optional[str],
    sol_key: Optional[str],
    max_hist: int = 12,
) -> Optional[Dict[str, Any]]:
    """
    Convert a raw dataset row into the prompt/answer payload expected by GRPO.

    Returns None when the slate is empty or the gold label cannot be mapped to it.
    """
    items = _load_slate_items(ex)
    if not items:
        return None
    gold_id = _get_gold_next_id(ex, sol_key)
    gold_idx = _gold_index_from_items(gold_id, items)
    if gold_idx < 1:
        return None

    user_msg = build_user_prompt(ex, max_hist=max_hist)
    sys_msg = system_prompt or DEFAULT_SYSTEM_PROMPT
    slate_text = "\n".join(
        f"{idx}. {(item.get('title') or item.get('id') or '(untitled)').strip()}"
        for idx, item in enumerate(items, 1)
    )

    example = {
        "prompt": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        "answer": str(gold_idx),
        "gold_index": gold_idx,
        "gold_id": gold_id,
        "n_options": int(ex.get("n_options") or len(items) or 0),
        "viewer_profile": str(ex.get("viewer_profile_sentence") or synthesize_viewer_sentence(ex)),
        "state_text": user_msg,
        "slate_items": items,
        "slate_text": slate_text,
        "watched_detailed_json": as_list_json(ex.get("watched_detailed_json")),
        "watched_vids_json": as_list_json(ex.get("watched_vids_json")),
        "current_video_id": str(ex.get("current_video_id") or ""),
        "current_video_title": str(ex.get("current_video_title") or ""),
        "task": "GRAIL",
        "is_replay": False,
        "accuracy": 0.0,
        "mix_group_id": -1,
        "mix_copy_idx": -1,
    }

    for key in PASSTHROUGH_KEYS:
        if key in ex:
            example[key] = ex.get(key)

    return example


def _valid_example(example: Dict[str, Any], solution_key: Optional[str]) -> bool:
    items = _load_slate_items(example)
    if not items:
        return False
    gold = _get_gold_next_id(example, solution_key)
    if not gold:
        return False
    return _gold_index_from_items(gold, items) >= 1


def _drop_marked_rows(dataset: DatasetDict, train_split: str) -> None:
    if "__drop__" not in dataset[train_split].column_names:
        return
    for split in list(dataset.keys()):
        mask = [not flag for flag in dataset[split]["__drop__"]]
        keep_indices = [idx for idx, keep in enumerate(mask) if keep]
        dataset[split] = dataset[split].select(keep_indices)


def _prune_columns(dataset: DatasetDict) -> DatasetDict:
    for split in list(dataset.keys()):
        drop = [name for name in dataset[split].column_names if name not in KEEP_COLUMNS]
        if drop:
            dataset[split] = dataset[split].remove_columns(drop)
    return dataset


def _build_dataset(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    solution_key: Optional[str],
    max_hist: int,
) -> DatasetDict:
    raw = get_dataset(script_args)
    filtered = raw.filter(_valid_example, fn_kwargs={"solution_key": solution_key})
    mapped = filtered.map(
        _row_to_example,
        fn_kwargs={
            "system_prompt": training_args.system_prompt,
            "sol_key": solution_key,
            "max_hist": max_hist,
        },
        load_from_cache_file=False,
    )
    _drop_marked_rows(mapped, script_args.dataset_train_split)
    return _prune_columns(mapped)


def _load_reward_functions(
    script_args: GRPOScriptArguments,
    tokenizer,
) -> List[Any]:
    try:
        return get_reward_funcs(script_args, ref_model=None, tokenizer=tokenizer)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("[rewards] get_reward_funcs failed: %s", exc)
        return []


def _ensure_reward_weights(training_args: GRPOConfig, reward_fns: List[Any]) -> None:
    weights = getattr(training_args, "reward_weights", None)
    if weights is None:
        training_args.reward_weights = [1.0] * len(reward_fns)
        return
    if len(weights) != len(reward_fns):
        raise ValueError(
            "reward_weights length ({}) != number of rewards ({}). "
            "Update the recipe so every reward has a matching weight.".format(
                len(weights),
                len(reward_fns),
            )
        )
    if training_args.reward_weights:
        normalised = [max(0.0, float(value)) for value in training_args.reward_weights]
        total = sum(normalised) or 1.0
        training_args.reward_weights = [value / total for value in normalised]


def _prepare_eval_dataset(
    dataset: DatasetDict,
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
) -> Optional[Any]:
    if not getattr(training_args, "do_eval", False):
        return None
    test_split = getattr(script_args, "dataset_test_split", None)
    if not (test_split and test_split in dataset):
        logger.warning(
            "[grpo] do_eval enabled but missing test split '%s'; disabling eval",
            test_split,
        )
        training_args.do_eval = False
        return None
    eval_ds = dataset[test_split]
    max_eval = getattr(training_args, "max_eval_samples", None)
    if isinstance(max_eval, int) and 0 < max_eval < len(eval_ds):
        return eval_ds.shuffle(seed=training_args.seed).select(range(max_eval))
    return eval_ds


def _configure_eval(training_args: GRPOConfig, eval_ds: Optional[Any]) -> None:
    if not getattr(training_args, "do_eval", False) or eval_ds is None:
        return
    strategy = getattr(training_args, "evaluation_strategy", IntervalStrategy.NO)
    if strategy == IntervalStrategy.NO:
        logger.info("[grpo] forcing evaluation_strategy='steps' because do_eval is enabled")
        training_args.evaluation_strategy = IntervalStrategy.STEPS
    eval_steps = getattr(training_args, "eval_steps", None)
    if eval_steps is None or int(eval_steps) <= 0:
        raise ValueError(
            "eval_steps must be > 0 when do_eval is enabled. "
            "Set a positive value in the config."
        )


def _resolve_checkpoint(training_args: GRPOConfig) -> Optional[str]:
    resume = getattr(training_args, "resume_from_checkpoint", None)
    if resume:
        return resume
    output_dir = training_args.output_dir
    if os.path.isdir(output_dir):
        return get_last_checkpoint(output_dir)
    return None


def main(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
) -> None:
    """Orchestrate dataset preparation, trainer construction, and the training loop."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(training_args.seed)

    solution_key = getattr(script_args, "dataset_solution_column", None)
    max_hist = int(os.environ.get("GRAIL_MAX_HISTORY", "12") or "12")
    dataset = _build_dataset(script_args, training_args, solution_key, max_hist)
    tokenizer = get_tokenizer(model_args, training_args)
    reward_fns = _load_reward_functions(script_args, tokenizer)
    _ensure_reward_weights(training_args, reward_fns)

    logger.info(
        "[grpo] rewards=%s weights=%s",
        [getattr(fn, "__name__", fn.__class__.__name__) for fn in reward_fns],
        training_args.reward_weights,
    )

    model = get_model(model_args, training_args)
    model.generation_config.return_dict_in_generate = True
    model.config.return_dict_in_generate = True

    train_split = script_args.dataset_train_split
    eval_ds = _prepare_eval_dataset(dataset, script_args, training_args)
    _configure_eval(training_args, eval_ds)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fns,
        train_dataset=dataset[train_split],
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
    )

    last_ckpt = _resolve_checkpoint(training_args)
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    if getattr(training_args, "do_eval", False) and eval_ds is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if getattr(training_args, "push_to_hub", False):
        trainer.push_to_hub(dataset_name=script_args.dataset_name, tags=["open-r1"])


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    parsed_script_args, parsed_training_args, parsed_model_args = parser.parse_args_and_config()
    main(parsed_script_args, parsed_training_args, parsed_model_args)
