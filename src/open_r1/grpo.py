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

import datasets  # noqa: F401 (ensures HF dataset scripts are registered)
import torch
import transformers  # noqa: F401 (aligns generation configs)
from transformers import set_seed
from transformers.trainer_utils import IntervalStrategy, get_last_checkpoint
from trl import ModelConfig, TrlParser, get_peft_config
from trl.trainer.grpo_trainer import GRPOTrainer

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from prompt_builder.formatters import clean_text
from prompt_builder.parsers import as_list_json, is_nanlike
from prompt_builder.prompt import build_user_prompt
from prompt_builder.profiles import synthesize_viewer_sentence
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer

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
        vid = clean_text(it.get("id") or it.get("raw_id") or it.get("video_id") or cleaned.get("id"))
        channel = clean_text(
            it.get("channel_title") or it.get("channel_name") or it.get("channel") or cleaned.get("channel_title"),
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
    """Fallback: infer the clicked id from watch history when the dataset lacks an explicit label."""
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
    sys_msg = system_prompt or (
        "You are choosing EXACTLY ONE item from a short slate for a specific viewer.\n"
        "Think briefly in <think>…</think>, then output ONLY the option NUMBER (1..N) inside <answer>…</answer>.\n"
        "Format (STRICT): <think>…</think><answer>3</answer>"
    )

    slate_names = []
    for idx, it in enumerate(items, 1):
        name = (it.get("title") or it.get("id") or "(untitled)").strip()
        slate_names.append(f"{idx}. {name}")

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
        "slate_text": "\n".join(slate_names),
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

    passthrough = {
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
    for key in passthrough:
        if key in ex:
            example[key] = ex.get(key)

    return example


def main(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig) -> None:
    """Orchestrate dataset preparation, trainer construction, and the training loop."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(training_args.seed)

    raw = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args, training_args)

    solution_key = getattr(script_args, "dataset_solution_column", None)
    max_hist = int(os.environ.get("GRAIL_MAX_HISTORY", "12") or "12")

    def _ok(example: Dict[str, Any]) -> bool:
        items = _load_slate_items(example)
        if not items:
            return False
        gold = _get_gold_next_id(example, solution_key)
        if not gold:
            return False
        return _gold_index_from_items(gold, items) >= 1

    # Filter out rows without a resolvable gold choice before building prompts.
    raw = raw.filter(_ok)

    # Build structured prompts / answers; disable cache so schema changes take effect immediately.
    ds = raw.map(
        lambda ex: _row_to_example(ex, training_args.system_prompt, solution_key, max_hist=max_hist),
        load_from_cache_file=False,
    )

    if "__drop__" in ds[script_args.dataset_train_split].column_names:
        # Datasets can mark bad rows with __drop__; mirror TRL convention by removing them now.
        for split in list(ds.keys()):
            mask = [not flag for flag in ds[split]["__drop__"]]
            keep_indices = [idx for idx, keep in enumerate(mask) if keep]
            ds[split] = ds[split].select(keep_indices)

    # Drop heavy/raw columns to keep memory predictable for GRPOTrainer.
    keep_cols = {
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
    for split in list(ds.keys()):
        drop = [name for name in ds[split].column_names if name not in keep_cols]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    try:
        reward_fns = get_reward_funcs(script_args, ref_model=None, tokenizer=tokenizer)
    except Exception as exc:
        logger.warning("[rewards] get_reward_funcs failed: %s", exc)
        reward_fns = []

    weights = getattr(training_args, "reward_weights", None)
    if weights is None:
        training_args.reward_weights = [1.0] * len(reward_fns)
    elif len(weights) != len(reward_fns):
        raise ValueError(
            f"reward_weights length ({len(weights)}) != number of rewards ({len(reward_fns)}). "
            "Update the recipe so every reward has a matching weight."
        )

    if training_args.reward_weights:
        # Normalise in case the YAML supplies unnormalised weights.
        ws = [max(0.0, float(w)) for w in training_args.reward_weights]
        total = sum(ws) or 1.0
        training_args.reward_weights = [w / total for w in ws]

    logger.info(
        "[grpo] rewards=%s weights=%s",
        [getattr(f, "__name__", f.__class__.__name__) for f in reward_fns],
        training_args.reward_weights,
    )

    model = get_model(model_args, training_args)
    model.generation_config.return_dict_in_generate = True
    model.config.return_dict_in_generate = True

    train_split = script_args.dataset_train_split
    eval_ds = None
    if getattr(training_args, "do_eval", False):
        test_split = getattr(script_args, "dataset_test_split", None)
        if test_split and test_split in ds:
            eval_ds = ds[test_split]
            max_eval = getattr(training_args, "max_eval_samples", None)
            if isinstance(max_eval, int) and max_eval > 0 and len(eval_ds) > max_eval:
                eval_ds = eval_ds.shuffle(seed=training_args.seed).select(range(max_eval))
        else:
            logger.warning("[grpo] do_eval enabled but test split '%s' missing; disabling eval", test_split)
            eval_ds = None
            training_args.do_eval = False

    if getattr(training_args, "do_eval", False) and eval_ds is not None:
        if getattr(training_args, "evaluation_strategy", IntervalStrategy.NO) == IntervalStrategy.NO:
            logger.info("[grpo] forcing evaluation_strategy='steps' because do_eval is enabled")
            training_args.evaluation_strategy = IntervalStrategy.STEPS
        eval_steps = getattr(training_args, "eval_steps", None)
        if eval_steps is None or int(eval_steps) <= 0:
            raise ValueError("eval_steps must be > 0 when do_eval is enabled. Set a positive value in the config.")

    # Instantiate the TRL GRPO trainer with the cleaned datasets and reward hooks.
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fns,
        train_dataset=ds[train_split],
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
    )

    # Resume training seamlessly when prior checkpoints exist or user specifies one.
    last_ckpt = (
        training_args.resume_from_checkpoint
        or (get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None)
    )
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    if getattr(training_args, "do_eval", False) and eval_ds is not None:
        # Mid-training eval runs follow the schedule enforced above.
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if getattr(training_args, "push_to_hub", False):
        # Defer to TRL helper for Hub interactions so YAML hub_* fields apply automatically.
        trainer.push_to_hub(dataset_name=script_args.dataset_name, tags=["open-r1"])


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
