#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standard GRPO training entrypoint for the GRAIL simulation dataset."""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from transformers import set_seed
except ImportError:  # pragma: no cover - optional dependency
    set_seed = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from trl import ModelConfig, get_peft_config
    from trl.trainer.grpo_trainer import GRPOTrainer
except ImportError:  # pragma: no cover - optional dependency
    ModelConfig = None  # type: ignore[assignment]
    get_peft_config = None  # type: ignore[assignment]
    GRPOTrainer = None  # type: ignore[assignment]

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.dataset_utils import drop_marked_rows, slate_has_gold
from open_r1.example_utils import (
    get_gold_next_id,
    gold_index_from_items,
    load_slate_items,
    row_to_training_example,
)
from open_r1.shared import (
    BASE_TRAIN_KEEP_COLUMNS,
    collect_passthrough_fields,
    configure_eval as shared_configure_eval,
    parse_and_run,
    prepare_eval_dataset as shared_prepare_eval_dataset,
    resolve_checkpoint as shared_resolve_checkpoint,
)
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from datasets import DatasetDict
else:  # pragma: no cover - fallback when datasets is unavailable at lint time
    DatasetDict = Any

KEEP_COLUMNS = BASE_TRAIN_KEEP_COLUMNS

logger = logging.getLogger(__name__)


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
    return row_to_training_example(
        ex,
        system_prompt=system_prompt,
        solution_key=sol_key,
        max_history=max_hist,
        passthrough_fn=collect_passthrough_fields,
    )


def _prune_columns(dataset: DatasetDict) -> DatasetDict:
    """Remove columns outside ``KEEP_COLUMNS`` from every split.

    :param dataset: Dataset dictionary to prune.
    :returns: Dataset dictionary with extraneous columns dropped.
    """

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
    """Build the training dataset with prompt conversion and filtering.

    :param script_args: High-level script arguments controlling dataset loading.
    :param training_args: Training hyperparameters providing system prompt, etc.
    :param solution_key: Optional column containing the gold target id.
    :param max_hist: Maximum history length to include in prompts.
    :returns: Dataset dictionary ready for GRPO training.
    """

    raw = get_dataset(script_args)
    validator = partial(
        slate_has_gold,
        load_slate_items=load_slate_items,
        lookup_gold_id=get_gold_next_id,
        resolve_gold_index=gold_index_from_items,
    )
    filtered = raw.filter(validator, fn_kwargs={"solution_key": solution_key})
    mapped = filtered.map(
        _row_to_example,
        fn_kwargs={
            "system_prompt": training_args.system_prompt,
            "sol_key": solution_key,
            "max_hist": max_hist,
        },
        load_from_cache_file=False,
    )
    drop_marked_rows(mapped, script_args.dataset_train_split)
    return _prune_columns(mapped)


def _load_reward_functions(
    script_args: GRPOScriptArguments,
    tokenizer,
) -> List[Any]:
    """Load reward functions configured for the training run.

    :param script_args: Script arguments containing reward configuration.
    :param tokenizer: Tokenizer instance passed to reward constructors.
    :returns: List of reward callables (may be empty on failure).
    """

    try:
        return get_reward_funcs(script_args, _ref_model=None, _tokenizer=tokenizer)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("[rewards] get_reward_funcs failed: %s", exc)
        return []


def _ensure_reward_weights(training_args: GRPOConfig, reward_fns: List[Any]) -> None:
    """Ensure reward weights align with the number of reward functions.

    :param training_args: Training arguments where weights are stored.
    :param reward_fns: List of reward functions enabled for the run.
    :raises ValueError: If the configured list length does not match ``reward_fns``.
    """

    weights = getattr(training_args, "reward_weights", None)
    if weights is None:
        training_args.reward_weights = [1.0] * len(reward_fns)
        return
    if len(weights) != len(reward_fns):
        raise ValueError(
            f"reward_weights length ({len(weights)}) != number of rewards ({len(reward_fns)}). "
            "Update the recipe so every reward has a matching weight."
        )
    if training_args.reward_weights:
        normalised = [max(0.0, float(value)) for value in training_args.reward_weights]
        total = sum(normalised) or 1.0
        training_args.reward_weights = [value / total for value in normalised]
def _resolve_checkpoint(training_args: GRPOConfig) -> Optional[str]:
    """Return the checkpoint path to resume from when available."""

    return shared_resolve_checkpoint(training_args)


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
    eval_ds = shared_prepare_eval_dataset(
        dataset,
        script_args,
        training_args,
        logger=logger,
        prefix="grpo",
    )
    shared_configure_eval(training_args, eval_ds, logger=logger, prefix="grpo")

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
    parse_and_run(main, (GRPOScriptArguments, GRPOConfig, ModelConfig))
