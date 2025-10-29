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

import logging
import os
from typing import Any, List, Optional

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

from common.data.hf_datasets import DatasetDict

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.dataset_utils import drop_marked_rows, make_slate_validator
from open_r1.example_utils import (
    call_row_to_training_example,
    get_gold_next_id,
    gold_index_from_items,
    load_slate_items,
)
from open_r1.shared import (
    BASE_TRAIN_KEEP_COLUMNS,
    collect_passthrough_fields,
    make_grpo_execute_kwargs,
    execute_grpo_pipeline,
    parse_and_run,
    build_default_component_factory,
)
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_tokenizer

KEEP_COLUMNS = BASE_TRAIN_KEEP_COLUMNS

logger = logging.getLogger(__name__)

COMPONENT_FACTORY = build_default_component_factory()


def _ensure_training_dependencies() -> None:
    """Ensure optional training dependencies are installed before execution."""

    if set_seed is None:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "transformers must be installed to run GRPO training "
            "(pip install transformers)."
        )
    if (
        ModelConfig is None
        or get_peft_config is None
        or GRPOTrainer is None
    ):  # pragma: no cover - optional dependency guard
        raise ImportError(
            "trl must be installed to run GRPO training "
            "(pip install trl)."
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
    validator = make_slate_validator(
        load_slate_items=load_slate_items,
        lookup_gold_id=get_gold_next_id,
        resolve_gold_index=gold_index_from_items,
    )
    filtered = raw.filter(validator, fn_kwargs={"solution_key": solution_key})
    mapped = filtered.map(
        call_row_to_training_example,
        fn_kwargs={
            "system_prompt": training_args.system_prompt,
            "solution_key": solution_key,
            "max_history": max_hist,
            "passthrough_fn": collect_passthrough_fields,
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


def main(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
) -> None:
    """Orchestrate dataset preparation, trainer construction, and the training loop."""
    _ensure_training_dependencies()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(training_args.seed)

    solution_key = getattr(script_args, "dataset_solution_column", None)
    max_hist = int(os.environ.get("GRAIL_MAX_HISTORY", "12") or "12")
    dataset = _build_dataset(script_args, training_args, solution_key, max_hist)
    # Use dataset in a lightweight way to satisfy static analyzers.
    logger.debug("[grpo] dataset splits: %s", list(dataset.keys()))
    tokenizer = get_tokenizer(model_args, training_args)
    reward_fns = _load_reward_functions(script_args, tokenizer)
    _ensure_reward_weights(training_args, reward_fns)

    logger.info(
        "[grpo] rewards=%s weights=%s",
        [getattr(fn, "__name__", fn.__class__.__name__) for fn in reward_fns],
        training_args.reward_weights,
    )

    execute_grpo_pipeline(
        **make_grpo_execute_kwargs(
            prefix="grpo",
            dataset=dataset,
            namespace=locals(),
        )
    )


if __name__ == "__main__":
    _ensure_training_dependencies()
    parse_and_run(main, (GRPOScriptArguments, GRPOConfig, ModelConfig))
