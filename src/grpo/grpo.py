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

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

_SENTINEL = object()

## Exposed for tests: allow monkeypatching to simulate missing deps.
# These are initialised lazily from dynamic imports when left as _SENTINEL.
set_seed = _SENTINEL  # pylint: disable=invalid-name
ModelConfig = _SENTINEL  # pylint: disable=invalid-name
get_peft_config = _SENTINEL  # pylint: disable=invalid-name
GRPOTrainer = _SENTINEL  # pylint: disable=invalid-name


def _require_transformers_set_seed():
    """Return ``transformers.set_seed`` or raise an informative error.

    Uses dynamic import to avoid static import errors in lint-only environments.
    """

    try:  # pragma: no cover - optional dependency
        transformers = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            (
                "transformers must be installed to run GRPO training "
                "(pip install transformers)."
            )
        ) from exc
    func = getattr(transformers, "set_seed", None)
    if func is None:  # pragma: no cover - optional dependency
        raise ImportError(
            (
                "transformers.set_seed not found; ensure a compatible "
                "transformers version is installed."
            )
        )
    return func


def _require_trl_symbols():
    """Return ``(ModelConfig, get_peft_config, GRPOTrainer)`` from TRL.

    Imports dynamically to sidestep static import errors when TRL isn't present.
    """

    try:  # pragma: no cover - optional dependency
        trl_mod = importlib.import_module("trl")
        grpo_mod = importlib.import_module("trl.trainer.grpo_trainer")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "trl must be installed to run GRPO training (pip install trl)."
        ) from exc

    model_config_cls = getattr(trl_mod, "ModelConfig", None)
    peft_fn = getattr(trl_mod, "get_peft_config", None)
    trainer_cls = getattr(grpo_mod, "GRPOTrainer", None)
    if model_config_cls is None or peft_fn is None or trainer_cls is None:  # pragma: no cover
        raise ImportError(
            (
                "trl installation is incomplete or incompatible; "
                "ModelConfig/get_peft_config/GRPOTrainer missing."
            )
        )
    return model_config_cls, peft_fn, trainer_cls


def _get_set_seed_fn():
    """Return a set_seed callable, resolving from module attr or dynamic import."""

    mod = sys.modules[__name__]
    existing = getattr(mod, "set_seed", _SENTINEL)
    if existing is _SENTINEL:  # lazily import when not overridden by tests
        existing = _require_transformers_set_seed()
        setattr(mod, "set_seed", existing)
    if existing is None:
        raise ImportError(
            "transformers must be installed to run GRPO training (pip install transformers)."
        )
    return existing  # type: ignore[return-value]


def _get_trl_symbols():
    """Return (ModelConfig, get_peft_config, GRPOTrainer) honoring test overrides."""

    mod = sys.modules[__name__]
    model_config_cls = getattr(mod, "ModelConfig", _SENTINEL)
    peft_config_fn = getattr(mod, "get_peft_config", _SENTINEL)
    grpo_trainer_cls = getattr(mod, "GRPOTrainer", _SENTINEL)
    if (
        model_config_cls is _SENTINEL
        or peft_config_fn is _SENTINEL
        or grpo_trainer_cls is _SENTINEL
    ):
        model_config_cls, peft_config_fn, grpo_trainer_cls = _require_trl_symbols()
        # Surface the resolved symbols on the module for test monkeypatching.
        setattr(mod, "ModelConfig", model_config_cls)
        setattr(mod, "get_peft_config", peft_config_fn)
        setattr(mod, "GRPOTrainer", grpo_trainer_cls)
    return model_config_cls, peft_config_fn, grpo_trainer_cls

try:
    from common.data.hf_datasets import DatasetDict
    from common.open_r1.configs import GRPOConfig, GRPOScriptArguments
    from common.open_r1.dataset_utils import drop_marked_rows, make_slate_validator
    from common.open_r1.example_utils import (
        call_row_to_training_example,
        get_gold_next_id,
        gold_index_from_items,
        load_slate_items,
    )
    from common.open_r1.shared import (
        BASE_TRAIN_KEEP_COLUMNS,
        PASSTHROUGH_FIELDS,
        collect_passthrough_fields,
        make_grpo_execute_kwargs,
        execute_grpo_pipeline,
        parse_and_run,
        build_default_component_factory,
    )
    from common.open_r1.rewards import get_reward_funcs
    from common.open_r1.utils import get_dataset, get_tokenizer
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    module_dir = Path(__file__).resolve().parent
    SRC_ROOT = str(module_dir.parent)
    if SRC_ROOT not in sys.path:
        sys.path.insert(0, SRC_ROOT)
    from common.data.hf_datasets import DatasetDict
    from common.open_r1.configs import GRPOConfig, GRPOScriptArguments
    from common.open_r1.dataset_utils import drop_marked_rows, make_slate_validator
    from common.open_r1.example_utils import (
        call_row_to_training_example,
        get_gold_next_id,
        gold_index_from_items,
        load_slate_items,
    )
    from common.open_r1.shared import (
        BASE_TRAIN_KEEP_COLUMNS,
        PASSTHROUGH_FIELDS,
        collect_passthrough_fields,
        make_grpo_execute_kwargs,
        execute_grpo_pipeline,
        parse_and_run,
        build_default_component_factory,
    )
    from common.open_r1.rewards import get_reward_funcs
    from common.open_r1.utils import get_dataset, get_tokenizer

KEEP_COLUMNS = BASE_TRAIN_KEEP_COLUMNS
PASSTHROUGH_KEYS = PASSTHROUGH_FIELDS

logger = logging.getLogger(__name__)

COMPONENT_FACTORY = build_default_component_factory()


def _ensure_training_dependencies() -> None:
    """Ensure optional training dependencies are installed before execution.

    :returns: ``None``. Raises ``ImportError`` when required packages are missing.
    """

    # Validate that we can import the optional training dependencies at runtime,
    # but respect test overrides when module-level names are explicitly set.
    try:
        _get_set_seed_fn()
    except ImportError as exc:
        raise ImportError(
            (
                "transformers must be installed to run GRPO training "
                "(pip install transformers)."
            )
        ) from exc

    try:
        model_config_cls, peft_fn, trainer_cls = _get_trl_symbols()
    except ImportError as exc:
        raise ImportError(
            "trl must be installed to run GRPO training (pip install trl)."
        ) from exc

    if model_config_cls is None or peft_fn is None or trainer_cls is None:
        raise ImportError("trl must be installed to run GRPO training (pip install trl).")


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
    :returns: ``None``. Normalises weights in-place.
    """

    weights = getattr(training_args, "reward_weights", None)
    if weights is None:
        training_args.reward_weights = [1.0] * len(reward_fns)
        return
    if len(weights) != len(reward_fns):
        raise ValueError(
            (
                f"reward_weights length ({len(weights)}) != number of rewards "
                f"({len(reward_fns)}). "
            )
            + "Update the recipe so every reward has a matching weight."
        )
    if training_args.reward_weights:
        normalised = [max(0.0, float(value)) for value in training_args.reward_weights]
        total = sum(normalised) or 1.0
        training_args.reward_weights = [value / total for value in normalised]


def main(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: Any,
) -> None:
    """Orchestrate dataset preparation, trainer construction, and the training loop.

    :param script_args: High-level script arguments sourced from CLI/YAML.
    :param training_args: Training configuration for GRPO runs.
    :param model_args: Model configuration required to build tokenizers/models.
    :returns: ``None``. Executes training for its side effects (logging, checkpoints).
    """
    _ensure_training_dependencies()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _get_set_seed_fn()(training_args.seed)

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
        inputs=make_grpo_execute_kwargs(
            prefix="grpo",
            dataset=dataset,
            namespace={**globals(), **locals()},
        )
    )

def _cli_entrypoint() -> None:
    """Entry point wrapper that supports ``--help`` without heavy deps."""

    # Allow "--help" without requiring heavy optional dependencies.
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        try:
            model_config_type, _, _ = _get_trl_symbols()
        except ImportError:
            # Defer to shared.parse_and_run's lightweight help when TRL is missing.
            parse_and_run(main, (GRPOScriptArguments, GRPOConfig, object))
        else:
            parse_and_run(main, (GRPOScriptArguments, GRPOConfig, model_config_type))
        return

    _ensure_training_dependencies()
    model_config_type, _, _ = _get_trl_symbols()
    parse_and_run(main, (GRPOScriptArguments, GRPOConfig, model_config_type))


if __name__ == "__main__":
    _cli_entrypoint()
