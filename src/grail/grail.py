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

"""GRPO training entrypoint with optional discriminator (GAIL-style) shaping."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

try:  # pragma: no cover - optional dependency
    from transformers import set_seed  # pylint: disable=import-error
except Exception:  # pragma: no cover - optional dependency
    # Avoid failing import-time when transformers has indirect import issues
    # (e.g. due to partial stubs injected into sys.modules by tests).
    set_seed = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from trl import ModelConfig  # pylint: disable=import-error
except Exception:  # pragma: no cover - optional dependency
    # Minimal placeholder used only for type annotations / script entrypoint.
    class ModelConfig:  # type: ignore[no-redef]
        """Lightweight stub for TRL's ``ModelConfig`` used at import time.

        This placeholder allows modules to be imported in environments where
        TRL is unavailable (e.g., Sphinx or lint-only runs). It should not be
        instantiated during real training; the actual class is provided by TRL.
        """
        pass

try:
    from common.open_r1.configs import GRPOConfig, GRPOScriptArguments
    from common.open_r1.shared import (
        execute_grpo_pipeline,
        make_grpo_execute_kwargs,
        parse_and_run,
        build_default_component_factory,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    module_dir = Path(__file__).resolve().parent
    _SRC_ROOT = str(module_dir.parent)
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)
    from common.open_r1.configs import GRPOConfig, GRPOScriptArguments
    from common.open_r1.shared import (
        execute_grpo_pipeline,
        make_grpo_execute_kwargs,
        parse_and_run,
        build_default_component_factory,
    )
try:
    from grail.grail_dataset import (
        PASSTHROUGH_FIELDS as _DATASET_PASSTHROUGH_FIELDS,
        TRAIN_KEEP_COLUMNS as _DATASET_TRAIN_KEEP_COLUMNS,
        _build_dataset_and_tokenizer,
        _grail_extra_fields,
        _prepare_dataset,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - script execution fallback
    if exc.name not in {"grail", "grail.grail_dataset"}:
        raise
    module_dir = Path(__file__).resolve().parent
    _SRC_ROOT = str(module_dir.parent)
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)
    from grail.grail_dataset import (
        PASSTHROUGH_FIELDS as _DATASET_PASSTHROUGH_FIELDS,
        TRAIN_KEEP_COLUMNS as _DATASET_TRAIN_KEEP_COLUMNS,
        _build_dataset_and_tokenizer,
        _grail_extra_fields,
        _prepare_dataset,
    )
try:  # pragma: no cover - optional dependency re-export
    from grail.grail_gail import (
        OnlineDiscriminator,
        RewardContext,
        _build_reward_contexts,
        _context_from_completion,
        _render_disc_text,
        _select_disc_device,
        _train_discriminator_from_contexts,
        make_gail_reward_fn,
    )
except Exception as _gail_import_error:  # pragma: no cover - optional dependency
    # Provide minimal shims so importing this module doesn't require transformers.
    class OnlineDiscriminator:  # type: ignore[too-few-public-methods]
        """Import-time stub that surfaces a helpful error when used.

        The real :class:`OnlineDiscriminator` lives in :mod:`grail.grail_gail`
        and requires ``transformers``. Attempting to construct this stub raises
        an :class:`ImportError` with installation guidance.
        """
        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError(
                "GAIL components require transformers. Install it with `pip install transformers`."
            ) from _gail_import_error

    RewardContext = None  # type: ignore[assignment]

    def _build_reward_contexts(*_args, **_kwargs):  # type: ignore[empty-body]
        raise ImportError(
            "GAIL components require transformers. Install it with `pip install transformers`."
        )

    def _context_from_completion(*_args, **_kwargs):  # type: ignore[empty-body]
        raise ImportError(
            "GAIL components require transformers. Install it with `pip install transformers`."
        )

    def _render_disc_text(*_args, **_kwargs):  # type: ignore[empty-body]
        raise ImportError(
            "GAIL components require transformers. Install it with `pip install transformers`."
        )

    def _select_disc_device(*_args, **_kwargs):  # type: ignore[empty-body]
        raise ImportError(
            "GAIL components require transformers. Install it with `pip install transformers`."
        )

    def _train_discriminator_from_contexts(*_args, **_kwargs):  # type: ignore[empty-body]
        raise ImportError(
            "GAIL components require transformers. Install it with `pip install transformers`."
        )

    def make_gail_reward_fn(*_args, **_kwargs):  # type: ignore[empty-body]
        """Stub for the GAIL reward-factory when ``transformers`` is missing.

        :raises ImportError: Always, with instructions to install dependencies.
        """
        raise ImportError(
            "GAIL components require transformers. Install it with `pip install transformers`."
        )
from grail.grail_mixer import LearnableRewardCallable, LearnableRewardMixer, MixerSetup
from grail.grail_rewards import (
    _adjust_reward_weights,
    _apply_reward_mixer,
    _maybe_enable_gail,
    _resolve_reward_functions,
)
from grail.grail_torch import nn, optim, resolve_torch_modules, torch  # noqa: F401
from grail.grail_utils import (
    _completion_text,
    _ensure_list,
    _parse_index_from_answer_block,
    _safe_int,
)

logger = logging.getLogger(__name__)

COMPONENT_FACTORY = build_default_component_factory()
TRAIN_KEEP_COLUMNS = _DATASET_TRAIN_KEEP_COLUMNS
PASSTHROUGH_FIELDS = _DATASET_PASSTHROUGH_FIELDS

# Ensure torch stubs are resolved for downstream modules.
resolve_torch_modules()


def main(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
) -> None:
    """
    Launch the GRPO (and optional GAIL) training loop for the configured setup.

    :param script_args: Command-line arguments describing data sources and rewards.
    :param training_args: Training hyperparameters for the GRPO trainer.
    :param model_args: Model configuration consumed by TRL for model/tokenizer loading.
    :returns: ``None``. Runs the training/evaluation pipeline for its side effects.
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(training_args.seed)

    dataset, tokenizer = _build_dataset_and_tokenizer(script_args, training_args, model_args)
    logger.debug("[grpo+gail] dataset splits: %s", list(dataset.keys()))
    reward_fns = _resolve_reward_functions(script_args, tokenizer)
    use_gail = _maybe_enable_gail(reward_fns)
    _adjust_reward_weights(training_args, reward_fns, use_gail)
    reward_fns = _apply_reward_mixer(training_args, reward_fns, use_gail)

    logger.info(
        "[grpo+gail] rewards=%s weights=%s",
        [getattr(f, "__name__", f.__class__.__name__) for f in reward_fns],
        training_args.reward_weights,
    )

    def _gail_eval_factory(grpo_trainer: Any) -> Callable[[], Mapping[str, Any]]:
        """
        Create an evaluation closure that temporarily disables GAIL gradients.

        :param grpo_trainer: Trainer instance whose ``evaluate`` method is wrapped.
        :returns: Zero-argument callable that runs evaluation with GAIL disabled.
        """

        def _evaluate_with_gail() -> Mapping[str, Any]:
            """
            Evaluate the trainer while ensuring GAIL gradients remain disabled.

            :returns: Mapping of evaluation metrics emitted by the wrapped trainer.
            """
            os.environ["GAIL_EVAL_MODE"] = "1"
            try:
                return grpo_trainer.evaluate()
            finally:
                os.environ["GAIL_EVAL_MODE"] = "0"

        return _evaluate_with_gail

    execute_grpo_pipeline(
        inputs=make_grpo_execute_kwargs(
            prefix="grail",
            evaluate_fn_factory=_gail_eval_factory,
            dataset=dataset,
            namespace={**globals(), **locals()},
        )
    )


__all__ = [
    "COMPONENT_FACTORY",
    "LearnableRewardCallable",
    "LearnableRewardMixer",
    "MixerSetup",
    "OnlineDiscriminator",
    "PASSTHROUGH_FIELDS",
    "RewardContext",
    "TRAIN_KEEP_COLUMNS",
    "_adjust_reward_weights",
    "_apply_reward_mixer",
    "_build_dataset_and_tokenizer",
    "_build_reward_contexts",
    "_completion_text",
    "_context_from_completion",
    "_ensure_list",
    "_grail_extra_fields",
    "_maybe_enable_gail",
    "_parse_index_from_answer_block",
    "_prepare_dataset",
    "_render_disc_text",
    "_safe_int",
    "_select_disc_device",
    "_train_discriminator_from_contexts",
    "main",
    "make_gail_reward_fn",
    "nn",
    "optim",
    "parse_and_run",
    "resolve_torch_modules",
    "torch",
]


if __name__ == "__main__":
    parse_and_run(main, (GRPOScriptArguments, GRPOConfig, ModelConfig))
