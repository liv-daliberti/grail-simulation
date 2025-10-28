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

"""Shared helpers for the Open R1 training and evaluation pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from open_r1.utils import get_model

try:  # pragma: no cover - optional dependency
    from transformers.trainer_utils import IntervalStrategy, get_last_checkpoint
except ImportError:  # pragma: no cover - optional dependency
    from enum import Enum

    class IntervalStrategyFallback(str, Enum):
        """Fallback interval strategy constants when transformers is unavailable."""

        NO = "no"
        STEPS = "steps"

    IntervalStrategy = IntervalStrategyFallback  # type: ignore[assignment]

    def get_last_checkpoint(*args: Any, **kwargs: Any) -> str:  # type: ignore[override]
        """Signal that transformers is required for checkpoint discovery."""

        raise ImportError(
            "transformers must be installed to resolve checkpoints "
            "(pip install transformers)."
        )

try:  # pragma: no cover - optional dependency
    from trl import TrlParser, get_peft_config  # pylint: disable=import-error
    from trl.trainer.grpo_trainer import GRPOTrainer  # pylint: disable=import-error
except ImportError:  # pragma: no cover - optional dependency
    TrlParser = None  # type: ignore[assignment]
    get_peft_config = None  # type: ignore[assignment]
    GRPOTrainer = None  # type: ignore[assignment]

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


EvalFn = Callable[[], Mapping[str, Any]]
EvalFnFactory = Callable[[Any], EvalFn]


@dataclass(frozen=True)
class GrpoTrainerSetup:
    """Immutable bundle describing how to initialise a GRPO trainer."""

    trainer_cls: Any
    model: Any
    training_args: Any
    reward_funcs: Sequence[Any]
    model_args: Any
    tokenizer: Any
    peft_config_fn: Optional[Callable[[Any], Any]] = None


@dataclass(frozen=True)
class GrpoTrainerDatasets:
    """Datasets passed to the trainer."""

    train_dataset: Any
    eval_dataset: Optional[Any]


@dataclass(frozen=True)
class GrpoTrainerDataConfig:
    """Dataset mapping and split selections used to build trainer datasets."""

    dataset: Mapping[str, Any]
    train_split: str
    eval_dataset: Optional[Any]


@dataclass(frozen=True)
class GrpoPipelineComponents:
    """Core objects required to build and run the GRPO training pipeline."""

    model_builder: Callable[[Any, Any], Any]
    trainer_cls: Any
    reward_funcs: Sequence[Any]
    tokenizer: Any
    peft_config_fn: Optional[Callable[[Any], Any]] = None
    evaluate_fn_factory: Optional[EvalFnFactory] = None


@dataclass(frozen=True)
class GrpoPipelineContext:
    """Execution context shared across pipeline stages."""

    dataset: Mapping[str, Any]
    script_args: Any
    training_args: Any
    model_args: Any
    logger: Any
    prefix: str


@dataclass(frozen=True)
class GrpoComponentFactory:
    """Factory that constructs :class:`GrpoPipelineComponents` with shared defaults."""

    model_builder: Callable[[Any, Any], Any]
    trainer_cls: Any
    peft_config_fn: Optional[Callable[[Any], Any]] = None

    def build(
        self,
        *,
        reward_funcs: Sequence[Any],
        tokenizer: Any,
        evaluate_fn_factory: Optional[EvalFnFactory] = None,
    ) -> GrpoPipelineComponents:
        """Return a :class:`GrpoPipelineComponents` instance with cached defaults."""

        return GrpoPipelineComponents(
            model_builder=self.model_builder,
            trainer_cls=self.trainer_cls,
            reward_funcs=reward_funcs,
            tokenizer=tokenizer,
            peft_config_fn=self.peft_config_fn,
            evaluate_fn_factory=evaluate_fn_factory,
        )


def build_default_component_factory() -> GrpoComponentFactory:
    """Return the default :class:`GrpoComponentFactory` used by GRPO entrypoints."""

    return GrpoComponentFactory(
        model_builder=get_model,
        trainer_cls=GRPOTrainer,
        peft_config_fn=get_peft_config,
    )


def build_grpo_context(
    dataset: Mapping[str, Any],
    script_args: Any,
    training_args: Any,
    model_args: Any,
    logger: Any,
    *,
    prefix: str,
) -> GrpoPipelineContext:
    """Return a :class:`GrpoPipelineContext` populated with the supplied arguments."""

    return GrpoPipelineContext(
        dataset=dataset,
        script_args=script_args,
        training_args=training_args,
        model_args=model_args,
        logger=logger,
        prefix=prefix,
    )


def build_grpo_pipeline_bundle(
    *,
    model_builder: Callable[[Any, Any], Any],
    trainer_cls: Any,
    reward_funcs: Sequence[Any],
    tokenizer: Any,
    dataset: Mapping[str, Any],
    script_args: Any,
    training_args: Any,
    model_args: Any,
    logger: Any,
    prefix: str,
    peft_config_fn: Optional[Callable[[Any], Any]] = None,
    evaluate_fn_factory: Optional[EvalFnFactory] = None,
) -> Tuple[GrpoPipelineComponents, GrpoPipelineContext]:
    """
    Assemble the shared GRPO pipeline components and execution context.

    Centralises the repeated constructor invocations used by the GRPO entry
    points so that pylint's duplicate-code check sees a single implementation.
    """

    components = GrpoPipelineComponents(
        model_builder=model_builder,
        trainer_cls=trainer_cls,
        reward_funcs=reward_funcs,
        tokenizer=tokenizer,
        peft_config_fn=peft_config_fn,
        evaluate_fn_factory=evaluate_fn_factory,
    )
    context = GrpoPipelineContext(
        dataset=dataset,
        script_args=script_args,
        training_args=training_args,
        model_args=model_args,
        logger=logger,
        prefix=prefix,
    )
    return components, context


def build_grpo_trainer(*, setup: GrpoTrainerSetup, datasets: GrpoTrainerDatasets) -> Any:
    """Instantiate a GRPO trainer with the shared configuration knobs."""

    peft_config = (
        setup.peft_config_fn(setup.model_args) if setup.peft_config_fn is not None else None
    )
    return setup.trainer_cls(
        model=setup.model,
        args=setup.training_args,
        reward_funcs=list(setup.reward_funcs),
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        peft_config=peft_config,
        processing_class=setup.tokenizer,
    )
def run_trainer_loop(
    trainer: Any,
    training_args: Any,
    *,
    dataset_name: str,
    eval_dataset: Optional[Any],
    evaluate_fn: Optional[Callable[[], Mapping[str, Any]]] = None,
) -> Tuple[Any, Optional[Mapping[str, Any]]]:
    """
    Execute the standard GRPO training loop with logging, evaluation, and hub push.

    :param trainer: Configured GRPO trainer.
    :param training_args: Training configuration namespace.
    :param dataset_name: Dataset identifier used when pushing to the hub.
    :param eval_dataset: Evaluation dataset (presence toggles evaluation).
    :param evaluate_fn: Optional callback overriding ``trainer.evaluate``.
    :returns: Tuple of the training result and optional evaluation metrics.
    """

    last_ckpt = resolve_checkpoint(training_args)
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    eval_metrics: Optional[Mapping[str, Any]] = None
    if getattr(training_args, "do_eval", False) and eval_dataset is not None:
        run_eval = evaluate_fn or trainer.evaluate
        eval_metrics = run_eval()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if getattr(training_args, "push_to_hub", False):
        trainer.push_to_hub(dataset_name=dataset_name, tags=["open-r1"])

    return train_result, eval_metrics


def run_trainer_with_script_args(
    trainer: Any,
    training_args: Any,
    script_args: Any,
    eval_dataset: Optional[Any],
    *,
    evaluate_fn: Optional[Callable[[], Mapping[str, Any]]] = None,
) -> Tuple[Any, Optional[Mapping[str, Any]]]:
    """Invoke :func:`run_trainer_loop` using attributes from ``script_args``.

    Centralises the common pattern of pulling ``dataset_name`` from CLI arguments
    before running the training loop.
    """

    dataset_name = str(getattr(script_args, "dataset_name", ""))
    return run_trainer_loop(
        trainer,
        training_args,
        dataset_name=dataset_name,
        eval_dataset=eval_dataset,
        evaluate_fn=evaluate_fn,
    )


def configure_and_run_grpo_trainer(
    *,
    setup: GrpoTrainerSetup,
    data_config: GrpoTrainerDataConfig,
    script_args: Any,
    evaluate_fn_factory: Optional[EvalFnFactory] = None,
) -> Tuple[Any, Tuple[Any, Optional[Mapping[str, Any]]]]:
    """
    Build a GRPO trainer using shared defaults and execute the training loop.

    :param setup: Bundle describing how to initialise the trainer.
    :param data_config: Dataset mapping and split information.
    :param script_args: CLI arguments namespace for logging and metadata.
    :param evaluate_fn_factory:
        Optional factory returning an evaluation function when provided the
        constructed trainer. Enables call-sites to inject custom evaluation
        wrappers without repeating boilerplate.
    :returns: Tuple of the trainer and the result from ``run_trainer_with_script_args``.
    """

    trainer = build_grpo_trainer(
        setup=setup,
        datasets=GrpoTrainerDatasets(
            train_dataset=data_config.dataset[data_config.train_split],
            eval_dataset=data_config.eval_dataset,
        ),
    )
    evaluate_fn = (
        evaluate_fn_factory(trainer) if evaluate_fn_factory is not None else None
    )
    return trainer, run_trainer_with_script_args(
        trainer,
        setup.training_args,
        script_args,
        data_config.eval_dataset,
        evaluate_fn=evaluate_fn,
    )


def prepare_model_eval_and_run_grpo(
    *,
    components: GrpoPipelineComponents,
    context: GrpoPipelineContext,
) -> Tuple[Any, Tuple[Any, Optional[Mapping[str, Any]]]]:
    """
    Build the GRPO model, prepare evaluation state, and run the shared trainer loop.

    :param components: Core callables and objects required to construct the trainer.
    :param context: Dataset, configuration, and logging context.
    :returns: Tuple of the trainer and the result from ``run_trainer_with_script_args``.
    """

    model = components.model_builder(context.model_args, context.training_args)
    if hasattr(model, "generation_config"):
        model.generation_config.return_dict_in_generate = True
    if hasattr(model, "config"):
        model.config.return_dict_in_generate = True

    train_split = getattr(context.script_args, "dataset_train_split", "train")
    eval_dataset = prepare_eval_dataset(
        context.dataset,
        context.script_args,
        context.training_args,
        logger=context.logger,
        prefix=context.prefix,
    )
    configure_eval(
        context.training_args,
        eval_dataset,
        logger=context.logger,
        prefix=context.prefix,
    )

    return configure_and_run_grpo_trainer(
        setup=GrpoTrainerSetup(
            trainer_cls=components.trainer_cls,
            model=model,
            training_args=context.training_args,
            reward_funcs=components.reward_funcs,
            model_args=context.model_args,
            tokenizer=components.tokenizer,
            peft_config_fn=components.peft_config_fn,
        ),
        data_config=GrpoTrainerDataConfig(
            dataset=context.dataset,
            train_split=train_split,
            eval_dataset=eval_dataset,
        ),
        script_args=context.script_args,
        evaluate_fn_factory=components.evaluate_fn_factory,
    )


def execute_grpo_pipeline(
    *,
    component_factory: GrpoComponentFactory,
    reward_funcs: Sequence[Any],
    tokenizer: Any,
    dataset: Mapping[str, Any],
    script_args: Any,
    training_args: Any,
    model_args: Any,
    logger: Any,
    prefix: str,
    evaluate_fn_factory: Optional[EvalFnFactory] = None,
) -> Tuple[Any, Tuple[Any, Optional[Mapping[str, Any]]]]:
    """
    Assemble components via ``component_factory`` and execute the shared GRPO pipeline.

    :returns: Tuple mirroring :func:`prepare_model_eval_and_run_grpo`.
    """

    components = component_factory.build(
        reward_funcs=reward_funcs,
        tokenizer=tokenizer,
        evaluate_fn_factory=evaluate_fn_factory,
    )
    context = build_grpo_context(
        dataset,
        script_args,
        training_args,
        model_args,
        logger,
        prefix=prefix,
    )
    return prepare_model_eval_and_run_grpo(components=components, context=context)


def make_grpo_execute_kwargs(
    *,
    component_factory: GrpoComponentFactory,
    reward_funcs: Sequence[Any],
    tokenizer: Any,
    dataset: Mapping[str, Any],
    script_args: Any,
    training_args: Any,
    model_args: Any,
    logger: Any,
    prefix: str,
    evaluate_fn_factory: Optional[EvalFnFactory] = None,
) -> Dict[str, Any]:
    """Return keyword arguments for :func:`execute_grpo_pipeline`.

    Small helper used by entrypoints to avoid repeating long keyword argument
    lists, which also helps silence duplicate-code warnings across modules.
    """

    return {
        "component_factory": component_factory,
        "reward_funcs": reward_funcs,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "script_args": script_args,
        "training_args": training_args,
        "model_args": model_args,
        "logger": logger,
        "prefix": prefix,
        "evaluate_fn_factory": evaluate_fn_factory,
    }


GRPO_PIPELINE_SCOPE_KEYS = {
    "component_factory": "COMPONENT_FACTORY",
    "reward_funcs": "reward_fns",
    "tokenizer": "tokenizer",
    "dataset": "dataset",
    "script_args": "script_args",
    "training_args": "training_args",
    "model_args": "model_args",
    "logger": "logger",
}


def collect_grpo_pipeline_kwargs(namespace: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract shared pipeline keyword arguments from ``namespace``.

    Expected to be used with a module's ``locals()`` mapping.
    """

    return {
        param: namespace[source]
        for param, source in GRPO_PIPELINE_SCOPE_KEYS.items()
        if source in namespace
    }


def parse_and_run(
    main_fn: Callable[[Any, Any, Any], None],
    argument_classes: Tuple[type, type, type],
) -> None:
    """Parse CLI arguments using ``argument_classes`` and execute ``main_fn``."""

    if TrlParser is None:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "trl must be installed to parse Open R1 command-line arguments "
            "(pip install trl)."
        )
    parser = TrlParser(argument_classes)
    main_fn(*parser.parse_args_and_config())
