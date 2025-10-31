"""Dataset preparation utilities for the GRAIL training pipeline."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from trl import ModelConfig  # pylint: disable=import-error
except Exception:  # pragma: no cover - optional dependency
    # Minimal placeholder for import-time; real usage happens in training scripts.
    class ModelConfig:  # type: ignore[no-redef]
        pass

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
    PASSTHROUGH_FIELDS as _SHARED_PASSTHROUGH_FIELDS,
    collect_passthrough_fields,
)
from common.open_r1.utils import get_dataset, get_tokenizer
from prompt_builder import as_list_json

TRAIN_KEEP_COLUMNS = BASE_TRAIN_KEEP_COLUMNS | {"slate_items_with_meta"}
PASSTHROUGH_FIELDS = _SHARED_PASSTHROUGH_FIELDS


def _grail_extra_fields(
    example: Mapping[str, Any],
    _: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return additional metadata fields to attach to each GRPO training example.

    :param example: Raw dataset row containing slate metadata.
    :param _: Ignored additional rows supplied by ``datasets``.
    :returns: Mapping of supplemental fields that should accompany the example.
    """
    return {
        "slate_items_with_meta": as_list_json(example.get("slate_items_json")),
        **collect_passthrough_fields(example),
    }


def _prepare_dataset(
    raw_dataset,
    system_prompt: Optional[str],
    solution_key: Optional[str],
    max_hist: int,
    train_split: str,
):
    """Convert raw rows into GRPO-ready prompts and filter unusable examples.

    :param raw_dataset: Dataset dictionary returned by :func:`get_dataset`.
    :param system_prompt: Optional system prompt inserted ahead of user text.
    :param solution_key: Optional column providing the gold next-video id.
    :param max_hist: Maximum history length to include in viewer prompts.
    :param train_split: Name of the training split used for drop handling.
    :returns: Dataset dictionary with cleaned columns ready for training.
    """

    def _format_example(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the formatted example or ``None`` when the slate is invalid."""

        return call_row_to_training_example(
            example,
            system_prompt=system_prompt,
            solution_key=solution_key,
            max_history=max_hist,
            passthrough_fn=None,
            extra_fields_fn=_grail_extra_fields,
        )

    validator = make_slate_validator(
        load_slate_items=load_slate_items,
        lookup_gold_id=get_gold_next_id,
        resolve_gold_index=gold_index_from_items,
    )
    filtered = raw_dataset.filter(validator, fn_kwargs={"solution_key": solution_key})
    formatted = filtered.map(_format_example, load_from_cache_file=False)

    drop_marked_rows(formatted, train_split)

    for split in list(formatted.keys()):
        drop_cols = [
            name
            for name in formatted[split].column_names
            if name not in TRAIN_KEEP_COLUMNS
        ]
        if drop_cols:
            formatted[split] = formatted[split].remove_columns(drop_cols)

    return formatted


def _build_dataset_and_tokenizer(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
) -> Tuple[DatasetDict, Any]:
    """Return processed dataset and tokenizer for the GAIL pipeline.

    :param script_args: Script arguments describing dataset sources.
    :param training_args: Training configuration used to build prompts.
    :param model_args: Model configuration for resolving tokenisers.
    :returns: Tuple containing the processed dataset and tokenizer.
    """

    raw_dataset = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args, training_args)
    solution_key = getattr(script_args, "dataset_solution_column", None)
    max_hist = int(os.environ.get("GRAIL_MAX_HISTORY", "12") or "12")
    dataset = _prepare_dataset(
        raw_dataset,
        training_args.system_prompt,
        solution_key,
        max_hist,
        script_args.dataset_train_split,
    )
    return dataset, tokenizer


__all__ = [
    "TRAIN_KEEP_COLUMNS",
    "PASSTHROUGH_FIELDS",
    "_grail_extra_fields",
    "_prepare_dataset",
    "_build_dataset_and_tokenizer",
]
