#!/usr/bin/env python
"""Tokenizer and model helpers used by the Open-R1 training pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union, Any

from ..configs import GRPOConfig, SFTConfig

if TYPE_CHECKING:  # pragma: no cover
    from transformers import AutoModelForCausalLM, PreTrainedTokenizer
    from trl import ModelConfig


def _require_transformers() -> Any:
    """Import ``transformers`` lazily and raise a helpful error when missing."""

    try:
        import transformers  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for Open-R1 training utilities. "
            "Install it with `pip install transformers`."
        ) from exc
    return transformers


def _require_trl() -> Any:
    """Import ``trl`` lazily and raise a helpful error when missing."""

    try:
        import trl  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "trl is required for Open-R1 training utilities. "
            "Install it with `pip install trl`."
        ) from exc
    return trl


def _require_torch() -> Any:
    """Import ``torch`` lazily and raise a helpful error when missing."""

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch is required for Open-R1 training utilities. "
            "Install it with `pip install torch` (or the appropriate wheel for your platform)."
        ) from exc
    return torch



def get_tokenizer(
    model_args: "ModelConfig",
    training_args: Union[SFTConfig, GRPOConfig],
) -> "PreTrainedTokenizer":
    """Instantiate the tokenizer requested by the training configuration.

    :param model_args: Configuration describing the base pretrained checkpoint.
    :param training_args: Training hyper-parameters including optional chat template.
    :returns: Loaded and optionally chat-template enriched tokenizer instance.
    """

    transformers = _require_transformers()
    auto_tokenizer = transformers.AutoTokenizer

    tokenizer = auto_tokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(
    model_args: "ModelConfig",
    training_args: Union[SFTConfig, GRPOConfig],
) -> "AutoModelForCausalLM":
    """Instantiate the causal language model used for supervised or GRPO training.

    :param model_args: Model configuration taken from the CLI or config file.
    :param training_args: Training hyper-parameters that decide caching/quantisation.
    :returns: Initialised ``AutoModelForCausalLM`` ready for fine-tuning.
    """

    torch = _require_torch()
    transformers = _require_transformers()
    trl = _require_trl()

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = trl.get_quantization_config(model_args)
    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": torch_dtype,
        "use_cache": not training_args.gradient_checkpointing,
        "device_map": trl.get_kbit_device_map() if quantization_config is not None else None,
        "quantization_config": quantization_config,
    }
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model
