"""Model and tokenizer factory helpers for Open-R1 training scripts."""

from typing import Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from ..configs import GRPOConfig, SFTConfig


def get_tokenizer(model_args: ModelConfig, training_args: Union[SFTConfig, GRPOConfig]) -> PreTrainedTokenizer:
    """Instantiate the tokenizer requested by the training configuration.

    :param model_args: Configuration describing the base pretrained checkpoint.
    :param training_args: Training hyper-parameters including optional chat template.
    :returns: Loaded and optionally chat-template enriched tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: Union[SFTConfig, GRPOConfig]) -> AutoModelForCausalLM:
    """Instantiate the causal language model used for supervised or GRPO training.

    :param model_args: Model configuration taken from the CLI or config file.
    :param training_args: Training hyper-parameters that decide caching/quantisation.
    :returns: Initialised ``AutoModelForCausalLM`` ready for fine-tuning.
    """
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": torch_dtype,
        "use_cache": not training_args.gradient_checkpointing,
        "device_map": get_kbit_device_map() if quantization_config is not None else None,
        "quantization_config": quantization_config,
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model
