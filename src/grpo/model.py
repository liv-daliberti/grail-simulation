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

"""Model-loading and generation helpers for GRPO inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

try:
    from transformers import (
        AutoModelForCausalLM as _TRANSFORMERS_CAUSAL_LM_FACTORY,
        AutoTokenizer as _TRANSFORMERS_TOKENIZER_FACTORY,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    _TRANSFORMERS_CAUSAL_LM_FACTORY = None  # type: ignore[assignment]
    _TRANSFORMERS_TOKENIZER_FACTORY = None  # type: ignore[assignment]
    _transformers_import_error = exc
else:  # pragma: no cover - trivial branch
    _transformers_import_error = None

if _TRANSFORMERS_TOKENIZER_FACTORY is not None:
    TokenizerLike = _TRANSFORMERS_TOKENIZER_FACTORY
    ModelLike = _TRANSFORMERS_CAUSAL_LM_FACTORY
else:  # pragma: no cover - exercised when transformers missing
    TokenizerLike = Any  # type: ignore[assignment]
    ModelLike = Any  # type: ignore[assignment]

try:
    import torch as _TORCH_MODULE  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    _TORCH_MODULE = None  # type: ignore[assignment]
    _torch_import_error = exc
else:  # pragma: no cover - trivial branch
    _torch_import_error = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


def _require_torch() -> Any:
    """Return the ``torch`` module or raise a helpful error."""

    if _TORCH_MODULE is None:  # pragma: no cover - exercised when torch missing
        raise ModuleNotFoundError(
            "PyTorch is required for GRPO model utilities. Install 'torch' to enable "
            "model loading or adjust the code to skip model-dependent paths."
        ) from _torch_import_error
    return _TORCH_MODULE


def _require_transformers() -> tuple[Any, Any]:
    """Return the tokenizer/model factories or raise a helpful error."""

    if (
        _TRANSFORMERS_TOKENIZER_FACTORY is None
        or _TRANSFORMERS_CAUSAL_LM_FACTORY is None
    ):  # pragma: no cover
        raise ModuleNotFoundError(
            "The 'transformers' package is required for GRPO model utilities. Install "
            "'transformers' to enable model loading or adjust the code to skip model-dependent "
            "paths."
        ) from _transformers_import_error
    return _TRANSFORMERS_TOKENIZER_FACTORY, _TRANSFORMERS_CAUSAL_LM_FACTORY


@dataclass(frozen=True)
class GenerationSettings:
    """Sampling configuration for GRPO inference.

    :ivar int max_new_tokens: Maximum number of new tokens to generate.
    :ivar float temperature: Sampling temperature controlling randomness.
    :ivar float | None top_p: Optional nucleus sampling threshold.
    """

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float | None = None


def _resolve_dtype(dtype: str | None) -> torch.dtype | None:
    """Convert a string dtype to the corresponding torch dtype.

    :param dtype: Torch dtype string (e.g. ``"float16"``) or ``None``.
    :returns: Torch dtype object or ``None`` for automatic selection.
    :rtype: torch.dtype | None
    :raises ValueError: If ``dtype`` does not map to an attribute on :mod:`torch`.
    """

    module = _require_torch()
    if dtype in (None, "", "auto"):
        return None
    try:
        return getattr(module, str(dtype))
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported torch dtype '{dtype}'.") from exc


def load_tokenizer_and_model(
    model_name_or_path: str,
    *,
    revision: str | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    device_map: str | dict[str, Any] | None = "auto",
) -> tuple[TokenizerLike, ModelLike]:
    """Load the tokenizer and causal LM used for evaluation.

    :param model_name_or_path: Hugging Face model identifier or local path.
    :param revision: Optional model revision (branch, tag, or commit).
    :param dtype: Requested torch dtype string (``"auto"`` uses model defaults).
    :param trust_remote_code: Whether to allow custom model code from the hub.
    :param device_map: Device placement strategy passed to ``from_pretrained``.
    :returns: Tuple containing the tokenizer and loaded causal LM model.
    :rtype: tuple[TokenizerLike, ModelLike]
    """

    tokenizer_cls, model_cls = _require_transformers()

    tokenizer = tokenizer_cls.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = model_cls.from_pretrained(
        model_name_or_path,
        revision=revision,
        torch_dtype=_resolve_dtype(dtype),
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    model.eval()
    return tokenizer, model


def generate_chat_completion(
    model: ModelLike,
    tokenizer: TokenizerLike,
    messages: Sequence[dict[str, str]],
    *,
    settings: GenerationSettings,
    device: torch.device | None = None,
) -> str:
    """Return the decoded completion for ``messages`` using ``model``.

    :param model: Loaded causal LM performing inference.
    :param tokenizer: Tokenizer associated with ``model``.
    :param messages: Chat template messages describing the prompt.
    :param settings: Generation hyperparameters controlling sampling.
    :param device: Optional target device; defaults to the model device.
    :returns: Decoded text completion stripped of special tokens.
    :rtype: str
    """

    if device is None:
        device = model.device  # type: ignore[assignment]

    inputs = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(inputs, dict):
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    else:
        input_ids = inputs.to(device)
        attention_mask = None

    module = _require_torch()
    with module.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            do_sample=settings.temperature > 0.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[:, input_ids.shape[-1]:]
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text
