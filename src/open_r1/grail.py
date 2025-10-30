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

"""GRPO training entrypoint with optional discriminator (GAIL-style) shaping.

This variant mirrors the standard GRPO pipeline but can append an auxiliary
reward derived from a learned discriminator when ``$GAIL_USE=1``. The
underlying user prompt construction is shared with ``open_r1.grpo`` through the
``prompt_builder`` utilities.
"""
# pylint: disable=too-many-lines

from __future__ import annotations

import inspect
import logging
import os
import re
import types
from typing import (
    Any,
    Dict,
    Callable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
try:
    import torch  # pylint: disable=import-error
    from torch import nn, optim  # pylint: disable=import-error
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
from transformers import (  # pylint: disable=import-error
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from trl import ModelConfig  # pylint: disable=import-error
from common.data.hf_datasets import DatasetDict
from prompt_builder import as_list_json
from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.dataset_utils import drop_marked_rows, make_slate_validator
from open_r1.example_utils import (
    call_row_to_training_example,
    get_gold_next_id,
    gold_index_from_items,
    load_slate_items,
)
from open_r1.rewards import get_reward_funcs
from open_r1.shared import (
    BASE_TRAIN_KEEP_COLUMNS,
    PASSTHROUGH_FIELDS as _SHARED_PASSTHROUGH_FIELDS,
    collect_passthrough_fields,
    make_grpo_execute_kwargs,
    build_default_component_factory,
    execute_grpo_pipeline,
    parse_and_run,
)
from open_r1.utils import get_dataset, get_tokenizer

logger = logging.getLogger(__name__)

COMPONENT_FACTORY = build_default_component_factory()

_TORCH_FALLBACK_DEVICE = "cpu"


class _TensorStub:
    """Minimal tensor stub to satisfy documentation builds without torch."""

    # pylint: disable=missing-function-docstring

    def __init__(
        self,
        data: Optional[Sequence[float]] = None,
        *,
        device: Any = None,
        **_unused: Any,
    ):
        self._data = list(data) if data is not None else []
        self.device = device or _TORCH_FALLBACK_DEVICE

    def numel(self) -> int:
        return len(self._data)

    def sum(self, dim: int | None = None):
        del dim
        return self

    def unsqueeze(self, dim: int):
        del dim
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def mean(self):
        return self

    def item(self) -> float:
        return float(self._data[0]) if self._data else 0.0

    def to(self, *_args, **_kwargs):
        return self

    def tolist(self) -> List[float]:
        return list(self._data)

    def __mul__(self, _other):
        return self

    def __rmul__(self, _other):
        return self

    def __add__(self, _other):
        return self

    def __radd__(self, _other):
        return self

    def __bool__(self) -> bool:
        return bool(self._data)


def _install_torch_stubs() -> None:
    """Provide lightweight fallbacks when torch is unavailable or mocked."""

    # pylint: disable=global-statement
    global torch, nn, optim  # type: ignore[global-statement]

    class _CudaStub:  # pylint: disable=too-few-public-methods
        CudaError = RuntimeError

        @staticmethod
        def is_available() -> bool:
            return False

    class _ModuleStub:  # pylint: disable=too-few-public-methods
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def register_buffer(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class _ParameterStub:  # pylint: disable=too-few-public-methods
        def __init__(self, value: Any) -> None:
            self.value = value

    class _AdamStub:  # pylint: disable=too-few-public-methods
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def zero_grad(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def step(self) -> None:
            return None

    def _make_tensor(*args: Any, **kwargs: Any) -> _TensorStub:
        data = None
        if args:
            candidate = args[0]
            if isinstance(candidate, int):
                data = [0.0] * candidate
            elif isinstance(candidate, (list, tuple)):
                data = list(candidate)
        return _TensorStub(data, device=kwargs.get("device"))

    torch = types.SimpleNamespace(  # type: ignore[assignment]
        Tensor=_TensorStub,
        tensor=_make_tensor,
        zeros=lambda *args, **kwargs: _TensorStub(
            [0.0] * int(args[0]) if args else [], device=kwargs.get("device")
        ),
        zeros_like=lambda *_args, **_kwargs: _TensorStub(),
        ones=lambda *args, **kwargs: _TensorStub(
            [1.0] * int(args[0]) if args else [], device=kwargs.get("device")
        ),
        ones_like=lambda *_args, **_kwargs: _TensorStub(),
        stack=lambda *_args, **_kwargs: _TensorStub(),
        softmax=lambda *_args, **_kwargs: _TensorStub(),
        no_grad=lambda: (lambda func: func),
        isfinite=lambda *_args, **_kwargs: types.SimpleNamespace(all=lambda: True),
        allclose=lambda *_args, **_kwargs: False,
        cuda=_CudaStub(),
        device=lambda spec: spec,
        log=lambda *_args, **_kwargs: _TensorStub(),
        float32="float32",
    )
    nn = types.SimpleNamespace(
        Module=_ModuleStub,
        Parameter=_ParameterStub,
    )  # type: ignore[assignment]
    optim = types.SimpleNamespace(Adam=_AdamStub)  # type: ignore[assignment]


try:
    if not inspect.isclass(getattr(nn, "Module", None)):  # type: ignore[arg-type]
        raise AttributeError("nn.Module is not a class")
    if not callable(getattr(optim, "Adam", None)):  # type: ignore[arg-type]
        raise AttributeError("optim.Adam is not callable")
    if not hasattr(torch, "cuda") or not hasattr(torch.cuda, "is_available"):
        raise AttributeError("torch.cuda missing expected attributes")
except (NameError, AttributeError):
    _install_torch_stubs()

ANS_RE = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
IDX_ONLY = re.compile(r"^\s*(?:option\s*)?(\d+)\s*$", re.I)
TRAIN_KEEP_COLUMNS = BASE_TRAIN_KEEP_COLUMNS | {"slate_items_with_meta"}
PASSTHROUGH_FIELDS = _SHARED_PASSTHROUGH_FIELDS


def _completion_text(payload: Any) -> str:
    """Extract the assistant text payload from chat-style or raw completion objects."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        return str(payload.get("content", "")).strip()
    if isinstance(payload, list) and payload:
        for message in reversed(payload):
            if isinstance(message, dict) and "content" in message:
                content = str(message.get("content", "")).strip()
                if content:
                    return content
        joined_contents = [
            str(msg.get("content", "")).strip()
            for msg in payload
            if isinstance(msg, dict) and str(msg.get("content", "")).strip()
        ]
        if joined_contents:
            return " ".join(joined_contents)
    return str(payload)


def _parse_index_from_answer_block(text: str) -> Optional[int]:
    """Parse the integer inside <answer> tags; return None when the format is invalid."""
    match = ANS_RE.search(text or "")
    payload = (match.group(1).strip() if match else (text or "").strip())
    match_idx = IDX_ONLY.match(payload)
    if not match_idx:
        return None
    try:
        return int(match_idx.group(1))
    except (TypeError, ValueError):
        return None


def _grail_extra_fields(
    example: Mapping[str, Any],
    _: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return additional metadata fields to attach to each GRPO training example."""

    return {
        "slate_items_with_meta": as_list_json(example.get("slate_items_json")),
        **collect_passthrough_fields(example),
    }


def _render_disc_text(
    viewer: str,
    state_text: str,
    slate_items: List[Dict[str, Any]],
    action_surface: str,
    action_id: Optional[str],
) -> str:
    """Render a discriminator input string that mirrors the policy observation format."""
    show_ids = os.getenv("GRAIL_DISC_SHOW_IDS", "0") == "1"
    names = [
        f"{i}. {(it.get('title') or (it.get('id') if show_ids else '') or '(untitled)')}"
        for i, it in enumerate(slate_items, 1)
    ]
    lines = [
        f"VIEWER: {viewer or '(none)'}",
        "STATE:",
        state_text or "(none)",
        "SLATE (names):",
        *(names if names else ["(none)"]),
    ]
    if show_ids:
        lines.append("SLATE_IDS:")
        id_lines = [
            f"{i}. {(it.get('id') or '(none)')}"
            for i, it in enumerate(slate_items, 1)
        ]
        lines.extend(id_lines or ["(none)"])
        lines.append(f"ACTION_ID: {action_id or '(none)'}")
    lines.append(f"ACTION_NAME: {action_surface or '(none)'}")
    return "\n".join(lines)


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


class OnlineDiscriminator:
    """Lightweight text classifier trained on-policy to supply optional GAIL rewards."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        learning_rate: float = 2e-5,
    ):
        """Initialise the discriminator with the specified pretrained backbone.

        :param model_name: Hugging Face model identifier for the classifier.
        :param device: Torch device where the model should live.
        :param learning_rate: Learning rate used for the discriminator optimiser.
        """
        self._model_name = model_name
        self._learning_rate = learning_rate
        self.device = device

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        cfg = AutoConfig.from_pretrained(model_name, num_labels=2)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=cfg,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.model.to(self.device).train()

        self.opt = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self._sanity_check_embeddings()

    def _sanity_check_embeddings(self) -> None:
        """Reload the model if the embedding matrix was sharded away by HF lazy loading."""
        embeddings = self.model.get_input_embeddings()
        weights = getattr(embeddings, "weight", None)
        is_invalid = weights is None or getattr(weights, "dim", lambda: 0)() != 2
        if is_invalid or getattr(weights, "is_meta", False):
            self._reload_clean()

    def _reload_clean(self) -> None:
        """Hard reset the discriminator weights to keep training numerically stable."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.model = model.to(self.device).train()
        self.opt = optim.AdamW(self.model.parameters(), lr=self._learning_rate)

    @torch.no_grad()
    def prob_positive(self, texts: List[str]) -> np.ndarray:
        """Return the positive-class probability for each input string."""
        if not texts:
            return np.zeros((0,), dtype=np.float32)
        payload = [t if isinstance(t, str) and t.strip() else "[PAD]" for t in texts]

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self.model.eval()
        try:
            batch = self.tok(
                payload, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)

            weights = self.model.get_input_embeddings().weight
            if weights is None or weights.dim() != 2 or weights.is_meta:
                self._reload_clean()
                batch = {k: v.to(self.device) for k, v in batch.items()}

            logits = self.model(**batch).logits
            return logits.softmax(dim=-1)[:, 1].detach().cpu().numpy()
        except (OSError, RuntimeError, ValueError, torch.cuda.CudaError):
            return np.zeros((len(texts),), dtype=np.float32)
        finally:
            self.model.train()

    def train_batch(self, texts: List[str], labels: List[int]) -> Optional[float]:
        """Perform one gradient step on the discriminator; returns the loss for logging."""
        if not texts:
            return None
        payload = [t if isinstance(t, str) and t.strip() else "[PAD]" for t in texts]
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        batch = self.tok(
            payload, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        label_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        out = self.model(**batch, labels=label_tensor)
        loss = out.loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return float(loss.detach().cpu().item())


class RewardContext(NamedTuple):
    """Container for discriminator training data extracted from GRPO completions."""
    policy_text: str
    is_valid: bool
    chosen_index: int
    viewer: str
    state: str
    items: List[Dict[str, Any]]
    gold_id: str
    gold_index: int


class MixerSetup(NamedTuple):
    """Configuration used to initialise :class:`LearnableRewardMixer`."""

    base_reward_fns: Sequence[Any]
    base_weights: Sequence[float]
    initial_mix: Tuple[float, float]


def _ensure_list(value: Any, count: int) -> List[Any]:
    """Return ``value`` as a list, repeating scalars ``count`` times."""

    if isinstance(value, list):
        return value
    return [value] * count


def _safe_int(value: Any, default: int = -1) -> int:
    """Cast ``value`` to ``int`` and fall back to ``default`` on failure."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _context_from_completion(
    completion: Any,
    viewer: Any,
    state: Any,
    *,
    metadata: Mapping[str, Any],
) -> RewardContext:
    """Construct a :class:`RewardContext` from a completion and metadata."""

    safe_items = metadata.get("items")
    safe_items = safe_items if isinstance(safe_items, list) else []
    parsed_idx = _parse_index_from_answer_block(_completion_text(completion))
    is_valid = isinstance(parsed_idx, int) and 1 <= parsed_idx <= len(safe_items)
    chosen_index = parsed_idx if is_valid else -1
    choice = safe_items[chosen_index - 1] if is_valid else {}
    surface = (choice.get("title") or choice.get("id") or "") if choice else ""
    action_id = choice.get("id") if choice else None
    gold_index = _safe_int(metadata.get("gold_index"))
    policy_text = (
        _render_disc_text(
            str(viewer or ""),
            str(state or ""),
            safe_items or [],
            surface,
            action_id,
        )
        if is_valid
        else "[PAD]"
    )
    return RewardContext(
        policy_text=policy_text,
        is_valid=is_valid,
        chosen_index=chosen_index,
        viewer=str(viewer or ""),
        state=str(state or ""),
        items=safe_items or [],
        gold_id=str(metadata.get("gold_id") or "").strip(),
        gold_index=gold_index,
    )


def _build_reward_contexts(
    completions: Sequence[Any],
    kwargs: Dict[str, Any],
) -> List[RewardContext]:
    """Return reward contexts for all completions in a batch."""

    num_completions = len(completions)
    if num_completions == 0:
        return []

    viewer_list = _ensure_list(kwargs.get("viewer_profile") or "", num_completions)
    state_list = _ensure_list(kwargs.get("state_text") or "", num_completions)
    items_list = _ensure_list(kwargs.get("slate_items") or [], num_completions)
    gold_id_list = _ensure_list(kwargs.get("gold_id") or "", num_completions)
    gold_idx_list = _ensure_list(kwargs.get("gold_index") or -1, num_completions)

    metadata_list = [
        {"items": items, "gold_id": gold_id, "gold_index": gold_idx}
        for items, gold_id, gold_idx in zip(items_list, gold_id_list, gold_idx_list)
    ]

    return [
        _context_from_completion(
            completion,
            viewer,
            state,
            metadata=metadata,
        )
        for completion, viewer, state, metadata in zip(
            completions,
            viewer_list,
            state_list,
            metadata_list,
        )
    ]


def _train_discriminator_from_contexts(
    disc: OnlineDiscriminator,
    contexts: Sequence[RewardContext],
) -> None:
    """Train the online discriminator on positive/negative examples."""

    pos_texts: List[str] = []
    pos_labels: List[int] = []
    neg_texts: List[str] = []
    neg_labels: List[int] = []

    for ctx in contexts:
        if ctx.gold_id and ctx.items:
            surface = ""
            for item in ctx.items:
                if ctx.gold_id == (item.get("id") or ""):
                    surface = item.get("title") or item.get("id") or ""
                    break
            if surface:
                pos_texts.append(
                    _render_disc_text(ctx.viewer, ctx.state, ctx.items, surface, ctx.gold_id)
                )
                pos_labels.append(1)

        if ctx.is_valid and ctx.gold_index >= 1 and ctx.chosen_index != ctx.gold_index:
            neg_texts.append(ctx.policy_text)
            neg_labels.append(0)

    payload = pos_texts + neg_texts
    if not payload:
        return

    labels = pos_labels + neg_labels
    try:
        disc.train_batch(payload, labels)
    except (OSError, RuntimeError, ValueError, torch.cuda.CudaError):
        logger.debug("discriminator train_batch failed; continuing without update", exc_info=True)


def _select_disc_device() -> torch.device:
    """Return the torch device used for online discriminator training."""

    override = (os.getenv("GAIL_DEVICE", "").strip() or "").lower()
    device_hint = os.getenv("GAIL_DEVICE", "").strip()
    if override == "cuda":
        local_rank = os.getenv("LOCAL_RANK")
        if local_rank is not None:
            try:
                device_hint = f"cuda:{int(local_rank)}"
            except ValueError:
                device_hint = "cuda:0"
        else:
            device_hint = "cuda:0"
    if device_hint:
        return torch.device(device_hint)
    if torch.cuda.is_available():
        local_rank = os.getenv("LOCAL_RANK")
        if local_rank is not None:
            try:
                rank = int(local_rank)
            except ValueError:
                rank = 0
            return torch.device(f"cuda:{rank}")
        return torch.device("cuda:0")
    return torch.device("cpu")


class LearnableRewardMixer(nn.Module):
    """Combine base and discriminator rewards via trainable mixture weights."""

    def __init__(
        self,
        setup: MixerSetup,
        discriminator_reward_fn: Callable[..., Sequence[float]],
        *,
        learning_rate: float = 5e-2,
    ) -> None:
        """
        :param setup: Base reward configuration containing functions, weights, and mixture.
        :param discriminator_reward_fn: Callable returning discriminator rewards.
        :param learning_rate: Optimiser learning rate for the mixture weights.
        """
        super().__init__()
        if not setup.base_reward_fns:
            raise ValueError("LearnableRewardMixer requires at least one base reward function")

        self.base_reward_fns = tuple(setup.base_reward_fns)
        self.discriminator_reward_fn = discriminator_reward_fn

        base_weights_tensor = torch.tensor(setup.base_weights, dtype=torch.float32)
        if base_weights_tensor.numel() == 0 or not torch.isfinite(base_weights_tensor).all():
            base_weights_tensor = torch.ones(len(self.base_reward_fns), dtype=torch.float32)
        if torch.allclose(base_weights_tensor, torch.zeros_like(base_weights_tensor)):
            base_weights_tensor = torch.ones(len(self.base_reward_fns), dtype=torch.float32)
        base_weights_tensor = base_weights_tensor / base_weights_tensor.sum()

        self.register_buffer("_base_weights", base_weights_tensor, persistent=False)

        alpha_init = float(max(setup.initial_mix[0], 1e-6))
        beta_init = float(max(setup.initial_mix[1], 1e-6))
        logits_init = torch.log(torch.tensor([alpha_init, beta_init], dtype=torch.float32))
        self.logits = nn.Parameter(logits_init)
        self._optim = optim.Adam([self.logits], lr=learning_rate)

        # Expose a friendly name for logging
        self.__name__ = "learnable_reward_mixer"

    @staticmethod
    def _should_train() -> bool:
        """Return whether the mixer should update weights for the current invocation."""
        return (
            os.getenv("GAIL_TRAIN", "1") == "1"
            and os.getenv("GAIL_EVAL_MODE", "0") != "1"
        )

    def _current_weights(self) -> torch.Tensor:
        """Return the simplex-projected mixture weights."""
        return torch.softmax(self.logits, dim=0)

    def current_alpha_beta(self) -> Tuple[float, float]:
        """Return the current alpha/beta weights as floats."""
        weights = self._current_weights().detach().cpu().tolist()
        alpha = float(weights[0]) if weights else 0.5
        beta = float(weights[1]) if len(weights) > 1 else 0.5
        return alpha, beta

    def _base_reward_tensor(
        self,
        completions: Sequence[Any],
        answer: Any,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Return weighted environment rewards as a tensor on the correct device."""

        device = self.logits.device
        size = len(completions)
        if not self.base_reward_fns:
            return torch.zeros(size, dtype=torch.float32, device=device)

        tensors: List[torch.Tensor] = []
        for reward_fn in self.base_reward_fns:
            values = reward_fn(completions, answer, **params)
            if values is None:
                values = [0.0] * size
            tensor = torch.as_tensor(values, dtype=torch.float32, device=device).view(-1)
            if tensor.numel() == 0 and size:
                tensor = torch.zeros(size, dtype=torch.float32, device=device)
            elif tensor.numel() != size and size:
                tensor = torch.zeros(size, dtype=torch.float32, device=device)
            tensors.append(tensor)

        stacked = torch.stack(tensors, dim=0)
        weights = self._base_weights.to(device)
        return torch.matmul(weights, stacked)

    def _disc_reward_tensor(
        self,
        completions: Sequence[Any],
        answer: Any,
        params: Dict[str, Any],
        expected_len: int,
    ) -> torch.Tensor:
        """Return discriminator rewards as a tensor with fallback zero fill."""

        device = self.logits.device
        rewards = self.discriminator_reward_fn(completions, answer, **params)
        if rewards is None:
            rewards = []
        tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(-1)
        if tensor.numel() == 0 and expected_len:
            tensor = torch.zeros(expected_len, dtype=torch.float32, device=device)
        elif tensor.numel() != expected_len and expected_len:
            tensor = torch.zeros(expected_len, dtype=torch.float32, device=device)
        return tensor

    @staticmethod
    def _log_state(
        base_combined: torch.Tensor,
        disc_tensor: torch.Tensor,
        alpha: float,
        beta: float,
    ) -> None:
        """Emit wandb-friendly logging for the current mixer state."""

        logger_fn = globals().get("_wb_log")
        if not callable(logger_fn):
            return

        payload = {
            "reward/mixer/alpha": alpha,
            "reward/mixer/beta": beta,
        }

        if base_combined.numel():
            payload["reward/mixer/base_mean"] = float(base_combined.detach().mean().cpu().item())
        else:
            payload["reward/mixer/base_mean"] = 0.0

        if disc_tensor.numel():
            payload["reward/mixer/disc_mean"] = float(disc_tensor.detach().mean().cpu().item())
        else:
            payload["reward/mixer/disc_mean"] = 0.0

        try:
            logger_fn(payload)
        except (TypeError, ValueError):
            pass

    def forward(self, completions, answer, **kwargs) -> List[float]:
        """Return combined rewards with learnable mixture weights."""
        params = dict(kwargs)
        base_combined = self._base_reward_tensor(completions, answer, params)
        expected_len = base_combined.shape[0] if base_combined.ndim else len(completions)
        disc_tensor = self._disc_reward_tensor(completions, answer, params, expected_len)

        weights = self._current_weights()
        combined = weights[0] * base_combined + weights[1] * disc_tensor

        if self._should_train() and combined.numel() > 0:
            loss = -combined.mean()
            self._optim.zero_grad(set_to_none=True)
            loss.backward()
            self._optim.step()

        alpha, beta = self.current_alpha_beta()
        self._log_state(base_combined, disc_tensor, alpha, beta)

        return combined.detach().cpu().tolist()


def make_gail_reward_fn(disc: Optional[OnlineDiscriminator], alpha: float = 1.0):
    """Wrap discriminator scores so they plug into the GRPO reward interface."""
    def _reward(completions, answer, **kwargs):
        """Return discriminator-based rewards for a batch of completions."""

        del answer
        if disc is None:
            return [0.0] * len(completions)

        train_on = (
            os.getenv("GAIL_TRAIN", "1") == "1"
            and os.getenv("GAIL_EVAL_MODE", "0") != "1"
        )

        contexts = _build_reward_contexts(completions, kwargs)
        if not contexts:
            return [0.0] * len(completions)

        probs = disc.prob_positive([ctx.policy_text for ctx in contexts])

        if train_on:
            _train_discriminator_from_contexts(disc, contexts)

        return [
            float(alpha * prob) if ctx.is_valid else 0.0
            for prob, ctx in zip(probs, contexts)
        ]

    return _reward


def _resolve_reward_functions(script_args: GRPOScriptArguments, tokenizer) -> List[Any]:
    """Load baseline reward functions for GRPO training."""

    try:
        return get_reward_funcs(script_args, _ref_model=None, _tokenizer=tokenizer)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        logger.warning("[rewards] get_reward_funcs failed: %s", exc)
        return []


def _maybe_enable_gail(reward_fns: List[Any]) -> bool:
    """Optionally append a GAIL reward function based on environment variables."""

    use_gail = os.environ.get("GAIL_USE", "1") != "0"
    if not use_gail:
        logger.info("GAIL shaping DISABLED")
        return False

    disc_model = os.environ.get("GAIL_DISC_MODEL", "distilbert-base-uncased")
    disc_device = _select_disc_device()
    disc_lr = float(os.environ.get("GAIL_LR", "2e-5"))
    gail_alpha = float(os.environ.get("GAIL_ALPHA", "1.0"))

    discriminator = OnlineDiscriminator(
        disc_model,
        disc_device,
        learning_rate=disc_lr,
    )
    gail_fn = make_gail_reward_fn(discriminator, alpha=gail_alpha)
    gail_fn.__name__ = "gail_reward"
    reward_fns.append(gail_fn)
    logger.info(
        "GAIL shaping ENABLED (alpha=%.3f, model=%s, device=%s)",
        gail_alpha,
        disc_model,
        str(disc_device),
    )
    return True


def _adjust_reward_weights(
    training_args: GRPOConfig,
    reward_fns: Sequence[Any],
    use_gail: bool,
) -> None:
    """Normalise reward weights and append a GAIL weight when required."""

    weights = getattr(training_args, "reward_weights", None)
    if weights is None:
        if use_gail and len(reward_fns) >= 2:
            gail_weight = float(os.environ.get("GAIL_WEIGHT", "0.5"))
            training_args.reward_weights = [1.0] * (len(reward_fns) - 1) + [gail_weight]
        else:
            training_args.reward_weights = [1.0] * len(reward_fns)
    elif len(weights) != len(reward_fns):
        if use_gail and len(weights) == len(reward_fns) - 1:
            gail_weight = float(os.environ.get("GAIL_WEIGHT", "0.5"))
            training_args.reward_weights = list(weights) + [gail_weight]
        else:
            message = (
                f"reward_weights length ({len(weights)}) != number of rewards "
                f"({len(reward_fns)}). Update YAML or set $GAIL_WEIGHT to auto-extend."
            )
            raise ValueError(message)

    if training_args.reward_weights:
        weights_clean = [max(0.0, float(w)) for w in training_args.reward_weights]
        total = sum(weights_clean) or 1.0
        training_args.reward_weights = [w / total for w in weights_clean]


def _apply_reward_mixer(
    training_args: GRPOConfig,
    reward_fns: List[Any],
    use_gail: bool,
) -> List[Any]:
    """Return reward functions with optional learnable mixer applied."""

    initial_weights = list(training_args.reward_weights or [])
    if not use_gail or not reward_fns:
        logger.info(
            "[grpo] rewards=%s weights=%s",
            [getattr(f, "__name__", f.__class__.__name__) for f in reward_fns],
            initial_weights,
        )
        return reward_fns

    base_reward_fns = tuple(reward_fns[:-1])
    gail_reward_fn = reward_fns[-1]
    if not base_reward_fns:
        logger.warning(
            "[grpo+gail] skipping learnable mixer because no base rewards are configured"
        )
        return reward_fns
    base_names = [getattr(f, "__name__", f.__class__.__name__) for f in base_reward_fns]
    gail_name = getattr(gail_reward_fn, "__name__", gail_reward_fn.__class__.__name__)
    logger.info(
        "[grpo+gail] raw rewards=%s + %s weights=%s",
        base_names,
        gail_name,
        initial_weights,
    )

    base_weights = initial_weights[:-1] if len(initial_weights) >= len(reward_fns) else [1.0]
    beta_init = initial_weights[-1] if initial_weights else 0.5
    alpha_init = sum(base_weights) if base_weights else max(1.0 - beta_init, 1e-6)
    mixer_lr = float(os.environ.get("GRAIL_WEIGHT_LR", "5e-2"))
    mixer = LearnableRewardMixer(
        setup=MixerSetup(
            base_reward_fns=base_reward_fns,
            base_weights=base_weights,
            initial_mix=(alpha_init, beta_init),
        ),
        discriminator_reward_fn=gail_reward_fn,
        learning_rate=mixer_lr,
    )
    alpha0, beta0 = mixer.current_alpha_beta()
    training_args.reward_weights = [1.0]
    logger.info(
        "[grpo+gail] using learnable mixer (alpha=%.4f beta=%.4f lr=%.4f)",
        alpha0,
        beta0,
        mixer_lr,
    )
    return [mixer]


def _build_dataset_and_tokenizer(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
) -> Tuple[DatasetDict, Any]:
    """Return processed dataset and tokenizer for the GAIL pipeline."""

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


def main(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
) -> None:
    """Launch the GRPO + optional GAIL training loop with the configured rewards."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(training_args.seed)

    dataset, tokenizer = _build_dataset_and_tokenizer(script_args, training_args, model_args)
    # Use dataset in a lightweight way to satisfy static analyzers.
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
        """Return an evaluation wrapper that freezes GAIL gradients."""

        def _evaluate_with_gail() -> Mapping[str, Any]:
            os.environ["GAIL_EVAL_MODE"] = "1"
            try:
                return grpo_trainer.evaluate()
            finally:
                os.environ["GAIL_EVAL_MODE"] = "0"

        return _evaluate_with_gail

    execute_grpo_pipeline(
        **make_grpo_execute_kwargs(
            prefix="grail",
            evaluate_fn_factory=_gail_eval_factory,
            dataset=dataset,
            namespace=locals(),
        )
    )


if __name__ == "__main__":
    parse_and_run(main, (GRPOScriptArguments, GRPOConfig, ModelConfig))
