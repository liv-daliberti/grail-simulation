"""Discriminator helpers powering the optional GAIL reward for GRAIL training."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

import numpy as np
from transformers import (  # pylint: disable=import-error
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .grail_torch import nn, optim, torch
from .grail_utils import (
    _completion_text,
    _ensure_list,
    _parse_index_from_answer_block,
    _safe_int,
)

logger = logging.getLogger(__name__)


def _render_disc_text(
    viewer: str,
    state_text: str,
    slate_items: List[Dict[str, Any]],
    action_surface: str,
    action_id: Optional[str],
) -> str:
    """Render a discriminator input string that mirrors the policy observation format.

    :param viewer: Viewer identifier inserted into the rendered prompt.
    :param state_text: Text describing the current environment state.
    :param slate_items: Slate entries supplied by the environment.
    :param action_surface: Human-readable action chosen by the policy.
    :param action_id: Optional identifier of the chosen action.
    :returns: Multi-line discriminator prompt.
    """
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


class OnlineDiscriminator:
    """Lightweight text classifier trained on-policy to supply optional GAIL rewards."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        learning_rate: float = 2e-5,
    ):
        """Initialise the discriminator with the specified pretrained backbone.

        :param model_name: Hugging Face identifier for the discriminator backbone.
        :param device: Torch device that should host the model.
        :param learning_rate: Optimiser learning rate.
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
        """Return the positive-class probability for each input string.

        :param texts: Candidate discriminator prompts.
        :returns: Numpy array containing positive-class probabilities.
        """
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
        """Perform one gradient step on the discriminator.

        :param texts: Prompts that should be scored during training.
        :param labels: Supervision aligned with ``texts``.
        :returns: Training loss or ``None`` when the batch is empty.
        """
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


def _context_from_completion(
    completion: Any,
    viewer: Any,
    state: Any,
    *,
    metadata: Mapping[str, Any],
) -> RewardContext:
    """Construct a :class:`RewardContext` from a completion and metadata.

    :param completion: Raw completion payload emitted by the policy.
    :param viewer: Viewer identifier associated with the completion.
    :param state: Text describing the policy state.
    :param metadata: Additional metadata such as slate items and gold labels.
    :returns: Structured reward context.
    """

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
    """Return reward contexts for all completions in a batch.

    :param completions: Completions produced by the policy.
    :param kwargs: Additional metadata emitted by the GRPO forward pass.
    :returns: List of :class:`RewardContext` instances.
    """

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
    """Train the online discriminator on positive/negative examples.

    :param disc: Online discriminator to update.
    :param contexts: Reward contexts extracted from completions.
    :returns: ``None``. Operates via side effects.
    """

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
    """Return the torch device used for online discriminator training.

    :returns: Torch device derived from environment hints or hardware availability.
    """

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


def make_gail_reward_fn(disc: Optional[OnlineDiscriminator], alpha: float = 1.0):
    """Wrap discriminator scores so they plug into the GRPO reward interface.

    :param disc: Online discriminator producing ``prob_positive`` scores.
    :param alpha: Scaling factor applied to discriminator probabilities.
    :returns: Callable compatible with the GRPO reward protocol.
    """

    def _reward(completions, answer, **kwargs):
        """Return discriminator-based rewards for a batch of completions.

        :param completions: Policy completions that should be scored.
        :param answer: Reference answer forwarded by the GRPO trainer (unused).
        :param kwargs: Additional metadata forwarded by the trainer.
        :returns: List of rewards aligned with ``completions``.
        """

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


__all__ = [
    "OnlineDiscriminator",
    "RewardContext",
    "_render_disc_text",
    "_context_from_completion",
    "_build_reward_contexts",
    "_train_discriminator_from_contexts",
    "_select_disc_device",
    "make_gail_reward_fn",
]
