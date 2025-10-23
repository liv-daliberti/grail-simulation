#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO training entrypoint with optional discriminator (GAIL-style) shaping.

This variant mirrors the standard GRPO pipeline but can append an auxiliary
reward derived from a learned discriminator when $GAIL_USE=1. The underlying
user prompt construction is shared with `open_r1.grpo` through the
`prompt_builder` utilities.
"""

from __future__ import annotations

import logging
import os
import re
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
from torch import nn, optim
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from trl import ModelConfig, get_peft_config
from trl.trainer.grpo_trainer import GRPOTrainer

from prompt_builder import as_list_json
from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.dataset_utils import drop_marked_rows, slate_has_gold
from open_r1.example_utils import (
    get_gold_next_id,
    gold_index_from_items,
    load_slate_items,
    row_to_training_example,
)
from open_r1.rewards import get_reward_funcs
from open_r1.shared import (
    BASE_TRAIN_KEEP_COLUMNS,
    collect_passthrough_fields,
    configure_eval as shared_configure_eval,
    parse_and_run,
    prepare_eval_dataset as shared_prepare_eval_dataset,
    resolve_checkpoint as shared_resolve_checkpoint,
)
from open_r1.utils import get_dataset, get_model, get_tokenizer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from datasets import DatasetDict
else:  # pragma: no cover - fallback for optional dependency
    DatasetDict = Any

logger = logging.getLogger(__name__)

ANS_RE = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
IDX_ONLY = re.compile(r"^\s*(?:option\s*)?(\d+)\s*$", re.I)
TRAIN_KEEP_COLUMNS = BASE_TRAIN_KEEP_COLUMNS | {"slate_items_with_meta"}


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


def _row_to_example(
    ex: Dict[str, Any],
    system_prompt: Optional[str],
    sol_key: Optional[str],
    max_hist: int = 12,
) -> Optional[Dict[str, Any]]:
    """Return the shared GRPO training payload for ``ex`` or ``None`` when invalid."""

    def _extras(example: Mapping[str, Any], _: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        return {
            "slate_items_with_meta": as_list_json(example.get("slate_items_json")),
            **collect_passthrough_fields(example),
        }

    return row_to_training_example(
        ex,
        system_prompt=system_prompt,
        solution_key=sol_key,
        max_history=max_hist,
        passthrough_fn=lambda example: {},
        extra_fields_fn=_extras,
    )


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

        return _row_to_example(example, system_prompt, solution_key, max_hist=max_hist)

    validator = partial(
        slate_has_gold,
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

    def __init__(self, model_name: str, device: torch.device, lr: float = 2e-5):
        """Initialise the discriminator with the specified pretrained backbone.

        :param model_name: Hugging Face model identifier for the classifier.
        :param device: Torch device where the model should live.
        :param lr: Learning rate used for the discriminator optimiser.
        """
        self._model_name = model_name
        self._lr = lr
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

        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
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
        self.opt = optim.AdamW(self.model.parameters(), lr=self._lr)

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
        y = torch.tensor(labels, dtype=torch.long, device=self.device)
        out = self.model(**batch, labels=y)
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


def _context_from_completion(  # pylint: disable=too-many-arguments
    completion: Any,
    viewer: Any,
    state: Any,
    *,
    items: Any,
    gold_id: Any,
    gold_idx: Any,
) -> RewardContext:
    """Construct a :class:`RewardContext` from a completion and metadata."""

    safe_items = items if isinstance(items, list) else []
    parsed_idx = _parse_index_from_answer_block(_completion_text(completion))
    is_valid = isinstance(parsed_idx, int) and 1 <= parsed_idx <= len(safe_items)
    chosen_index = parsed_idx if is_valid else -1
    choice = safe_items[chosen_index - 1] if is_valid else {}
    surface = (choice.get("title") or choice.get("id") or "") if choice else ""
    action_id = choice.get("id") if choice else None
    gold_index = _safe_int(gold_idx)
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
        gold_id=str(gold_id or "").strip(),
        gold_index=gold_index,
    )


def _build_reward_contexts(
    completions: Sequence[Any],
    kwargs: Dict[str, Any],
) -> List[RewardContext]:
    """Return reward contexts for all completions in a batch."""

    n = len(completions)
    if n == 0:
        return []

    viewer_list = _ensure_list(kwargs.get("viewer_profile") or "", n)
    state_list = _ensure_list(kwargs.get("state_text") or "", n)
    items_list = _ensure_list(kwargs.get("slate_items") or [], n)
    gold_id_list = _ensure_list(kwargs.get("gold_id") or "", n)
    gold_idx_list = _ensure_list(kwargs.get("gold_index") or -1, n)

    return [
        _context_from_completion(
            completion,
            viewer,
            state,
            items=items,
            gold_id=gold_id,
            gold_idx=gold_idx,
        )
        for completion, viewer, state, items, gold_id, gold_idx in zip(
            completions,
            viewer_list,
            state_list,
            items_list,
            gold_id_list,
            gold_idx_list,
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

    discriminator = OnlineDiscriminator(disc_model, disc_device, lr=disc_lr)
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
def _resolve_checkpoint(training_args: GRPOConfig) -> Optional[str]:
    """Return checkpoint path for the GAIL pipeline, respecting overrides."""

    return shared_resolve_checkpoint(training_args)


def main(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: ModelConfig,
) -> None:
    """Launch the GRPO + optional GAIL training loop with the configured rewards."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(training_args.seed)

    dataset, tokenizer = _build_dataset_and_tokenizer(script_args, training_args, model_args)
    reward_fns = _resolve_reward_functions(script_args, tokenizer)
    use_gail = _maybe_enable_gail(reward_fns)
    _adjust_reward_weights(training_args, reward_fns, use_gail)

    logger.info(
        "[grpo+gail] rewards=%s weights=%s",
        [getattr(f, "__name__", f.__class__.__name__) for f in reward_fns],
        training_args.reward_weights,
    )

    model = get_model(model_args, training_args)
    model.generation_config.return_dict_in_generate = True
    model.config.return_dict_in_generate = True

    train_split = script_args.dataset_train_split
    eval_ds = shared_prepare_eval_dataset(
        dataset,
        script_args,
        training_args,
        logger=logger,
        prefix="grail",
    )
    shared_configure_eval(training_args, eval_ds, logger=logger, prefix="grail")

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fns,
        train_dataset=dataset[train_split],
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
    )

    last_ckpt = _resolve_checkpoint(training_args)
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    if getattr(training_args, "do_eval", False) and eval_ds is not None:
        # Toggle GAIL into eval mode so discriminator gradients stay frozen during validation.
        os.environ["GAIL_EVAL_MODE"] = "1"
        metrics = trainer.evaluate()
        os.environ["GAIL_EVAL_MODE"] = "0"
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if getattr(training_args, "push_to_hub", False):
        # Use TRL helper so hub_model_id, hub_strategy, etc., from YAML apply automatically.
        trainer.push_to_hub(dataset_name=script_args.dataset_name, tags=["open-r1"])


if __name__ == "__main__":
    parse_and_run(main, (GRPOScriptArguments, GRPOConfig, ModelConfig))
