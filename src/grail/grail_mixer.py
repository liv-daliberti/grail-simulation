"""Learnable reward mixer used to blend environment and GAIL rewards."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Sequence, Tuple

from common.open_r1.torch_stub_utils import TensorStub  # lightweight type detection
from .grail_torch import nn, optim, torch

logger = logging.getLogger(__name__)


class MixerSetup(NamedTuple):
    """Configuration used to initialise :class:`LearnableRewardMixer`."""

    base_reward_fns: Sequence[Any]
    base_weights: Sequence[float]
    initial_mix: Tuple[float, float]


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
        :returns: ``None``. Initialises learnable mixture parameters and optimiser.
        """
        super().__init__()
        if not setup.base_reward_fns:
            raise ValueError("LearnableRewardMixer requires at least one base reward function")

        self.base_reward_fns = tuple(setup.base_reward_fns)
        self.discriminator_reward_fn = discriminator_reward_fn

        # Normalise base weights with a tensor-aware fallback for stub environments
        base_weights_tensor = torch.tensor(setup.base_weights, dtype=torch.float32)
        if base_weights_tensor.numel() == 0 or not torch.isfinite(base_weights_tensor).all():
            base_weights_tensor = torch.ones(len(self.base_reward_fns), dtype=torch.float32)
        if torch.allclose(base_weights_tensor, torch.zeros_like(base_weights_tensor)):
            base_weights_tensor = torch.ones(len(self.base_reward_fns), dtype=torch.float32)
        base_weights_tensor = base_weights_tensor / base_weights_tensor.sum()

        self.register_buffer("_base_weights", base_weights_tensor, persistent=False)
        # Ensure attribute exists when running with stubbed nn.Module
        if not hasattr(self, "_base_weights"):
            self._base_weights = base_weights_tensor  # type: ignore[assignment]
        # Python-side cache used when running with TensorStub semantics
        total = float(sum(setup.base_weights) or 1.0)
        self._base_weights_py = [float(w) / total for w in setup.base_weights]

        alpha_init = float(max(setup.initial_mix[0], 1e-6))
        beta_init = float(max(setup.initial_mix[1], 1e-6))
        logits_init = torch.log(torch.tensor([alpha_init, beta_init], dtype=torch.float32))
        self.logits = nn.Parameter(logits_init)
        self._optim = optim.Adam([self.logits], lr=learning_rate)
        self._alpha_beta = [
            alpha_init / (alpha_init + beta_init),
            beta_init / (alpha_init + beta_init),
        ]

        # Note: ``config`` is exposed via a read-only property to avoid
        # introducing an extra instance attribute and satisfy pylint's
        # attribute-count limit while still presenting a TRL-compatible shape.

    @property
    def config(self) -> SimpleNamespace:  # pylint: disable=missing-function-docstring
        # Some TRL utilities expect reward callables/modules to expose a
        # lightweight ``config`` with a ``_name_or_path`` attribute. Returning a
        # fresh namespace keeps this adapter dependency‑free and read‑only.
        return SimpleNamespace(_name_or_path="learnable_reward_mixer")

    @staticmethod
    def _should_train() -> bool:
        """Return whether the mixer should update weights for the current invocation.

        :returns: ``True`` when training is enabled and not in eval mode.
        """
        return (
            os.getenv("GAIL_TRAIN", "1") == "1"
            and os.getenv("GAIL_EVAL_MODE", "0") != "1"
        )

    def _current_weights(self) -> torch.Tensor:
        """Return the simplex-projected mixture weights.

        :returns: 2-element tensor of softmax-normalised mixture weights.
        """
        # In stub mode return a simple tuple to keep downstream logic working
        if isinstance(self.logits, TensorStub) or isinstance(
            getattr(self, "_base_weights", None), TensorStub
        ):
            return self._alpha_beta  # type: ignore[return-value]
        return torch.softmax(self.logits, dim=0)

    def current_alpha_beta(self) -> Tuple[float, float]:
        """Return the current alpha/beta weights as floats.

        :returns: Tuple containing the base (alpha) and discriminator (beta) weights.
        """
        mix_weights = self._current_weights()
        if isinstance(mix_weights, (list, tuple)):
            alpha = float(mix_weights[0]) if mix_weights else 0.5
            beta = float(mix_weights[1]) if len(mix_weights) > 1 else 0.5
            return alpha, beta
        weights = mix_weights.detach().cpu().tolist()
        alpha = float(weights[0]) if weights else 0.5
        beta = float(weights[1]) if len(weights) > 1 else 0.5
        return alpha, beta

    # --- Training helpers -------------------------------------------------

    def _stub_train_step(self, completions, answer, params) -> None:
        """Update alpha/beta with a small step using mean rewards (stub mode)."""
        base_means: List[float] = []
        for reward_fn in self.base_reward_fns:
            vals = reward_fn(completions, answer, **params) or []
            mean = (sum(vals) / len(vals)) if vals else 0.0
            base_means.append(mean)
        base_mean = sum(wt * m for wt, m in zip(self._base_weights_py, base_means))
        disc_vals = self.discriminator_reward_fn(completions, answer, **params) or []
        disc_mean = (sum(disc_vals) / len(disc_vals)) if disc_vals else 0.0
        step = 0.05  # small, stable increment
        if base_mean >= disc_mean:
            self._alpha_beta[0] = min(1.0, self._alpha_beta[0] + step)
        else:
            self._alpha_beta[0] = max(0.0, self._alpha_beta[0] - step)
        self._alpha_beta[1] = 1.0 - self._alpha_beta[0]

    def _autograd_train_step(self, combined) -> None:
        """One autograd step on the negative mean combined reward."""
        loss = -combined.mean()
        self._optim.zero_grad(set_to_none=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Can't initialize NVML",
                category=UserWarning,
            )
            loss.backward()
            self._optim.step()

    def _combine_stub_outputs(self, completions, answer, params) -> List[float]:
        """Return combined rewards using python lists in stub mode."""
        size = len(completions)
        alpha, beta = self.current_alpha_beta()
        base_vals = [0.0] * size
        if self.base_reward_fns:
            per_fn: List[Sequence[float]] = []
            for reward_fn in self.base_reward_fns:
                vals = reward_fn(completions, answer, **params) or [0.0] * size
                per_fn.append(vals)
            base_vals = [
                sum(wt * vf[i] for wt, vf in zip(self._base_weights_py, per_fn))
                for i in range(size)
            ]
        disc_vals = self.discriminator_reward_fn(completions, answer, **params) or [0.0] * size
        return [alpha * b + beta * d for b, d in zip(base_vals, disc_vals)]

    def _base_reward_tensor(
        self,
        completions: Sequence[Any],
        answer: Any,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Return weighted environment rewards as a tensor on the correct device.

        :param completions: Policy completions produced by GRPO.
        :param answer: Reference answer forwarded to underlying reward functions.
        :param params: Keyword arguments to pass through to reward functions.
        :returns: Tensor holding the weighted combination of environment rewards.
        """

        device = getattr(self.logits, "device", "cpu")
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
        """Return discriminator rewards as a tensor with fallback zero fill.

        :param completions: Policy completions produced by GRPO.
        :param answer: Reference answer forwarded to the discriminator reward.
        :param params: Additional arguments forwarded to the discriminator reward function.
        :param expected_len: Number of samples expected in the output tensor.
        :returns: Tensor containing discriminator-provided rewards.
        """

        device = getattr(self.logits, "device", "cpu")
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
        """Emit wandb-friendly logging for the current mixer state.

        :param base_combined: Tensor with the aggregated base rewards.
        :param disc_tensor: Tensor with the discriminator rewards.
        :param alpha: Current base reward weight.
        :param beta: Current discriminator weight.
        :returns: ``None``. Logging side effects are attempted best-effort.
        """

        logger_fn = globals().get("_wb_log")
        if not callable(logger_fn):
            parent = sys.modules.get("grail.grail")
            logger_fn = getattr(parent, "_wb_log", None) if parent else None
        if not callable(logger_fn):
            return

        payload = {
            "reward/mixer/alpha": alpha,
            "reward/mixer/beta": beta,
        }

        if base_combined.numel():
            base_mean = float(base_combined.detach().mean().cpu().item())
        else:
            base_mean = 0.0
        payload["reward/mixer/base_mean"] = base_mean

        if disc_tensor.numel():
            disc_mean = float(disc_tensor.detach().mean().cpu().item())
        else:
            disc_mean = 0.0
        payload["reward/mixer/disc_mean"] = disc_mean

        try:
            logger_fn(payload)
        except (TypeError, ValueError):
            logger.debug("wandb logging failed for reward mixer payload", exc_info=True)

    def forward(self, completions, answer, **params):  # type: ignore[override]
        """Return the combined reward tensor for GRPO training.

        :param completions: Policy completions to score.
        :param answer: Reference answer forwarded by the trainer.
        :param params: Additional metadata forwarded by the trainer.
        :returns: List of combined reward values on the CPU.
        """

        expected_len = len(completions)
        base_combined = self._base_reward_tensor(completions, answer, params)
        disc_tensor = self._disc_reward_tensor(completions, answer, params, expected_len)

        mix_weights = self._current_weights()
        combined = mix_weights[0] * base_combined + mix_weights[1] * disc_tensor

        if self._should_train():
            # Use a lightweight update rule in stub environments instead of autograd.
            if isinstance(combined, TensorStub) or isinstance(mix_weights, list):
                self._stub_train_step(completions, answer, params)
            elif hasattr(combined, "numel") and combined.numel() > 0:
                self._autograd_train_step(combined)

        alpha, beta = self.current_alpha_beta()
        self._log_state(base_combined, disc_tensor, alpha, beta)

        # Return a plain list when running with stubs
        if isinstance(combined, TensorStub):
            return self._combine_stub_outputs(completions, answer, params)

        return combined.detach().cpu().tolist()


class LearnableRewardCallable:
    """Adaptor exposing :class:`LearnableRewardMixer` with TRL's reward function interface."""

    def __init__(self, mixer: LearnableRewardMixer) -> None:
        self._mixer = mixer
        self.__name__ = getattr(mixer, "__name__", mixer.__class__.__name__)
        self.config = SimpleNamespace(_name_or_path=self.__name__)

    def __getattr__(self, item: str) -> Any:
        """Fallback attribute access to the underlying mixer for state dict and buffers."""
        return getattr(self._mixer, item)

    def __call__(
        self,
        *,
        prompts: Iterable[Any],
        completions: Sequence[Any],
        completion_ids: Sequence[Sequence[int]] | None = None,
        **reward_kwargs: Any,
    ) -> List[float]:
        """Route TRL reward invocations to the underlying mixer."""
        _ = prompts, completion_ids  # unused in current mixer implementation
        answer = reward_kwargs.pop("answer", None)
        rewards = self._mixer(completions, answer, **reward_kwargs)
        if rewards is None:
            return [0.0] * len(completions)
        if len(rewards) != len(completions):
            return [0.0] * len(completions)
        return rewards


__all__ = [
    "MixerSetup",
    "LearnableRewardMixer",
    "LearnableRewardCallable",
]
