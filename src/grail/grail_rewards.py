"""Reward wiring helpers for the GRAIL GRPO entrypoint."""

from __future__ import annotations

import logging
import os
from typing import Any, List, Sequence

from common.open_r1.rewards import get_reward_funcs
from .grail_gail import OnlineDiscriminator, make_gail_reward_fn, _select_disc_device
from .grail_mixer import LearnableRewardCallable, LearnableRewardMixer, MixerSetup

logger = logging.getLogger(__name__)


def _resolve_reward_functions(script_args, tokenizer) -> List[Any]:
    """Load baseline reward functions for GRPO training.

    :param script_args: GRPO script arguments that describe baseline rewards.
    :param tokenizer: Tokeniser forwarded to reward factories.
    :returns: Sequence of reward callables declared by the configuration.
    """

    try:
        return get_reward_funcs(script_args, _ref_model=None, _tokenizer=tokenizer)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        logger.warning("[rewards] get_reward_funcs failed: %s", exc)
        return []


def _maybe_enable_gail(reward_fns: List[Any]) -> bool:
    """Optionally append a GAIL reward function based on environment variables.

    :param reward_fns: Mutable list of reward functions configured for GRPO.
    :returns: ``True`` when a GAIL reward has been appended.
    """

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
    training_args,
    reward_fns: Sequence[Any],
    use_gail: bool,
) -> None:
    """Normalise reward weights and append a GAIL weight when required.

    :param training_args: Training configuration containing reward weights.
    :param reward_fns: Sequence of reward functions currently active.
    :param use_gail: Whether a GAIL reward is enabled and expects a weight.
    """

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
    training_args,
    reward_fns: List[Any],
    use_gail: bool,
) -> List[Any]:
    """Return reward functions with optional learnable mixer applied.

    :param training_args: Training configuration containing reward weights.
    :param reward_fns: Reward functions configured for GRPO.
    :param use_gail: Whether GAIL shaping is enabled.
    :returns: List of reward callables after applying the learnable mixer when needed.
    """

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
    return [LearnableRewardCallable(mixer)]


__all__ = [
    "_resolve_reward_functions",
    "_maybe_enable_gail",
    "_adjust_reward_weights",
    "_apply_reward_mixer",
]
