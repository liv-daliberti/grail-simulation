#!/usr/bin/env python
"""Unit tests for the learnable reward mixer used in GRAIL training."""

from __future__ import annotations

from typing import Any, List

import pytest

try:
    from grail.grail import LearnableRewardCallable, LearnableRewardMixer, MixerSetup
except ImportError as exc:  # pragma: no cover - optional dependency guard
    pytest.skip(f"Skipping mixer tests: {exc}", allow_module_level=True)


def _constant_reward(value: float):
    def _reward(completions: List[Any], _answer: Any, **_kwargs) -> List[float]:
        return [value] * len(completions)

    return _reward


def test_learnable_reward_mixer_prefers_stronger_signal(monkeypatch):
    """Mixer should learn to emphasise the higher-magnitude reward signal."""
    monkeypatch.setenv("GAIL_TRAIN", "1")
    monkeypatch.setenv("GAIL_EVAL_MODE", "0")

    completions = ["dummy"] * 4
    base_reward_fn = _constant_reward(1.0)
    disc_reward_fn = _constant_reward(0.1)

    mixer = LearnableRewardMixer(
        setup=MixerSetup(
            base_reward_fns=[base_reward_fn],
            base_weights=[1.0],
            initial_mix=(0.5, 0.5),
        ),
        discriminator_reward_fn=disc_reward_fn,
        learning_rate=0.2,
    )

    for _ in range(40):
        result = mixer(completions, None)
        assert len(result) == len(completions)

    alpha, beta = mixer.current_alpha_beta()
    assert alpha > 0.9
    assert beta < 0.1


def test_learnable_reward_mixer_exposes_config():
    """Mixer should expose a minimal config namespace for TRL compatibility."""
    mixer = LearnableRewardMixer(
        setup=MixerSetup(
            base_reward_fns=[_constant_reward(1.0)],
            base_weights=[1.0],
            initial_mix=(0.6, 0.4),
        ),
        discriminator_reward_fn=_constant_reward(0.5),
    )

    assert hasattr(mixer, "config")
    assert getattr(mixer.config, "_name_or_path", "") == "learnable_reward_mixer"


def test_learnable_reward_callable_routes_to_mixer(monkeypatch):
    """Wrapper should adapt mixer to TRL's reward function signature."""
    monkeypatch.setenv("GAIL_TRAIN", "0")  # prevent optimiser stepping noise
    mixer = LearnableRewardMixer(
        setup=MixerSetup(
            base_reward_fns=[_constant_reward(0.5)],
            base_weights=[1.0],
            initial_mix=(0.8, 0.2),
        ),
        discriminator_reward_fn=_constant_reward(0.1),
    )
    wrapper = LearnableRewardCallable(mixer)

    prompts = [["system prompt"]]
    completions = ["hello"]
    rewards = wrapper(prompts=prompts, completions=completions, completion_ids=[[1, 2, 3]], answer="gold")

    assert rewards == pytest.approx(mixer(completions, "gold"))
