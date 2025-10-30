#!/usr/bin/env python
"""Unit tests for the learnable reward mixer used in GRAIL training."""

from __future__ import annotations

from typing import Any, List

import pytest

try:
    from open_r1.grail import LearnableRewardMixer, MixerSetup
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
