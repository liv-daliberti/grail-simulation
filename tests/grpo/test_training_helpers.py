#!/usr/bin/env python
# pylint: disable=missing-function-docstring
from __future__ import annotations

import types

import pytest

import grpo.grpo as grpo_mod


class _Cfg:
    def __init__(self, reward_weights=None):
        self.reward_weights = reward_weights


def test_ensure_reward_weights_initialises_when_missing() -> None:
    cfg = _Cfg(reward_weights=None)
    grpo_mod._ensure_reward_weights(cfg, [lambda x: x, lambda x: x])
    assert cfg.reward_weights == [1.0, 1.0]


def test_ensure_reward_weights_raises_on_mismatch() -> None:
    cfg = _Cfg(reward_weights=[0.2])
    with pytest.raises(ValueError, match="reward_weights length"):
        grpo_mod._ensure_reward_weights(cfg, [object(), object()])


def test_ensure_reward_weights_normalises_and_clamps() -> None:
    cfg = _Cfg(reward_weights=[-1.0, 3.0])
    grpo_mod._ensure_reward_weights(cfg, [object(), object()])
    assert cfg.reward_weights == [0.0, 1.0]


def test_load_reward_functions_handles_exception(monkeypatch) -> None:
    # Force get_reward_funcs to raise; wrapper should return []
    monkeypatch.setattr(grpo_mod, "get_reward_funcs", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("err")))
    out = grpo_mod._load_reward_functions(types.SimpleNamespace(), object())
    assert out == []

