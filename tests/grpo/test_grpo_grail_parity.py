#!/usr/bin/env python
# pylint: disable=missing-function-docstring
"""Parity checks between GRPO and GRAIL training setups."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("transformers")

import grail.grail_rewards as grail_rewards


def _load_config(name: str) -> dict:
    root = Path(__file__).resolve().parents[2]
    path = root / "recipes" / "Qwen2.5-1.5B-Instruct" / "grpo" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("grpo_name", "grail_name"),
    [
        ("config_grpo_gun.yaml", "config_grail_gun.yaml"),
        ("config_grpo_wage.yaml", "config_grail_wage.yaml"),
    ],
)
def test_config_grail_variants_match_grpo_except_outputs(grpo_name, grail_name):
    grpo_cfg = _load_config(grpo_name)
    grail_cfg = _load_config(grail_name)

    ignore = {"output_dir", "hub_model_id"}
    filtered_grpo = {k: v for k, v in grpo_cfg.items() if k not in ignore}
    filtered_grail = {k: v for k, v in grail_cfg.items() if k not in ignore}

    assert filtered_grail == filtered_grpo


def test_grail_reward_resolver_reuses_grpo_rewards(monkeypatch):
    captured = {}

    def fake_get(script_args, _ref_model=None, _tokenizer=None):  # noqa: D401 - test helper
        captured["args"] = (script_args, _ref_model, _tokenizer)
        return ["pure_accuracy_reward"]

    monkeypatch.setattr(grail_rewards, "get_reward_funcs", fake_get)
    rewards = grail_rewards._resolve_reward_functions("script_args", "tokenizer")

    assert rewards == ["pure_accuracy_reward"]
    assert captured["args"] == ("script_args", None, "tokenizer")


def test_grail_rewards_match_grpo_when_gail_disabled(monkeypatch):
    base_rewards = ["pure_accuracy_reward"]
    monkeypatch.setenv("GAIL_USE", "0")

    rewards = base_rewards.copy()
    use_gail = grail_rewards._maybe_enable_gail(rewards)

    assert use_gail is False
    assert rewards == base_rewards

    args = SimpleNamespace(reward_weights=None)
    grail_rewards._adjust_reward_weights(args, rewards, use_gail)
    assert args.reward_weights == [1.0]
