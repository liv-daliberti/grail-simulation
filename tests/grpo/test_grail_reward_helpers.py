#!/usr/bin/env python
"""Unit tests for helper utilities powering the GAIL reward pipeline."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

if "transformers" not in sys.modules:  # pragma: no cover - optional dependency stub
    stub_module = types.ModuleType("transformers")

    class _StubFactory:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # pylint: disable=unused-argument
            raise ImportError("transformers is required for OnlineDiscriminator tests")

    def _stub_set_seed(*args, **kwargs):  # pylint: disable=unused-argument
        raise ImportError("transformers set_seed unavailable in test stub")

    def _stub_pipeline(*args, **kwargs):  # pylint: disable=unused-argument
        raise ImportError("transformers pipeline unavailable in test stub")

    stub_module.AutoConfig = _StubFactory
    stub_module.AutoModelForSequenceClassification = _StubFactory
    stub_module.AutoModelForCausalLM = _StubFactory
    stub_module.AutoTokenizer = _StubFactory
    stub_module.pipeline = _stub_pipeline
    stub_module.set_seed = _stub_set_seed
    sys.modules["transformers"] = stub_module

from grail.grail_gail import _build_reward_contexts, make_gail_reward_fn
from grail.grail_utils import _completion_text, _parse_index_from_answer_block


def test_completion_text_prefers_last_non_empty_message() -> None:
    payload = [
        {"role": "system", "content": ""},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "  "},
        {"role": "assistant", "content": "final answer"},
    ]
    assert _completion_text(payload) == "final answer"


def test_parse_index_from_answer_block_handles_html_tags() -> None:
    assert _parse_index_from_answer_block("<answer> 3 </answer>") == 3
    assert _parse_index_from_answer_block("<answer>option 2</answer>") == 2
    assert _parse_index_from_answer_block("no answer") is None


def test_build_reward_contexts_extracts_metadata() -> None:
    completions = ["<answer>1</answer>"]
    kwargs = {
        "viewer_profile": ["viewer-123"],
        "state_text": ["State summary"],
        "slate_items": [
            [
                {"id": "vid-1", "title": "First"},
                {"id": "vid-2", "title": "Second"},
            ]
        ],
        "gold_id": ["vid-1"],
        "gold_index": [1],
    }

    contexts = _build_reward_contexts(completions, kwargs)

    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx.is_valid, "Valid index should produce a usable context"
    assert ctx.gold_id == "vid-1"
    assert "ACTION_NAME: First" in ctx.policy_text


class _DummyDisc:
    def __init__(self) -> None:
        self.calls = 0
        self.training_payloads: list[tuple[list[str], list[int]]] = []

    def prob_positive(self, texts):
        self.calls += 1
        return np.asarray([0.25 for _ in texts], dtype=np.float32)

    def train_batch(self, texts, labels):
        self.training_payloads.append((list(texts), list(labels)))
        return 0.1


def _reward_kwargs():
    return {
        "viewer_profile": ["viewer-123", "viewer-456"],
        "state_text": ["State A", "State B"],
        "slate_items": [
            [
                {"id": "vid-1", "title": "First"},
                {"id": "vid-2", "title": "Second"},
            ],
            [
                {"id": "vid-3", "title": "Third"},
                {"id": "vid-4", "title": "Fourth"},
            ],
        ],
        "gold_id": ["vid-1", "vid-3"],
        "gold_index": [1, 1],
    }


def test_make_gail_reward_fn_scales_scores(monkeypatch):
    monkeypatch.setenv("GAIL_TRAIN", "0")
    monkeypatch.setenv("GAIL_EVAL_MODE", "0")

    completions = ["<answer>1</answer>", "<answer>1</answer>"]
    disc = _DummyDisc()
    reward_fn = make_gail_reward_fn(disc, alpha=2.0)

    rewards = reward_fn(completions, answer=None, **_reward_kwargs())

    assert disc.calls == 1
    assert rewards == pytest.approx([0.5, 0.5])
    assert not disc.training_payloads, "Training should be skipped when GAIL_TRAIN=0"


def test_make_gail_reward_fn_trains_when_enabled(monkeypatch):
    monkeypatch.setenv("GAIL_TRAIN", "1")
    monkeypatch.setenv("GAIL_EVAL_MODE", "0")

    completions = ["<answer>2</answer>", "<answer>1</answer>"]
    disc = _DummyDisc()
    reward_fn = make_gail_reward_fn(disc, alpha=1.0)

    rewards = reward_fn(completions, answer=None, **_reward_kwargs())

    assert rewards == pytest.approx([0.25, 0.25])
    assert disc.training_payloads, "Training should run when GAIL_TRAIN=1"
