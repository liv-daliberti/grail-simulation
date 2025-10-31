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
        def from_pretrained(cls, *_args, **_kwargs):
            raise ImportError("transformers is required for OnlineDiscriminator tests")

    def _stub_set_seed(*_args, **_kwargs):
        raise ImportError("transformers set_seed unavailable in test stub")

    def _stub_pipeline(*_args, **_kwargs):
        raise ImportError("transformers pipeline unavailable in test stub")

    stub_module.AutoConfig = _StubFactory
    stub_module.AutoModelForSequenceClassification = _StubFactory
    stub_module.AutoModelForCausalLM = _StubFactory
    stub_module.AutoTokenizer = _StubFactory
    stub_module.pipeline = _stub_pipeline
    stub_module.set_seed = _stub_set_seed
    sys.modules["transformers"] = stub_module

import grail.grail_gail as gail_module
from grail.grail_gail import (
    RewardContext,
    _build_reward_contexts,
    _context_from_completion,
    _select_disc_device,
    _train_discriminator_from_contexts,
    make_gail_reward_fn,
)
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


def test_context_from_completion_handles_edge_cases(monkeypatch):
    monkeypatch.setenv("GRAIL_DISC_SHOW_IDS", "1")

    invalid_ctx = _context_from_completion(
        completion="no numeric answer",
        viewer="viewer-1",
        state="state-1",
        metadata={
            "items": [{"id": "vid-1", "title": "Keep me"}],
            "gold_id": "  vid-1  ",
            "gold_index": "2",
        },
    )
    assert invalid_ctx.policy_text == "[PAD]"
    assert invalid_ctx.is_valid is False
    assert invalid_ctx.chosen_index == -1
    assert invalid_ctx.gold_id == "vid-1"
    assert invalid_ctx.gold_index == 2

    valid_ctx = _context_from_completion(
        "<answer>1</answer>",
        "viewer-2",
        "state-2",
        metadata={
            "items": [{"id": "vid-9"}],
            "gold_id": "vid-9",
            "gold_index": 1,
        },
    )
    assert valid_ctx.is_valid is True
    policy_text = valid_ctx.policy_text
    assert "SLATE_IDS" in policy_text
    assert "1. vid-9" in policy_text
    assert "ACTION_ID: vid-9" in policy_text
    assert "ACTION_NAME: vid-9" in policy_text


class _TrainSpyDisc:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[int]]] = []

    def train_batch(self, texts, labels):
        self.calls.append((list(texts), list(labels)))
        return 0.0


class _EvalOnlyDisc(_DummyDisc):
    def __init__(self) -> None:
        super().__init__()
        self.train_calls = 0

    def train_batch(self, texts, labels):
        self.train_calls += 1
        return super().train_batch(texts, labels)


def test_train_discriminator_from_contexts_builds_payload(monkeypatch):
    disc = _TrainSpyDisc()

    contexts = [
        RewardContext(
            policy_text="policy-valid",
            is_valid=True,
            chosen_index=1,
            viewer="viewer-A",
            state="state-A",
            items=[
                {"id": "vid-1", "title": "Primary"},
                {"id": "vid-2", "title": "Secondary"},
            ],
            gold_id="vid-1",
            gold_index=1,
        ),
        RewardContext(
            policy_text="policy-negative",
            is_valid=True,
            chosen_index=2,
            viewer="viewer-B",
            state="state-B",
            items=[
                {"id": "vid-3", "title": "Third"},
                {"id": "vid-4", "title": "Fourth"},
            ],
            gold_id="vid-3",
            gold_index=1,
        ),
    ]

    _train_discriminator_from_contexts(disc, contexts)

    assert disc.calls, "train_batch should be invoked when positive/negative samples exist"
    texts, labels = disc.calls[0]
    assert labels.count(1) == 2
    assert labels.count(0) == 1
    assert any("ACTION_NAME: Primary" in text for text in texts)
    assert any("ACTION_NAME: Third" in text for text in texts)
    assert any(text == "policy-negative" for text in texts)

    # No additional training should occur when contexts yield no payload.
    _train_discriminator_from_contexts(
        disc,
        [
            RewardContext(
                policy_text="[PAD]",
                is_valid=False,
                chosen_index=-1,
                viewer="viewer-C",
                state="state-C",
                items=[],
                gold_id="",
                gold_index=-1,
            )
        ],
    )
    assert len(disc.calls) == 1


def test_make_gail_reward_fn_skips_training_in_eval_mode(monkeypatch):
    monkeypatch.setenv("GAIL_TRAIN", "1")
    monkeypatch.setenv("GAIL_EVAL_MODE", "1")

    disc = _EvalOnlyDisc()
    reward_fn = make_gail_reward_fn(disc, alpha=1.0)

    rewards = reward_fn(["<answer>1</answer>"], answer=None, **_reward_kwargs())

    assert rewards == pytest.approx([0.25])
    assert disc.train_calls == 0, "Evaluation mode should block discriminator updates"


def test_make_gail_reward_fn_handles_empty_context(monkeypatch):
    monkeypatch.setenv("GAIL_TRAIN", "1")
    monkeypatch.setenv("GAIL_EVAL_MODE", "0")

    disc = _EvalOnlyDisc()
    reward_fn = make_gail_reward_fn(disc, alpha=2.0)

    rewards = reward_fn([], answer=None)

    assert rewards == []
    assert disc.calls == 0
    assert disc.train_calls == 0


class _DummyDevice:
    def __init__(self, spec: str) -> None:
        self._spec = spec
        self.type = spec.split(":", 1)[0]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self._spec


class _DummyCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _DummyTorch:
    def __init__(self, available: bool) -> None:
        self.cuda = _DummyCuda(available)

    def device(self, spec: str) -> _DummyDevice:
        return _DummyDevice(spec)


def test_select_disc_device_prefers_local_rank(monkeypatch):
    monkeypatch.setattr(gail_module, "torch", _DummyTorch(True), raising=False)
    monkeypatch.setenv("GAIL_DEVICE", "cuda")
    monkeypatch.setenv("LOCAL_RANK", "2")

    device = _select_disc_device()

    assert str(device) == "cuda:2"
    assert device.type == "cuda"


def test_select_disc_device_accepts_explicit_hint(monkeypatch):
    monkeypatch.setattr(gail_module, "torch", _DummyTorch(True), raising=False)
    monkeypatch.setenv("GAIL_DEVICE", "cuda:5")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    device = _select_disc_device()

    assert str(device) == "cuda:5"
    assert device.type == "cuda"


def test_select_disc_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(gail_module, "torch", _DummyTorch(False), raising=False)
    monkeypatch.delenv("GAIL_DEVICE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    device = _select_disc_device()

    assert str(device) == "cpu"
    assert device.type == "cpu"


def test_select_disc_device_handles_invalid_rank(monkeypatch):
    monkeypatch.setattr(gail_module, "torch", _DummyTorch(True), raising=False)
    monkeypatch.setenv("GAIL_DEVICE", "cuda")
    monkeypatch.setenv("LOCAL_RANK", "oops")

    device = _select_disc_device()

    assert str(device) == "cuda:0"
    assert device.type == "cuda"
