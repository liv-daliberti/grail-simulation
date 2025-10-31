"""Unit tests covering reward weight adjustments and mixer wiring for GRAIL."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

if "transformers" not in sys.modules:  # pragma: no cover - optional dependency stub
    stub_module = types.ModuleType("transformers")

    class _StubFactory:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise ImportError("transformers is required for grail rewards tests")

    def _stub_set_seed(*_args, **_kwargs):
        raise ImportError("transformers set_seed unavailable in test stub")

    stub_module.AutoConfig = _StubFactory
    stub_module.AutoModelForSequenceClassification = _StubFactory
    stub_module.AutoTokenizer = _StubFactory
    stub_module.pipeline = lambda *args, **kwargs: (args, kwargs)  # noqa: ARG005 - stub
    stub_module.set_seed = _stub_set_seed
    sys.modules["transformers"] = stub_module

import grail.grail_rewards as module


def test_adjust_reward_weights_appends_and_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIL_WEIGHT", "0.75")
    training_args = SimpleNamespace(reward_weights=[0.4, 0.6])
    rewards = ["r1", "r2", "gail"]

    module._adjust_reward_weights(training_args, rewards, use_gail=True)

    weights = training_args.reward_weights
    assert len(weights) == 3
    assert pytest.approx(sum(weights), rel=1e-6) == 1.0
    assert weights[-1] == pytest.approx(0.75 / (0.4 + 0.6 + 0.75))


def test_adjust_reward_weights_raises_on_mismatch_without_gail() -> None:
    training_args = SimpleNamespace(reward_weights=[0.3, 0.7])
    rewards = ["r1", "r2", "r3"]

    with pytest.raises(ValueError) as excinfo:
        module._adjust_reward_weights(training_args, rewards, use_gail=False)

    assert "reward_weights length" in str(excinfo.value)


def test_resolve_reward_functions_handles_failures(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    def raising(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "get_reward_funcs", raising, raising=False)
    caplog.set_level("WARNING", logger=module.logger.name)

    rewards = module._resolve_reward_functions("script_args", "tokenizer")

    assert rewards == []
    assert any("get_reward_funcs failed" in record.message for record in caplog.records)


def test_maybe_enable_gail_respects_disable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAIL_USE", "0")

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("OnlineDiscriminator should not be constructed when GAIL is disabled")

    monkeypatch.setattr(module, "OnlineDiscriminator", _fail_if_called, raising=False)

    rewards = ["baseline"]
    enabled = module._maybe_enable_gail(rewards)

    assert enabled is False
    assert rewards == ["baseline"]


class _StubDisc:
    def __init__(self, model_name, device, learning_rate):
        self.model_name = model_name
        self.device = device
        self.learning_rate = learning_rate
        self.calls = []

    def prob_positive(self, texts):
        self.calls.append(("prob", list(texts)))
        return [0.5 for _ in texts]

    def train_batch(self, texts, labels):
        self.calls.append(("train", list(texts), list(labels)))
        return 0.0


def test_maybe_enable_gail_appends_callable_with_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    created = {}

    def fake_disc(model_name, device, learning_rate):
        instance = _StubDisc(model_name, device, learning_rate)
        created["instance"] = instance
        return instance

    monkeypatch.setattr(module, "OnlineDiscriminator", fake_disc, raising=False)
    monkeypatch.setattr(module, "_select_disc_device", lambda: "cuda:7", raising=False)
    monkeypatch.setenv("GAIL_USE", "1")
    monkeypatch.setenv("GAIL_DISC_MODEL", "stub-model")
    monkeypatch.setenv("GAIL_LR", "3e-5")
    monkeypatch.setenv("GAIL_ALPHA", "1.5")
    monkeypatch.setenv("GAIL_TRAIN", "0")
    monkeypatch.setenv("GAIL_EVAL_MODE", "0")

    rewards: list = []
    enabled = module._maybe_enable_gail(rewards)

    assert enabled is True
    assert len(rewards) == 1
    reward_fn = rewards[0]
    assert callable(reward_fn)
    assert getattr(reward_fn, "__name__", "") == "gail_reward"

    instance = created["instance"]
    assert isinstance(instance, _StubDisc)
    assert instance.model_name == "stub-model"
    assert instance.device == "cuda:7"
    assert instance.learning_rate == 3e-5

    kwargs = {
        "viewer_profile": ["viewer-1"],
        "state_text": ["State"],
        "slate_items": [[{"id": "vid-1", "title": "Video"}]],
        "gold_id": ["vid-1"],
        "gold_index": [1],
    }

    rewards_out = reward_fn(["<answer>1</answer>"], answer=None, **kwargs)
    assert rewards_out == pytest.approx([0.75])  # alpha=1.5 * prob 0.5
    assert instance.calls and instance.calls[0][0] == "prob"
    assert not any(call[0] == "train" for call in instance.calls)


class _FakeMixer:
    def __init__(self, *, setup, discriminator_reward_fn, learning_rate):
        self.setup = setup
        self.discriminator_reward_fn = discriminator_reward_fn
        self.learning_rate = learning_rate
        self.current_alpha_beta_calls = 0

    def current_alpha_beta(self):
        self.current_alpha_beta_calls += 1
        return 0.8, 0.2


class _FakeCallable:
    def __init__(self, mixer):
        self.mixer = mixer
        self.__name__ = "fake_callable"
        self.config = SimpleNamespace(_name_or_path=self.__name__)

    def __call__(self, *args, **kwargs):  # pragma: no cover - unused interface shim
        return []


def test_apply_reward_mixer_wraps_with_learnable_callable(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setattr(module, "LearnableRewardMixer", _FakeMixer, raising=False)
    monkeypatch.setattr(module, "LearnableRewardCallable", _FakeCallable, raising=False)
    monkeypatch.setattr(module, "MixerSetup", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)
    monkeypatch.setenv("GRAIL_WEIGHT_LR", "0.123")

    caplog.set_level("INFO", logger=module.logger.name)

    base1 = lambda *args, **kwargs: [0.1]  # noqa: E731 - simple stub
    base2 = lambda *args, **kwargs: [0.2]
    gail_reward = lambda *args, **kwargs: [0.3]

    training_args = SimpleNamespace(reward_weights=[0.3, 0.4, 0.3])

    result = module._apply_reward_mixer(
        training_args,
        [base1, base2, gail_reward],
        use_gail=True,
    )

    assert isinstance(result, list) and len(result) == 1
    wrapper = result[0]
    assert isinstance(wrapper, _FakeCallable)
    assert training_args.reward_weights == [1.0]

    mixer = wrapper.mixer
    assert isinstance(mixer, _FakeMixer)
    assert mixer.learning_rate == 0.123
    assert mixer.setup.initial_mix == (0.7, 0.3)
    assert mixer.setup.base_reward_fns == (base1, base2)
    assert mixer.setup.base_weights == [0.3, 0.4]
    assert mixer.current_alpha_beta_calls == 1

    assert any("using learnable mixer (alpha=0.8000 beta=0.2000" in record.message for record in caplog.records)


def test_apply_reward_mixer_skips_when_no_base_rewards(caplog) -> None:
    caplog.set_level("WARNING", logger=module.logger.name)
    training_args = SimpleNamespace(reward_weights=[0.5])
    solo_reward = lambda *args, **kwargs: [0.1]  # noqa: E731

    result = module._apply_reward_mixer(training_args, [solo_reward], use_gail=True)

    assert result == [solo_reward]
    assert any("skipping learnable mixer" in record.message for record in caplog.records)
