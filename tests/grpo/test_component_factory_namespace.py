"""Unit tests ensuring entrypoints supply module-level component factories."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from common.open_r1 import shared as shared_module


@pytest.fixture
def dummy_runtime_args() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    """Provide lightweight argument containers consumed by main functions."""

    script_args = SimpleNamespace()
    training_args = SimpleNamespace(seed=1234, reward_weights=[1.0])
    model_args = SimpleNamespace()
    return script_args, training_args, model_args


def test_grail_main_supplies_component_factory(monkeypatch, dummy_runtime_args):
    """grail.main should populate namespace with COMPONENT_FACTORY for GRPO inputs."""

    from grail import grail as grail_module

    original_collect = shared_module.collect_grpo_pipeline_kwargs
    dataset = {"train": []}
    tokenizer = object()
    reward_fns = [object()]
    script_args, training_args, model_args = dummy_runtime_args

    monkeypatch.setattr(grail_module, "set_seed", lambda _: None)
    monkeypatch.setattr(
        grail_module,
        "_build_dataset_and_tokenizer",
        lambda *_: (dataset, tokenizer),
    )
    monkeypatch.setattr(grail_module, "_resolve_reward_functions", lambda *_: reward_fns)
    monkeypatch.setattr(grail_module, "_maybe_enable_gail", lambda *_: False)
    monkeypatch.setattr(
        grail_module,
        "_adjust_reward_weights",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        grail_module,
        "_apply_reward_mixer",
        lambda _training_args, _reward_fns, _use_gail: reward_fns,
    )

    captured_inputs = {}

    def fake_execute_grpo_pipeline(*, inputs):
        captured_inputs["value"] = inputs

    def wrapped_collect(namespace):
        assert "COMPONENT_FACTORY" in namespace
        assert namespace["COMPONENT_FACTORY"] is grail_module.COMPONENT_FACTORY
        return original_collect(namespace)

    monkeypatch.setattr(shared_module, "collect_grpo_pipeline_kwargs", wrapped_collect)
    monkeypatch.setattr(grail_module, "execute_grpo_pipeline", fake_execute_grpo_pipeline)

    monkeypatch.setattr(grail_module, "logger", logging.getLogger("grail-test"))

    grail_module.main(script_args, training_args, model_args)

    assert "value" in captured_inputs
    assert captured_inputs["value"].component_factory is grail_module.COMPONENT_FACTORY


def test_grpo_main_supplies_component_factory(monkeypatch, dummy_runtime_args):
    """grpo.main should populate namespace with COMPONENT_FACTORY for GRPO inputs."""

    from grpo import grpo as grpo_module

    original_collect = shared_module.collect_grpo_pipeline_kwargs
    dataset = {"train": []}
    tokenizer = object()
    reward_fns = [object()]
    script_args, training_args, model_args = dummy_runtime_args

    monkeypatch.setattr(grpo_module, "_ensure_training_dependencies", lambda: None)
    monkeypatch.setattr(grpo_module, "set_seed", lambda _: None)
    monkeypatch.setattr(grpo_module, "_build_dataset", lambda *_: dataset)
    monkeypatch.setattr(grpo_module, "get_tokenizer", lambda *_: tokenizer)
    monkeypatch.setattr(grpo_module, "_load_reward_functions", lambda *_: reward_fns)
    monkeypatch.setattr(
        grpo_module,
        "_ensure_reward_weights",
        lambda *_args, **_kwargs: None,
    )

    captured_inputs = {}

    def fake_execute_grpo_pipeline(*, inputs):
        captured_inputs["value"] = inputs

    def wrapped_collect(namespace):
        assert "COMPONENT_FACTORY" in namespace
        assert namespace["COMPONENT_FACTORY"] is grpo_module.COMPONENT_FACTORY
        return original_collect(namespace)

    monkeypatch.setattr(shared_module, "collect_grpo_pipeline_kwargs", wrapped_collect)
    monkeypatch.setattr(grpo_module, "execute_grpo_pipeline", fake_execute_grpo_pipeline)

    monkeypatch.setattr(grpo_module, "logger", logging.getLogger("grpo-test"))

    grpo_module.main(script_args, training_args, model_args)

    assert "value" in captured_inputs
    assert captured_inputs["value"].component_factory is grpo_module.COMPONENT_FACTORY
