"""Lightweight integration tests for Grail training and pipeline entrypoints."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from grail import grail as grail_module
from grail import pipeline as pipeline_module


def test_grail_main_executes_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = {"train": []}
    tokenizer = object()
    rewards = ["reward"]

    script_args = SimpleNamespace(dataset_train_split="train", dataset_solution_column="answer")
    training_args = SimpleNamespace(seed=1234, system_prompt="sys", reward_weights=[1.0])
    model_args = SimpleNamespace(model="model")

    monkeypatch.setattr(grail_module, "set_seed", lambda *_: None, raising=False)
    monkeypatch.setattr(
        grail_module,
        "_build_dataset_and_tokenizer",
        lambda *_: (dataset, tokenizer),
        raising=False,
    )
    monkeypatch.setattr(grail_module, "_resolve_reward_functions", lambda *_: rewards, raising=False)
    monkeypatch.setattr(grail_module, "_maybe_enable_gail", lambda *_: False, raising=False)
    monkeypatch.setattr(grail_module, "_adjust_reward_weights", lambda *_: None, raising=False)
    monkeypatch.setattr(grail_module, "_apply_reward_mixer", lambda *_: rewards, raising=False)

    captured = {}

    def fake_make_kwargs(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(**kwargs)

    def fake_execute_grpo_pipeline(*, inputs):
        captured["inputs"] = inputs

    monkeypatch.setattr(grail_module, "make_grpo_execute_kwargs", fake_make_kwargs, raising=False)
    monkeypatch.setattr(grail_module, "execute_grpo_pipeline", fake_execute_grpo_pipeline, raising=False)

    grail_module.main(script_args, training_args, model_args)

    assert "kwargs" in captured and "inputs" in captured

    kwargs = captured["kwargs"]
    assert kwargs["dataset"] is dataset
    assert kwargs["namespace"]["COMPONENT_FACTORY"] is grail_module.COMPONENT_FACTORY

    evaluate_factory = kwargs["evaluate_fn_factory"]
    trainer_calls = []
    env_values = {}

    class _Trainer:
        def evaluate(self):
            trainer_calls.append("evaluate")
            env_values["during"] = os.environ.get("GAIL_EVAL_MODE")
            return {"metric": 1.0}

    monkeypatch.setenv("GAIL_EVAL_MODE", "0")
    evaluate_callable = evaluate_factory(_Trainer())
    result = evaluate_callable()
    assert result == {"metric": 1.0}
    assert trainer_calls == ["evaluate"]
    assert os.environ.get("GAIL_EVAL_MODE") == "0"
    assert env_values.get("during") == "1"
    assert "kwargs" in captured


def test_pipeline_main_injects_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured = {}

    monkeypatch.setattr(pipeline_module, "_repo_root", lambda: tmp_path, raising=False)
    monkeypatch.setattr(
        pipeline_module,
        "_grpo_main",
        lambda args: captured.setdefault("args", args),
        raising=False,
    )

    pipeline_module.main(["--stage", "eval"])

    assert "args" in captured
    args = captured["args"]
    default_out = str(tmp_path / "models" / "grail")

    assert "--out-dir" in args and default_out in args
    assert "--reports-subdir" in args and "grail" in args
    assert "--baseline-label" in args and "GRAIL" in args
    assert "--regenerate-hint" in args and pipeline_module.DEFAULT_REGENERATE_HINT in args
    assert args[-2:] == ["--stage", "eval"]
