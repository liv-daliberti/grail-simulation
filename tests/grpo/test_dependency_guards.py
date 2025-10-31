"""Tests for optional dependency guards ensuring actionable errors."""

from __future__ import annotations

import pytest

from grpo import grpo as grpo_module


def test_grpo_ensure_training_dependencies_requires_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grpo_module, "set_seed", None, raising=False)

    with pytest.raises(ImportError) as excinfo:
        grpo_module._ensure_training_dependencies()

    assert "transformers must be installed" in str(excinfo.value)


def test_grpo_ensure_training_dependencies_requires_trl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grpo_module, "set_seed", lambda *_: None, raising=False)
    monkeypatch.setattr(grpo_module, "ModelConfig", None, raising=False)
    monkeypatch.setattr(grpo_module, "get_peft_config", None, raising=False)
    monkeypatch.setattr(grpo_module, "GRPOTrainer", None, raising=False)

    with pytest.raises(ImportError) as excinfo:
        grpo_module._ensure_training_dependencies()

    assert "trl must be installed" in str(excinfo.value)
