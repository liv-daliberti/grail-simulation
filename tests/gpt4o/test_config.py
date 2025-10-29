"""Unit tests covering configuration helpers for :mod:`gpt4o`."""

from __future__ import annotations

import os

import pytest

from gpt4o import config

pytestmark = pytest.mark.gpt4o


def test_ensure_azure_env_sets_defaults_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing environment variables should default to module constants."""

    monkeypatch.delenv("SANDBOX_API_KEY", raising=False)
    monkeypatch.delenv("SANDBOX_ENDPOINT", raising=False)
    monkeypatch.setattr(config, "SANDBOX_API_KEY", "test-key")
    monkeypatch.setattr(config, "SANDBOX_ENDPOINT", "https://example.test")

    config.ensure_azure_env()

    assert os.environ["SANDBOX_API_KEY"] == "test-key"
    assert os.environ["SANDBOX_ENDPOINT"] == "https://example.test"


def test_ensure_azure_env_respects_existing_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-existing environment variables should not be overwritten."""

    monkeypatch.setenv("SANDBOX_ENDPOINT", "https://already-set")
    monkeypatch.setattr(config, "SANDBOX_ENDPOINT", "https://should-not-overwrite")

    config.ensure_azure_env()

    assert os.environ["SANDBOX_ENDPOINT"] == "https://already-set"
