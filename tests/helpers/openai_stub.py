"""Stub for :mod:`openai` exposing the minimal surface used in tests."""

from __future__ import annotations

import sys


def ensure_openai_stub() -> None:
    """Register a minimal Azure OpenAI client stub."""

    if "openai" in sys.modules:  # pragma: no cover - dependency already available
        return

    module = type(sys)("openai")

    class AzureOpenAI:  # pragma: no cover - simple container
        def __init__(self, **_kwargs) -> None:
            pass

    module.AzureOpenAI = AzureOpenAI
    sys.modules["openai"] = module
