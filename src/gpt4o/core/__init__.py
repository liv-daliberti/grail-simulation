"""Core utilities shared across the GPT-4o tooling.

This module exposes convenient re-exports for the ``gpt4o.core`` submodules
without importing them eagerly at module import time. Submodules are loaded on
first attribute access via :pep:`562`'s module-level ``__getattr__`` hook.
"""

from __future__ import annotations

import importlib
from typing import Any, List

__all__: List[str] = [
    "client",
    "config",
    "conversation",
    "evaluate",
    "helpers",
    "models",
    "runner",
    "settings",
    "titles",
    "utils",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin import proxy
    """Dynamically import and return requested submodules on first access.

    Supports names listed in :data:`__all__`. ``helpers``, ``models``,
    ``runner``, and ``settings`` are resolved from the ``opinion`` subpackage.
    """

    if name not in __all__:
        raise AttributeError(name)
    if name in {"helpers", "models", "runner", "settings"}:
        return importlib.import_module(f"{__name__}.opinion.{name}")
    return importlib.import_module(f"{__name__}.{name}")


def __dir__() -> List[str]:  # pragma: no cover - introspection helper
    return sorted(list(globals().keys()) + __all__)
