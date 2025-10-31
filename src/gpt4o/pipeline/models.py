#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility shim re-exporting GPT-4o pipeline models from ``common``.

Implements lazy attribute loading to avoid unused-import suppressions while
preserving the public surface.
"""

from __future__ import annotations

import importlib
from typing import Any, List

# Public surface of this shim module
__all__: List[str] = [
    "SweepConfig",
    "SweepOutcome",
    "PipelinePaths",
    "coerce_float",
    "parse_config_label",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin re-export
    if name not in __all__:
        raise AttributeError(name)
    module = importlib.import_module("common.pipeline.gpt4o_models")
    return getattr(module, name)


def __dir__() -> List[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)


class _CallableProxy:
    """Callable proxy that forwards construction to the real object at runtime."""

    def __init__(self, target_name: str) -> None:
        self._target_name = target_name

    def __call__(self, *args: object, **kwargs: object) -> Any:
        module = importlib.import_module("common.pipeline.gpt4o_models")
        target = getattr(module, self._target_name)
        return target(*args, **kwargs)

    def resolve(self) -> Any:
        """Return the underlying callable resolved at runtime."""
        module = importlib.import_module("common.pipeline.gpt4o_models")
        return getattr(module, self._target_name)

    def name(self) -> str:
        """Return the fully qualified target name (for introspection)."""
        return self._target_name


class _FuncProxy:
    """Function proxy that resolves the underlying function on each call."""

    def __init__(self, target_name: str) -> None:
        self._target_name = target_name

    def __call__(self, *args: object, **kwargs: object) -> Any:
        module = importlib.import_module("common.pipeline.gpt4o_models")
        func = getattr(module, self._target_name)
        return func(*args, **kwargs)

    def resolve(self) -> Any:
        """Return the underlying function resolved at runtime."""
        module = importlib.import_module("common.pipeline.gpt4o_models")
        return getattr(module, self._target_name)

    def name(self) -> str:
        """Return the target name (for introspection)."""
        return self._target_name


class PipelinePaths:
    """Callable wrapper that forwards to common.pipeline.gpt4o_models.PipelinePaths."""

    def __new__(cls, *args: object, **kwargs: object) -> Any:
        return _CallableProxy("PipelinePaths")( *args, **kwargs)

    @staticmethod
    def resolve() -> Any:
        """Return the real PipelinePaths class from the common module."""
        return _CallableProxy("PipelinePaths").resolve()

    # Properties are declared for linter awareness only; at runtime instances
    # are the underlying common class returned from __new__.
    @property
    def out_dir(self) -> Any:
        """Output root directory path (shim property for linting)."""
        return None

    @property
    def final_out_dir(self) -> Any:
        """Final next-video output directory (shim property)."""
        return None

    @property
    def opinion_dir(self) -> Any:
        """Opinion outputs directory (shim property)."""
        return None

    @property
    def sweep_dir(self) -> Any:
        """Sweep outputs directory (shim property)."""
        return None

    @property
    def reports_dir(self) -> Any:
        """Reports directory path (shim property)."""
        return None

    @property
    def cache_dir(self) -> Any:
        """HF cache directory path (shim property)."""
        return None


class SweepConfig:
    """Callable wrapper that forwards to common.pipeline.gpt4o_models.SweepConfig."""

    def __new__(cls, *args: object, **kwargs: object) -> Any:
        return _CallableProxy("SweepConfig")( *args, **kwargs)

    @staticmethod
    def resolve() -> Any:
        """Return the real SweepConfig class from the common module."""
        return _CallableProxy("SweepConfig").resolve()


class SweepOutcome:
    """Callable wrapper that forwards to common.pipeline.gpt4o_models.SweepOutcome."""

    def __new__(cls, *args: object, **kwargs: object) -> Any:
        return _CallableProxy("SweepOutcome")( *args, **kwargs)

    @staticmethod
    def resolve() -> Any:
        """Return the real SweepOutcome class from the common module."""
        return _CallableProxy("SweepOutcome").resolve()
coerce_float: Any = _FuncProxy("coerce_float")
parse_config_label: Any = _FuncProxy("parse_config_label")
