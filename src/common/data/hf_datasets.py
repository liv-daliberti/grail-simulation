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

"""Utilities for optional Hugging Face ``datasets`` imports.

Several entrypoints load evaluation datasets but treat the dependency as
optional so the wider codebase can run without ``datasets`` installed.
Centralising the import logic avoids repeating the same try/except blocks and
reduces the number of duplicate-code warnings from pylint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

try:  # pragma: no cover - optional dependency
    from datasets import DownloadConfig, load_dataset, load_from_disk  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    DownloadConfig = None  # type: ignore[assignment]
    load_dataset = None  # type: ignore[assignment]
    load_from_disk = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from datasets import DatasetDict
else:  # pragma: no cover - fallback when datasets is unavailable
    DatasetDict = Any  # type: ignore

__all__ = [
    "DownloadConfig",
    "load_dataset",
    "load_from_disk",
    "get_dataset_loaders",
    "require_dataset_support",
    "DatasetDict",
]


def get_dataset_loaders() -> Tuple[Any, Any, Any]:
    """Return the trio of dataset helpers (may contain ``None`` placeholders)."""

    return DownloadConfig, load_dataset, load_from_disk


def require_dataset_support(*, needs_local: bool = False) -> None:
    """Raise informative errors when ``datasets`` support is missing.

    :param needs_local: When ``True``, ensure :func:`load_from_disk` is available.
    :raises ImportError: If the required dataset functionality is unavailable.
    """

    if load_dataset is None or DownloadConfig is None:
        raise ImportError(
            "The 'datasets' package is required to run evaluations. "
            "Install it with `pip install datasets`."
        )
    if needs_local and load_from_disk is None:
        raise ImportError(
            "The 'datasets' package with load_from_disk support is required "
            "to load local datasets."
        )
