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

"""Utilities for optional matplotlib support across pipelines."""

from __future__ import annotations

from typing import Any, Optional

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as _plt  # type: ignore[assignment]
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None  # type: ignore[assignment]
    _plt = None  # type: ignore[assignment]

plt: Optional[Any] = _plt

__all__ = ["plt", "matplotlib"]
