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

"""Lazy optional imports for optional XGBoost dependencies."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import joblib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    joblib = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from sklearn.preprocessing import LabelEncoder  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    LabelEncoder = None  # type: ignore[assignment]

__all__ = ["LabelEncoder", "TfidfVectorizer", "joblib"]
