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

"""Core training, feature extraction, and evaluation utilities for the XGBoost baseline."""

from __future__ import annotations

import os

# Support faster and lighter imports (useful for unit tests that don't need the
# full dependency surface like `knn`). When XGB_CORE_LIGHT_IMPORTS=1, only
# import a minimal subset of submodules to avoid importing `data`/`evaluate`
# which depend on the KNN pipeline.
if os.getenv("XGB_CORE_LIGHT_IMPORTS") == "1":  # pragma: no cover - import-time switch
    from . import model, vectorizers  # type: ignore
else:  # default behaviour preserves full API surface
    from . import (  # type: ignore
        data,
        evaluate,
        features,
        model,
        opinion,
        utils,
        vectorizers,
    )

__all__ = [
    "data",
    "evaluate",
    "features",
    "model",
    "opinion",
    "utils",
    "vectorizers",
]
