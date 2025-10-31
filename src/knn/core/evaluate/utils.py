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

"""Utility helpers shared across the KNN evaluation pipeline."""

from __future__ import annotations

import re
from typing import List, Optional

BUCKET_LABELS = ["unknown", "1", "2", "3", "4", "5+"]


def split_tokens(raw: Optional[str]) -> List[str]:
    """
    Return a list of comma-separated tokens with whitespace trimmed.

    :param raw: Raw comma-separated string or ``None`` when no tokens are supplied.
    :returns: Normalised list of non-empty tokens.
    """

    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def bin_nopts(option_count: int) -> str:
    """
    Bucket the number of slate options into reporting-friendly categories.

    :param option_count: Number of options contained in a recommendation slate.
    :returns: One of ``{"1", "2", "3", "4", "5+"}``.
    """

    if option_count <= 1:
        return "1"
    if option_count == 2:
        return "2"
    if option_count == 3:
        return "3"
    if option_count == 4:
        return "4"
    return "5+"


def bucket_from_pos(pos_idx: int) -> str:
    """
    Bucket a 0-based position index into the standard reporting bins.

    :param pos_idx: Zero-based position of the correct item within the retrieved slate.
    :returns: One of :data:`BUCKET_LABELS` describing the position bucket.
    """

    if pos_idx < 0:
        return "unknown"
    if pos_idx == 0:
        return "1"
    if pos_idx == 1:
        return "2"
    if pos_idx == 2:
        return "3"
    if pos_idx == 3:
        return "4"
    return "5+"


def canon(text: str) -> str:
    """
    Canonicalise a text fragment by lowercasing and removing punctuation.

    :param text: Raw text fragment to normalise.
    :returns: Canonical representation suitable for equality comparisons.
    """

    return re.sub(r"[^a-z0-9]+", "", (text or "").lower().strip())


__all__ = [
    "BUCKET_LABELS",
    "bin_nopts",
    "bucket_from_pos",
    "canon",
    "split_tokens",
]
