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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

from __future__ import annotations

from typing import Dict, Set

TRUE_STRINGS: Set[str] = {"1", "true", "t", "yes", "y"}
FALSE_STRINGS: Set[str] = {"0", "false", "f", "no", "n"}

YT_FREQ_MAP = {
    "0": "rarely",
    "1": "occasionally",
    "2": "a few times a month",
    "3": "weekly",
    "4": "several times a week",
    "5": "daily",
    "6": "multiple times per day",
}

LANGUAGE_FRIENDLY_NAMES = {
    "en": "English",
    "en-us": "English",
    "en_us": "English",
    "english": "English",
    "es": "Spanish",
    "es-mx": "Spanish",
    "es_mx": "Spanish",
    "spanish": "Spanish",
    "fr": "French",
    "fr-fr": "French",
    "fr-ca": "French",
    "fr_ca": "French",
}

NEWS_TRUST_FIELD_NAMES = (
    "news_trust",
    "trust_majornews_w1",
    "trust_localnews_w1",
    "trust_majornews_w2",
    "trust_localnews_w2",
    "trust_majornews_w3",
    "trust_localnews_w3",
)

GUN_FIELD_LABELS: Dict[str, str] = {
    "right_to_own_importance": "Right-to-own importance",
    "assault_ban": "Supports assault weapons ban",
    "handgun_ban": "Supports handgun ban",
    "concealed_safe": "Believes concealed carry is safe",
    "stricter_laws": "Supports stricter gun laws",
    "gun_index": "Gun index",
    "gun_index_2": "Gun index (alt)",
    "gun_enthusiasm": "Gun enthusiasm",
    "gun_importance": "Gun importance",
    "gun_priority": "Gun policy priority",
    "gun_policy": "Gun policy stance",
    "gun_identity": "Gun identity",
}

MIN_WAGE_FIELD_LABELS: Dict[str, str] = {
    "minwage_text_r_w1": "Minimum wage stance (wave 1, inferred)",
    "minwage_text_r_w2": "Minimum wage stance (wave 2, inferred)",
    "minwage_text_r_w3": "Minimum wage stance (wave 3, inferred)",
    "minwage_text_w1": "Minimum wage stance (wave 1, survey)",
    "minwage_text_w2": "Minimum wage stance (wave 2, survey)",
    "mw_index_w1": "Minimum wage support index (wave 1)",
    "mw_index_w2": "Minimum wage support index (wave 2)",
    "minwage15_w1": "$15 minimum wage support (wave 1)",
    "minwage15_w2": "$15 minimum wage support (wave 2)",
    "mw_support_w1": "Supports wage increase (wave 1)",
    "mw_support_w2": "Supports wage increase (wave 2)",
    "minwage_importance": "Minimum wage importance",
    "minwage_priority": "Minimum wage priority",
}

__all__ = [
    "FALSE_STRINGS",
    "GUN_FIELD_LABELS",
    "LANGUAGE_FRIENDLY_NAMES",
    "MIN_WAGE_FIELD_LABELS",
    "TRUE_STRINGS",
    "YT_FREQ_MAP",
]
