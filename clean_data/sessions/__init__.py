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

"""Session log parsing and feature engineering for CodeOcean exports."""

# pylint: disable=duplicate-code

from __future__ import annotations

from .build import build_codeocean_rows, split_dataframe
from .models import AllowlistState, ParticipantIdentifiers, SessionInfo, SessionTiming
from .participants import participant_key
from .slates import (
    build_slate_items,
    derive_next_from_history,
    get_gold_next_id,
    gold_index_from_items,
    load_slate_items,
    normalize_display_orders,
)
from .watch import (
    coerce_session_value,
    lookup_session_value,
    normalize_session_mapping,
)

__all__ = [
    "AllowlistState",
    "ParticipantIdentifiers",
    "SessionInfo",
    "SessionTiming",
    "participant_key",
    "coerce_session_value",
    "normalize_session_mapping",
    "lookup_session_value",
    "load_slate_items",
    "derive_next_from_history",
    "get_gold_next_id",
    "gold_index_from_items",
    "normalize_display_orders",
    "build_slate_items",
    "build_codeocean_rows",
    "split_dataframe",
]

# Backwards-compatible aliases
_participant_key = participant_key
_coerce_session_value = coerce_session_value
_normalize_session_mapping = normalize_session_mapping
_lookup_session_value = lookup_session_value
_load_slate_items = load_slate_items
_derive_next_from_history = derive_next_from_history
_get_gold_next_id = get_gold_next_id
_gold_index_from_items = gold_index_from_items
_normalize_display_orders = normalize_display_orders
_build_slate_items = build_slate_items
_build_codeocean_rows = build_codeocean_rows
_split_dataframe = split_dataframe
