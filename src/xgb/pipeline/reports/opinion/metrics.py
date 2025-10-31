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

"""Utility helpers shared across opinion report submodules."""

from __future__ import annotations

from typing import List, Optional


def _difference(baseline: Optional[float], value: Optional[float]) -> Optional[float]:
    """
    Return ``baseline - value`` when both inputs are present.

    :param baseline: Baseline metric value.
    :param value: Final metric value.
    :returns: Difference or ``None`` if either input is missing.
    """

    if baseline is None or value is None:
        return None
    return baseline - value


def _append_if_not_none(values: List[float], value: Optional[float]) -> None:
    """
    Append ``value`` to ``values`` when the value is not ``None``.

    :param values: Mutable list receiving numeric values.
    :param value: Candidate value to append.
    :returns: ``None``. Appends to ``values`` when appropriate.
    """

    if value is not None:
        values.append(value)


def _append_difference(
    values: List[float],
    baseline: Optional[float],
    value: Optional[float],
) -> None:
    """
    Append ``baseline - value`` to ``values`` when both numbers exist.

    :param values: Mutable list receiving numeric deltas.
    :param baseline: Baseline metric value.
    :param value: Updated metric value.
    :returns: ``None``. Appends the computed delta when valid.
    """

    delta_value = _difference(baseline, value)
    if delta_value is not None:
        values.append(delta_value)


__all__ = [
    "_difference",
    "_append_if_not_none",
    "_append_difference",
]
