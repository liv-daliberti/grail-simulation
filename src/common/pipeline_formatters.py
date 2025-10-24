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

"""Formatting helpers used by pipeline reports and logs."""

from __future__ import annotations

from typing import Optional


def format_float(value: float) -> str:
    """Format a floating-point metric with three decimal places."""
    return f"{value:.3f}"


def format_optional_float(value: Optional[float]) -> str:
    """Return ``format_float`` output or an em dash when the value is ``None``."""
    return format_float(value) if value is not None else "—"


def format_delta(value: Optional[float]) -> str:
    """Render a signed delta using ``+0.000`` formatting."""
    return f"{value:+.3f}" if value is not None else "—"


def format_count(value: Optional[int]) -> str:
    """Format integer counts with thousands separators or return an em dash."""
    return f"{value:,}" if value is not None else "—"


def format_ratio(numerator: Optional[int], denominator: Optional[int]) -> str:
    """Format ratios as ``hits/total`` when both parts are available."""
    if numerator is None or denominator is None:
        return "—"
    return f"{numerator:,}/{denominator:,}"


def safe_float(value: object) -> Optional[float]:
    """Attempt to convert ``value`` to ``float`` returning ``None`` on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: object) -> Optional[int]:
    """

    Best-effort conversion to int.



    :param value: Value provided for ``value``.

    :type value: object

    :returns: Result produced by ``safe_int``.

    :rtype: Optional[int]

    """


    try:
        return int(value)
    except (TypeError, ValueError):
        return None
