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

from typing import Optional


def format_float(value: float) -> str:
    """

    Format a floating-point metric with three decimal places.



    :param value: Value provided for ``value``.

    :type value: float

    :returns: Result produced by ``format_float``.

    :rtype: str

    """


    return f"{value:.3f}"


def format_optional_float(value: Optional[float]) -> str:
    """

    Format optional floating-point metrics.



    :param value: Value provided for ``value``.

    :type value: Optional[float]

    :returns: Result produced by ``format_optional_float``.

    :rtype: str

    """


    return format_float(value) if value is not None else "—"


def format_delta(value: Optional[float]) -> str:
    """

    Return a signed delta with three decimal places.



    :param value: Value provided for ``value``.

    :type value: Optional[float]

    :returns: Result produced by ``format_delta``.

    :rtype: str

    """


    return f"{value:+.3f}" if value is not None else "—"


def format_count(value: Optional[int]) -> str:
    """

    Render optional integer counts with thousands separators.



    :param value: Value provided for ``value``.

    :type value: Optional[int]

    :returns: Result produced by ``format_count``.

    :rtype: str

    """


    return f"{value:,}" if value is not None else "—"


def format_ratio(numerator: Optional[int], denominator: Optional[int]) -> str:
    """

    Format ratios as 'hit/total' when both sides are known.



    :param numerator: Value provided for ``numerator``.

    :type numerator: Optional[int]

    :param denominator: Value provided for ``denominator``.

    :type denominator: Optional[int]

    :returns: Result produced by ``format_ratio``.

    :rtype: str

    """


    if numerator is None or denominator is None:
        return "—"
    return f"{numerator:,}/{denominator:,}"


def safe_float(value: object) -> Optional[float]:
    """

    Best-effort conversion to float.



    :param value: Value provided for ``value``.

    :type value: object

    :returns: Result produced by ``safe_float``.

    :rtype: Optional[float]

    """


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
