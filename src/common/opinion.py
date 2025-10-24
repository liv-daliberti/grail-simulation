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

"""Opinion study dataclasses and helpers shared by multiple pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class OpinionSpec:
    """Configuration describing one opinion-study index."""
    key: str
    issue: str
    label: str
    before_column: str
    after_column: str


@dataclass
class OpinionExample:
    """Collapsed participant-level prompt and opinion measurements."""
    participant_id: str
    participant_study: str
    issue: str
    document: str
    before: float
    after: float


DEFAULT_SPECS: Tuple[OpinionSpec, ...] = (
    OpinionSpec(
        key="study1",
        issue="gun_control",
        label="Study 1 – Gun Control (MTurk)",
        before_column="gun_index",
        after_column="gun_index_2",
    ),
    OpinionSpec(
        key="study2",
        issue="minimum_wage",
        label="Study 2 – Minimum Wage (MTurk)",
        before_column="mw_index_w1",
        after_column="mw_index_w2",
    ),
    OpinionSpec(
        key="study3",
        issue="minimum_wage",
        label="Study 3 – Minimum Wage (YouGov)",
        before_column="mw_index_w1",
        after_column="mw_index_w2",
    ),
)


def float_or_none(value: Any) -> Optional[float]:
    """Return ``value`` as ``float`` when possible, otherwise ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def opinion_example_kwargs(  # pylint: disable=too-many-arguments
    *,
    participant_id: str,
    participant_study: str,
    issue: str,
    document: str,
    before: float,
    after: float,
) -> dict[str, object]:
    """

    Return keyword arguments common to opinion example dataclasses.



    :param participant_id: Value provided for ``participant_id``.

    :type participant_id: str

    :param participant_study: Value provided for ``participant_study``.

    :type participant_study: str

    :param issue: Value provided for ``issue``.

    :type issue: str

    :param document: Value provided for ``document``.

    :type document: str

    :param before: Value provided for ``before``.

    :type before: float

    :param after: Value provided for ``after``.

    :type after: float

    :returns: Result produced by ``opinion_example_kwargs``.

    :rtype: dict[str, object]

    """


    return {
        "participant_id": participant_id,
        "participant_study": participant_study,
        "issue": issue,
        "document": document,
        "before": before,
        "after": after,
    }


__all__ = [
    "DEFAULT_SPECS",
    "OpinionExample",
    "OpinionSpec",
    "float_or_none",
    "opinion_example_kwargs",
]
