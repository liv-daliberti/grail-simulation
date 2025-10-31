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

"""Dataset filtering helpers for KNN evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def filter_split_for_issues(split_ds, issues: Sequence[str]):
    """
    Return ``split_ds`` filtered to the requested issue tokens.

    :param split_ds: Dataset split object being filtered.
    :param issues: Iterable of issue identifiers used to filter the dataset.
    :returns: Filtered dataset respecting the requested issues.
    """

    normalized = {token.strip().lower() for token in issues if token.strip()}
    if not normalized:
        return split_ds
    if "issue" not in getattr(split_ds, "column_names", []):
        return split_ds

    def _match_issue(row: Mapping[str, Any]) -> bool:
        """Return ``True`` when the row belongs to the requested issue slice."""

        value = row.get("issue")
        return str(value).strip().lower() in normalized

    return split_ds.filter(_match_issue)


__all__ = ["filter_split_for_issues"]
