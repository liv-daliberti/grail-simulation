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

"""Facade module exposing recommendation tree visualisation helpers."""

from __future__ import annotations

from .recommendation_tree.cli import (
    DEFAULT_LABEL_TEMPLATE,
    SESSION_DEFAULT_LABEL_TEMPLATE,
    main,
    parse_args,
)
from .recommendation_tree.io import (
    collect_rows,
    extract_session_rows,
    group_rows_by_session,
    load_cleaned_dataset,
    load_metadata,
    load_trajectories,
    load_tree_csv,
    parse_issue_counts,
    _extract_sequences_from_object,
)
from .recommendation_tree.models import (
    LabelRenderOptions,
    OpinionAnnotation,
    OpinionFieldSpec,
    SafeDict,
    TreeData,
    TreeEdge,
    _extract_opinion_annotation,
    _natural_sort_key,
    _opinion_label,
    format_node_label,
)
from .recommendation_tree import render as _render
from .recommendation_tree.render import PUBLIC_API as _RENDER_PUBLIC_API

# Re-export render helpers without repeating the __all__ definitions.
for _name in _RENDER_PUBLIC_API:
    globals()[_name] = getattr(_render, _name)

__all__ = [
    "DEFAULT_LABEL_TEMPLATE",
    "SESSION_DEFAULT_LABEL_TEMPLATE",
    "OpinionAnnotation",
    "OpinionFieldSpec",
    "SafeDict",
    "TreeData",
    "TreeEdge",
    "LabelRenderOptions",
    "collect_rows",
    "_extract_opinion_annotation",
    "_extract_sequences_from_object",
    "extract_session_rows",
    "group_rows_by_session",
    "load_cleaned_dataset",
    "_natural_sort_key",
    "_opinion_label",
    "format_node_label",
    "load_metadata",
    "load_trajectories",
    "load_tree_csv",
    "main",
    "parse_args",
    "parse_issue_counts",
]
__all__ += list(_RENDER_PUBLIC_API)


if __name__ == "__main__":
    main()
