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

"""Compatibility shim forwarding legacy ``prompt_builder`` imports.

The actual prompt-construction utilities now live under the
``prompt_builder`` package; this module re-exports them for downstream
code that still references ``prompt_builder.py`` directly.
"""

from __future__ import annotations

# pylint: skip-file
"""Backward-compatibility shim for :mod:`prompt_builder`.

The original implementation lived in this single module.  The logic now
resides in the ``prompt_builder`` package under ``src/prompt_builder/``.
Importing from here continues to work while downstream code migrates to
the new module layout.
"""

import warnings

warnings.warn(
    (
        "Importing from 'prompt_builder.py' is deprecated; import from the "
        "'prompt_builder' package (e.g. 'from prompt_builder.prompt import "
        "build_user_prompt') instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)

from prompt_builder import (  # noqa: F401 - re-exported for legacy imports
    as_list_json,
    build_user_prompt,
    clean_text,
    is_nanlike,
    render_profile,
    secs,
    synthesize_viewer_sentence,
    truthy,
)

__all__ = [
    "as_list_json",
    "build_user_prompt",
    "clean_text",
    "is_nanlike",
    "render_profile",
    "secs",
    "synthesize_viewer_sentence",
    "truthy",
]
