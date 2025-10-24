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

"""Namespace exposing the prompt analytics CLI and programmatic API.

This package wires the prompt analytics entry points through a thin wrapper
so both ``python -m clean_data.prompt`` and direct imports use the same
implementation. Usage of these helpers is governed by the repository's
Apache 2.0 license; refer to LICENSE for details.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["generate_prompt_feature_report", "main"]


def _resolve_cli_attribute(name: str) -> Any:
    """Lazy-load the CLI module to avoid duplicate imports during ``python -m``.

    :param name: Attribute name to resolve from :mod:`clean_data.prompt.cli`.
    :returns: Requested callable or attribute from the CLI module.
    """
    module = import_module("clean_data.prompt.cli")
    return getattr(module, name)


def generate_prompt_feature_report(*args: Any, **kwargs: Any) -> Any:
    """Delegate to :func:`clean_data.prompt.cli.generate_prompt_feature_report`.

    :param args: Positional arguments forwarded to the CLI implementation.
    :param kwargs: Keyword arguments forwarded to the CLI implementation.
    :returns: Result of invoking the CLI helper.
    """

    return _resolve_cli_attribute("generate_prompt_feature_report")(*args, **kwargs)


def main(*args: Any, **kwargs: Any) -> Any:
    """Delegate to :func:`clean_data.prompt.cli.main`.

    :param args: Positional arguments forwarded to the CLI entry point.
    :param kwargs: Keyword arguments forwarded to the CLI entry point.
    :returns: Result of invoking the CLI ``main`` function.
    """

    return _resolve_cli_attribute("main")(*args, **kwargs)
