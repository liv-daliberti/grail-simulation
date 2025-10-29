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

"""Helper utilities for installing intra-package module aliases."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Mapping


def install_package_aliases(package_name: str, aliases: Mapping[str, str]) -> None:
    """Register module aliases relative to ``package_name``."""
    package_module = sys.modules[package_name]
    for alias, target in aliases.items():
        module = import_module(target, package_name)
        sys.modules[f"{package_name}.{alias}"] = module
        setattr(package_module, alias, module)


__all__ = ["install_package_aliases"]
