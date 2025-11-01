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

"""Visualization utilities for Grail Simulation reports.

This package exposes two public submodules and supports attribute-style access
patterns used by some dynamic importers:

- ``visualization.recommendation_tree_viz`` – convenience facade that
  re-exports the recommendation-tree helpers and provides the ``-m`` entry.
- ``visualization.recommendation_tree`` – internal package with the concrete
  ``cli``, ``io``, ``models``, and ``render`` modules.

Importers that do ``import visualization`` and then access
``visualization.recommendation_tree_viz`` (without an explicit submodule
import) will succeed thanks to the alias installation below.
"""

from __future__ import annotations

from importlib import import_module
import sys

# Prefer the shared alias helper if available, but fall back to a tiny local
# implementation during static-analysis or minimal environments.
try:  # pragma: no cover - exercised indirectly in integration contexts
    from common.import_utils import install_package_aliases  # type: ignore
except ImportError:  # pragma: no cover - fallback for lint-only contexts
    def install_package_aliases(
        package_name: str,
        aliases: dict[str, str],
    ) -> None:  # type: ignore[no-redef]
        """Install alias modules as attributes on the parent package.

        Mirrors the behavior of the shared helper when it's unavailable.
        """
        pkg = sys.modules[package_name]
        for alias, target in aliases.items():
            module = import_module(target, package_name)
            sys.modules[f"{package_name}.{alias}"] = module
            setattr(pkg, alias, module)


_ALIAS_MODULES: dict[str, str] = {
    # Public facade and the underlying package
    "recommendation_tree_viz": ".recommendation_tree_viz",
    "recommendation_tree": ".recommendation_tree",
}

# Install attribute aliases so ``visualization.recommendation_tree*`` resolves
# even when callers only import the top-level package.
install_package_aliases(__name__, _ALIAS_MODULES)


def __getattr__(name: str):  # pragma: no cover - thin import shim
    """Lazy-load direct child submodules on attribute access.

    Helps dynamic attribute lookups like ``visualization.<submodule>`` without
    requiring a separate ``import visualization.<submodule>`` first.
    """
    try:
        module = import_module(f".{name}", __name__)
    except ImportError as exc:  # mirror normal attribute semantics
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
    # Cache on the parent for subsequent lookups
    setattr(sys.modules[__name__], name, module)
    sys.modules[f"{__name__}.{name}"] = module
    return module


__all__ = [
    "recommendation_tree_viz",
    "recommendation_tree",
]

del install_package_aliases, _ALIAS_MODULES
