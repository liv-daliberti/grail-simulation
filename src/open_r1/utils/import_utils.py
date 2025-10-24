#!/usr/bin/env python
"""Availability checks for optional third-party integrations."""

from __future__ import annotations

from importlib.util import find_spec


def _is_package_available(package_name: str) -> bool:
    """Return ``True`` when the requested package can be imported."""
    return find_spec(package_name) is not None


# Use same as transformers.utils.import_utils
_e2b_available = _is_package_available("e2b")


def is_e2b_available() -> bool:
    """Return ``True`` when the optional e2b dependency is installed."""

    return _e2b_available


_morph_available = _is_package_available("morphcloud")


def is_morph_available() -> bool:
    """Return ``True`` when the optional morphcloud dependency is installed."""
    return _morph_available
