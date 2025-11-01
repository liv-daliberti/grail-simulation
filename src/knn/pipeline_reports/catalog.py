#!/usr/bin/env python
"""Compatibility shim exposing the catalog report builder.

Forwards to :mod:`knn.pipeline.reports.catalog`.
"""

from __future__ import annotations

from knn.pipeline.reports.catalog import _build_catalog_report  # type: ignore[F401]

__all__ = ["_build_catalog_report"]
