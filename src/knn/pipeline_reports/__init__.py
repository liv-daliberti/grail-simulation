#!/usr/bin/env python
"""Compatibility package for legacy KNN report imports.

This package preserves the older ``knn.pipeline_reports`` import path by
forwarding to the modern implementations under :mod:`knn.pipeline.reports`.

Individual modules (features, hyperparameter, next_video, opinion, shared)
re-export the corresponding builders from the new location.
"""

from __future__ import annotations

__all__: list[str] = []
