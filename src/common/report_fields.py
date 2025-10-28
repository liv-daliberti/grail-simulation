"""Common field name bundles used across report CSV writers."""

from __future__ import annotations

# Reusable next-video coverage-related columns that appear in multiple writers.
NEXT_VIDEO_COVERAGE_FIELDS = (
    "coverage",
    "known_hits",
    "known_total",
    "known_availability",
    "avg_probability",
)

__all__ = ["NEXT_VIDEO_COVERAGE_FIELDS"]

