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

def next_video_coverage_mapping(summary) -> dict:
    """
    Return a dict of next-video coverage fields for CSV writers.

    The ``summary`` object is expected to expose attributes matching the
    entries in :data:`NEXT_VIDEO_COVERAGE_FIELDS`.

    :param summary: Object exposing next-video coverage attributes.
    :returns: Mapping from canonical column name to value extracted from ``summary``.
    """

    return {
        "coverage": getattr(summary, "coverage", None),
        "known_hits": getattr(summary, "known_hits", None),
        "known_total": getattr(summary, "known_total", None),
        "known_availability": getattr(summary, "known_availability", None),
        "avg_probability": getattr(summary, "avg_probability", None),
    }

__all__ = ["NEXT_VIDEO_COVERAGE_FIELDS", "next_video_coverage_mapping"]
