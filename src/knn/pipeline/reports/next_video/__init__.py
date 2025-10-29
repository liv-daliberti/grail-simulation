"""Modular next-video report builders used by the KNN reporting pipeline."""

from .inputs import NextVideoReportInputs
from .report import _build_next_video_report

__all__ = ["NextVideoReportInputs", "_build_next_video_report"]
