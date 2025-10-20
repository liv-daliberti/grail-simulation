"""Analysis utilities replicating the PNAS filter-bubble study findings.

This package builds heatmaps and summary statistics that mirror the
short-term opinion shifts reported in *Short-term exposure to filter-bubble
recommendation systems has limited polarization effects: Naturalistic
experiments on YouTube* (PNAS, 2025).
"""

from .report import generate_research_article_report

__all__ = ["generate_research_article_report"]
