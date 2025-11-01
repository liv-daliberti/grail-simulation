"""Types shared by the reporting helpers for sample galleries.

This module defines a light-weight, frozen dataclass used by sample collection
and rendering code. It intentionally keeps only core fields as attributes to
avoid pylint's too-many-instance-attributes warning while still exposing
convenient derived properties for report generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class Sample:
    """Container for a single rendered sample.

    Kept minimal and shared across submodules to avoid circular imports.
    """

    issue: str
    task: str  # "next_video" or "opinion"
    question: str
    think: str
    answer: str
    # Optional task-specific enrichments for clearer notes
    before: Optional[float] = None
    predicted_after: Optional[float] = None

    @property
    def chosen_option(self) -> Optional[int]:
        """Best-effort integer parsed from ``answer`` for next-video tasks.

        Returns ``None`` when the answer is empty or not an integer.
        """
        text = (self.answer or "").strip()
        if not text:
            return None
        try:
            return int(text)
        except (ValueError, TypeError):
            return None

    @property
    def opinion_label(self) -> str:
        """Direction label derived from ``before`` and ``predicted_after``.

        Returns one of ``increase``, ``decrease`` or ``no_change``. When values
        are missing or cannot be interpreted as finite floats, ``no_change`` is
        returned for a stable display in reports.
        """
        before_value = (
            float(self.before) if self.before is not None else float("nan")
        )
        after_value = (
            float(self.predicted_after)
            if self.predicted_after is not None
            else float("nan")
        )

        if not (math.isfinite(before_value) and math.isfinite(after_value)):
            return "no_change"

        delta = after_value - before_value
        # Treat tiny differences as no change to reduce noise.
        tol = 1e-6
        if delta > tol:
            return "increase"
        if delta < -tol:
            return "decrease"
        return "no_change"


__all__ = ["Sample"]
