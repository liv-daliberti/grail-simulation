"""Very small stub for :mod:`pandas` when the real dependency is unavailable."""

from __future__ import annotations

import sys


def ensure_pandas_stub() -> None:
    """Register a minimal pandas substitute that exposes ``isna``."""

    try:  # pragma: no cover - executed only when pandas exists
        import pandas  # type: ignore
    except ModuleNotFoundError:
        module = type(sys)("pandas")

        def isna(value):
            try:
                return value != value  # NaN check works for floats
            except Exception:
                return value is None

        module.isna = isna  # type: ignore[attr-defined]
        sys.modules["pandas"] = module
