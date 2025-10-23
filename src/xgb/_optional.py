"""Optional third-party dependencies used by the XGBoost baseline."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import joblib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    joblib = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from sklearn.preprocessing import LabelEncoder  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    LabelEncoder = None  # type: ignore[assignment]

__all__ = ["LabelEncoder", "TfidfVectorizer", "joblib"]
