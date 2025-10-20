"""KNN index construction and persistence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class KNNIndex:
    """Wrapper for building/saving/loading nearest-neighbour indices."""

    index_dir: Path

    def fit(self, features: Any, labels: Any) -> None:
        raise NotImplementedError("KNN fit to be implemented during refactor")

    def save(self) -> None:
        raise NotImplementedError("KNN save to be implemented during refactor")

    def load(self) -> None:
        raise NotImplementedError("KNN load to be implemented during refactor")

    def predict(self, query_features: Any) -> Any:
        raise NotImplementedError("KNN predict to be implemented during refactor")


__all__ = ["KNNIndex"]
