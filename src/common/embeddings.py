"""Lightweight wrappers around optional embedding backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

__all__ = ["SentenceTransformerConfig", "SentenceTransformerEncoder"]


@dataclass(frozen=True)
class SentenceTransformerConfig:
    """Configuration controlling SentenceTransformer inference."""

    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str | None = None
    batch_size: int = 32
    normalize: bool = True


class SentenceTransformerEncoder:
    """Thin wrapper over ``sentence_transformers`` with lazy loading."""

    def __init__(self, config: SentenceTransformerConfig) -> None:
        """
        Initialise the encoder with runtime configuration.

        Parameters
        ----------
        config:
            Behavioural options used when loading and invoking the model.
        """

        self.config = config
        self._model: SentenceTransformer | None = None  # type: ignore[valid-type]

    def _ensure_model(self) -> SentenceTransformer:
        """Return a loaded SentenceTransformer model."""

        if SentenceTransformer is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install sentence-transformers to use the sentence_transformer feature space."
            )
        if self._model is None:
            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return embeddings for ``texts``."""

        model = self._ensure_model()
        if not texts:
            dim = int(getattr(model, "get_sentence_embedding_dimension", lambda: 0)() or 0)
            return np.zeros((0, dim), dtype=np.float32)
        embeddings = model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def encode_iter(self, texts: Iterable[str]) -> np.ndarray:
        """Encode an iterable of texts by materialising it once."""

        return self.encode(list(texts))

    def embedding_dimension(self) -> int:
        """Return the dimensionality of the produced embeddings."""

        model = self._ensure_model()
        if hasattr(model, "get_sentence_embedding_dimension"):
            return int(model.get_sentence_embedding_dimension())
        sample = self.encode(["sample"])
        return int(sample.shape[1] if sample.ndim == 2 else 0)
