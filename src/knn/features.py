"""Feature extraction for KNN baselines (TF-IDF & Word2Vec)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from prompt_builder.prompt import build_user_prompt


@dataclass
class Word2VecConfig:
    """Configuration for training or loading Word2Vec embeddings."""

    vector_size: int = 256
    window: int = 5
    min_count: int = 2
    epochs: int = 10
    model_dir: Path = Path("models/knn_word2vec")


def build_tfidf_matrix(texts: Sequence[str]) -> Any:  # pragma: no cover - placeholder
    """Return a TF-IDF sparse matrix for ``texts``.

    The implementation will be ported from the legacy script during the
    refactor.
    """

    raise NotImplementedError("TF-IDF builder will be implemented during refactor")


class Word2VecFeatureBuilder:
    """Create Word2Vec embeddings from viewer prompts."""

    def __init__(self, config: Optional[Word2VecConfig] = None) -> None:
        self.config = config or Word2VecConfig()
        self._model = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def train(self, corpus: Iterable[str]) -> None:
        """Train a Word2Vec model using ``corpus`` of sentences."""

        from gensim.models import Word2Vec  # type: ignore

        sentences = [self._tokenize(text) for text in corpus]
        self._model = Word2Vec(
            sentences=sentences,
            vector_size=self.config.vector_size,
            window=self.config.window,
            min_count=self.config.min_count,
            sg=1,
            epochs=self.config.epochs,
        )
        self.save(self.config.model_dir)

    def load(self, directory: Path) -> None:
        from gensim.models import Word2Vec  # type: ignore

        self._model = Word2Vec.load(str(directory / "word2vec.model"))

    def save(self, directory: Path) -> None:
        if self._model is None:
            raise RuntimeError("Word2Vec model must be trained before saving")
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save(str(directory / "word2vec.model"))

    def encode(self, text: str) -> List[float]:
        """Return an embedding vector for ``text`` using the trained model."""

        if self._model is None:
            raise RuntimeError("Word2Vec model has not been trained/loaded")
        tokens = [t for t in self._tokenize(text) if t in self._model.wv]
        if not tokens:
            return [0.0] * self._model.vector_size
        return list(self._model.wv[tokens].mean(axis=0))


def build_prompt_text(row: dict) -> str:
    """Return the prompt text used to create features for a dataset row."""

    return build_user_prompt(row)


__all__ = [
    "Word2VecConfig",
    "Word2VecFeatureBuilder",
    "build_prompt_text",
    "build_tfidf_matrix",
]
