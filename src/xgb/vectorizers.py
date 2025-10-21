"""Text vectoriser abstractions for the XGBoost baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore[assignment]

from common.embeddings import SentenceTransformerConfig, SentenceTransformerEncoder
from knn.features import Word2VecConfig, Word2VecFeatureBuilder

__all__ = [
    "BaseTextVectorizer",
    "SentenceTransformerConfig",
    "TfidfVectorizerWrapper",
    "Word2VecVectorizer",
    "SentenceTransformerVectorizer",
    "create_vectorizer",
    "load_vectorizer",
]

VECTORISER_META = "vectorizer.json"


class BaseTextVectorizer:
    """Interface for text vectorisers used by the XGBoost slate model."""

    kind: str

    def fit_transform(self, documents: Sequence[str]) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def transform(self, documents: Sequence[str]) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def feature_dimension(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def save(self, directory: Path) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path) -> "BaseTextVectorizer":  # pragma: no cover - abstract
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        """Return serialisable metadata describing the vectoriser."""

        return {"kind": self.kind, "dimension": self.feature_dimension()}


@dataclass
class TfidfConfig:
    """Configuration parameters for TF-IDF vectorisation."""

    max_features: int | None = 200_000


class TfidfVectorizerWrapper(BaseTextVectorizer):
    """Wrapper around scikit-learn's TF-IDF vectoriser."""

    kind = "tfidf"

    def __init__(self, config: TfidfConfig) -> None:
        if TfidfVectorizer is None:  # pragma: no cover - optional dependency
            raise ImportError("Install scikit-learn to use TF-IDF vectorisation.")
        self.config = config
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            stop_words=None,
            token_pattern=r"(?u)\b[\w\-]{2,}\b",
            max_features=config.max_features,
        )

    def fit_transform(self, documents: Sequence[str]):
        return self.vectorizer.fit_transform(documents)

    def transform(self, documents: Sequence[str]):
        return self.vectorizer.transform(documents)

    def feature_dimension(self) -> int:
        n_features = getattr(self.vectorizer, "max_features", None)
        if n_features and n_features > 0:
            return int(n_features)
        if hasattr(self.vectorizer, "vocabulary_"):
            return int(len(self.vectorizer.vocabulary_))
        return 0

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if joblib is None:  # pragma: no cover - optional dependency
            raise ImportError("Install joblib to save TF-IDF vectorisers.")
        joblib.dump(self.vectorizer, directory / "tfidf_vectorizer.joblib")
        with open(directory / VECTORISER_META, "w", encoding="utf-8") as handle:
            json.dump({"kind": self.kind, "config": asdict(self.config)}, handle, indent=2)

    @classmethod
    def load(cls, directory: Path) -> "TfidfVectorizerWrapper":
        if joblib is None:  # pragma: no cover - optional dependency
            raise ImportError("Install joblib to load TF-IDF vectorisers.")
        with open(directory / VECTORISER_META, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        config = TfidfConfig(**meta.get("config", {}))
        instance = cls(config)
        instance.vectorizer = joblib.load(directory / "tfidf_vectorizer.joblib")
        return instance


@dataclass
class Word2VecVectorizerConfig:
    """Configuration for Word2Vec-based embeddings."""

    vector_size: int = 256
    window: int = 5
    min_count: int = 2
    epochs: int = 10
    workers: int = 1
    seed: int = 42
    model_dir: str | None = None


class Word2VecVectorizer(BaseTextVectorizer):
    """Vectoriser that averages Word2Vec embeddings."""

    kind = "word2vec"

    def __init__(self, config: Word2VecVectorizerConfig) -> None:
        self.config = config
        model_dir = Path(config.model_dir) if config.model_dir else Path("models/xgb_word2vec")
        self._builder = Word2VecFeatureBuilder(
            Word2VecConfig(
                vector_size=config.vector_size,
                window=config.window,
                min_count=config.min_count,
                epochs=config.epochs,
                workers=config.workers,
                seed=config.seed,
                model_dir=model_dir,
            )
        )

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        self._builder.train(documents)
        return self._builder.transform(documents)

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        return self._builder.transform(documents)

    def feature_dimension(self) -> int:
        return int(self._builder.config.vector_size)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        model_dir = directory / "word2vec_model"
        self._builder.save(model_dir)
        payload = {
            "kind": self.kind,
            "config": {
                "vector_size": self._builder.config.vector_size,
                "window": self._builder.config.window,
                "min_count": self._builder.config.min_count,
                "epochs": self._builder.config.epochs,
                "workers": self._builder.config.workers,
                "seed": self._builder.config.seed,
                "model_dir": str(model_dir),
            },
        }
        with open(directory / VECTORISER_META, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @classmethod
    def load(cls, directory: Path) -> "Word2VecVectorizer":
        with open(directory / VECTORISER_META, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        cfg = Word2VecVectorizerConfig(**meta.get("config", {}))
        instance = cls(cfg)
        instance._builder.load(Path(cfg.model_dir))
        return instance


@dataclass
class SentenceTransformerVectorizerConfig:
    """Configuration bundle for SentenceTransformer vectorisation."""

    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str | None = None
    batch_size: int = 32
    normalize: bool = True


class SentenceTransformerVectorizer(BaseTextVectorizer):
    """Vectoriser using pre-trained sentence-transformer encoders."""

    kind = "sentence_transformer"

    def __init__(self, config: SentenceTransformerVectorizerConfig) -> None:
        self.config = config
        self._encoder = SentenceTransformerEncoder(
            SentenceTransformerConfig(
                model_name=config.model_name,
                device=config.device,
                batch_size=config.batch_size,
                normalize=config.normalize,
            )
        )
        self._dimension: int | None = None

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        encoded = self._encoder.encode(documents)
        self._dimension = int(encoded.shape[1]) if encoded.ndim == 2 else 0
        return encoded

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        encoded = self._encoder.encode(documents)
        if self._dimension is None and encoded.ndim == 2:
            self._dimension = int(encoded.shape[1])
        return encoded

    def feature_dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        self._dimension = self._encoder.embedding_dimension()
        return self._dimension

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "config": {
                "model_name": self.config.model_name,
                "device": self.config.device,
                "batch_size": self.config.batch_size,
                "normalize": self.config.normalize,
            },
        }
        with open(directory / VECTORISER_META, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @classmethod
    def load(cls, directory: Path) -> "SentenceTransformerVectorizer":
        with open(directory / VECTORISER_META, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        cfg = SentenceTransformerVectorizerConfig(**meta.get("config", {}))
        return cls(cfg)


def create_vectorizer(
    kind: str,
    *,
    tfidf: TfidfConfig | None = None,
    word2vec: Word2VecVectorizerConfig | None = None,
    sentence_transformer: SentenceTransformerVectorizerConfig | None = None,
) -> BaseTextVectorizer:
    """Return a vectoriser instance for ``kind``."""

    if kind == "tfidf":
        return TfidfVectorizerWrapper(tfidf or TfidfConfig())
    if kind == "word2vec":
        return Word2VecVectorizer(word2vec or Word2VecVectorizerConfig())
    if kind == "sentence_transformer":
        return SentenceTransformerVectorizer(sentence_transformer or SentenceTransformerVectorizerConfig())
    raise ValueError(f"Unsupported text vectorizer '{kind}'.")


def load_vectorizer(directory: Path) -> BaseTextVectorizer:
    """Load a vectoriser saved via :meth:`BaseTextVectorizer.save`."""

    with open(directory / VECTORISER_META, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    kind = meta.get("kind")
    if kind == "tfidf":
        return TfidfVectorizerWrapper.load(directory)
    if kind == "word2vec":
        return Word2VecVectorizer.load(directory)
    if kind == "sentence_transformer":
        return SentenceTransformerVectorizer.load(directory)
    raise ValueError(f"Unsupported vectorizer kind '{kind}'.")
