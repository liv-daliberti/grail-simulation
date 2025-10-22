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
        """
        Fit the vectoriser on ``documents`` and return the resulting features.

        Parameters
        ----------
        documents:
            Corpus used to establish model parameters.

        Returns
        -------
        Any
            Implementation-specific feature matrix.
        """

        raise NotImplementedError

    def transform(self, documents: Sequence[str]) -> Any:  # pragma: no cover - abstract
        """
        Encode ``documents`` using the fitted vectoriser.

        Parameters
        ----------
        documents:
            Inputs to transform with the learned representation.

        Returns
        -------
        Any
            Feature matrix aligned with the fitted model state.
        """

        raise NotImplementedError

    def feature_dimension(self) -> int:  # pragma: no cover - abstract
        """
        Return the number of features produced during transformation.

        Returns
        -------
        int
            Width of the feature representation.
        """

        raise NotImplementedError

    def save(self, directory: Path) -> None:  # pragma: no cover - abstract
        """
        Persist the trained vectoriser into ``directory``.

        Parameters
        ----------
        directory:
            Destination folder for serialised artefacts.
        """

        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path) -> "BaseTextVectorizer":  # pragma: no cover - abstract
        """
        Reconstruct a vectoriser from ``directory``.

        Parameters
        ----------
        directory:
            Location containing saved model artefacts.

        Returns
        -------
        BaseTextVectorizer
            Fully initialised subclass instance.
        """

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
        """
        Create a TF-IDF vectoriser wrapper.

        Parameters
        ----------
        config:
            Hyper-parameters controlling vocabulary size and tokenisation.
        """

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
        """
        Fit the TF-IDF vectoriser and transform ``documents``.

        Parameters
        ----------
        documents:
            Corpus used for fitting and transformation.

        Returns
        -------
        Any
            Sparse matrix of TF-IDF features.
        """

        return self.vectorizer.fit_transform(documents)

    def transform(self, documents: Sequence[str]):
        """
        Transform ``documents`` using the fitted TF-IDF model.

        Parameters
        ----------
        documents:
            Input texts to encode.

        Returns
        -------
        Any
            Sparse matrix of TF-IDF features matching the fitted vocabulary.
        """

        return self.vectorizer.transform(documents)

    def feature_dimension(self) -> int:
        """
        Return the number of TF-IDF features exposed by the model.

        Returns
        -------
        int
            Vocabulary size reflected in the vectoriser.
        """

        n_features = getattr(self.vectorizer, "max_features", None)
        if n_features and n_features > 0:
            return int(n_features)
        if hasattr(self.vectorizer, "vocabulary_"):
            return int(len(self.vectorizer.vocabulary_))
        return 0

    def save(self, directory: Path) -> None:
        """
        Persist the fitted TF-IDF vectoriser to ``directory``.

        Parameters
        ----------
        directory:
            Destination folder receiving the serialised model artefacts.
        """

        directory.mkdir(parents=True, exist_ok=True)
        if joblib is None:  # pragma: no cover - optional dependency
            raise ImportError("Install joblib to save TF-IDF vectorisers.")
        joblib.dump(self.vectorizer, directory / "tfidf_vectorizer.joblib")
        with open(directory / VECTORISER_META, "w", encoding="utf-8") as handle:
            json.dump({"kind": self.kind, "config": asdict(self.config)}, handle, indent=2)

    @classmethod
    def load(cls, directory: Path) -> "TfidfVectorizerWrapper":
        """
        Restore a TF-IDF vectoriser previously saved with :meth:`save`.

        Parameters
        ----------
        directory:
            Folder containing the saved estimator and metadata.

        Returns
        -------
        TfidfVectorizerWrapper
            Reconstructed wrapper ready for inference.
        """

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
        """
        Build a Word2Vec-backed vectoriser.

        Parameters
        ----------
        config:
            Training and inference options for the Word2Vec feature builder.
        """

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
        """
        Train the Word2Vec model and encode ``documents``.

        Parameters
        ----------
        documents:
            Corpus used for model fitting and embedding aggregation.

        Returns
        -------
        numpy.ndarray
            Dense embedding matrix with one row per document.
        """

        self._builder.train(documents)
        return self._builder.transform(documents)

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """
        Encode ``documents`` using the fitted Word2Vec model.

        Parameters
        ----------
        documents:
            Texts to convert into averaged embeddings.

        Returns
        -------
        numpy.ndarray
            Dense embedding matrix using the cached embeddings.
        """

        return self._builder.transform(documents)

    def feature_dimension(self) -> int:
        """
        Return the dimensionality of the produced Word2Vec embeddings.

        Returns
        -------
        int
            Embedding size configured for the model.
        """

        return int(self._builder.config.vector_size)

    def save(self, directory: Path) -> None:
        """
        Store the Word2Vec model artefacts in ``directory``.

        Parameters
        ----------
        directory:
            Destination folder receiving the trained embeddings and metadata.
        """

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
        """
        Load a Word2Vec vectoriser from ``directory``.

        Parameters
        ----------
        directory:
            Folder containing a previous call to :meth:`save`.

        Returns
        -------
        Word2VecVectorizer
            Rehydrated vectoriser with weights and configuration applied.
        """

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
        """
        Create a vectoriser backed by a sentence-transformer model.

        Parameters
        ----------
        config:
            Encoder settings forwarded to :class:`common.embeddings.SentenceTransformerEncoder`.
        """

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
        """
        Encode ``documents`` while caching the embedding dimensionality.

        Parameters
        ----------
        documents:
            Texts to embed via the underlying sentence-transformer.

        Returns
        -------
        numpy.ndarray
            Dense array of unit-normalised document embeddings.
        """

        encoded = self._encoder.encode(documents)
        self._dimension = int(encoded.shape[1]) if encoded.ndim == 2 else 0
        return encoded

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """
        Encode ``documents`` using the cached sentence-transformer model.

        Parameters
        ----------
        documents:
            Text inputs to convert into embeddings.

        Returns
        -------
        numpy.ndarray
            Dense embedding array reusing the cached model instance.
        """

        encoded = self._encoder.encode(documents)
        if self._dimension is None and encoded.ndim == 2:
            self._dimension = int(encoded.shape[1])
        return encoded

    def feature_dimension(self) -> int:
        """
        Return the dimensionality of the sentence-transformer embeddings.

        Returns
        -------
        int
            Embedding width either cached from training or derived on demand.
        """

        if self._dimension is not None:
            return self._dimension
        self._dimension = self._encoder.embedding_dimension()
        return self._dimension

    def save(self, directory: Path) -> None:
        """
        Persist the vectoriser configuration metadata to ``directory``.

        Parameters
        ----------
        directory:
            Destination folder receiving the serialised configuration.
        """

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
        """
        Restore configuration needed to rebuild the sentence-transformer encoder.

        Parameters
        ----------
        directory:
            Folder containing metadata generated by :meth:`save`.

        Returns
        -------
        SentenceTransformerVectorizer
            Vectoriser initialised with the persisted configuration.
        """

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
