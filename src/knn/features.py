"""Feature extraction helpers for the refactored KNN baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from gensim.models import Word2Vec  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Word2Vec = None

from common import get_logger
from common.prompt_docs import (
    DEFAULT_TITLE_DIRS as _DEFAULT_TITLE_DIRS,
    EXTRA_FIELD_LABELS as _COMMON_EXTRA_FIELD_LABELS,
    PromptDocumentBuilder,
    default_title_resolver,
)

from .data import PROMPT_COLUMN, PROMPT_MAX_HISTORY, SOLUTION_COLUMN

DEFAULT_TITLE_DIRS = _DEFAULT_TITLE_DIRS
EXTRA_FIELD_LABELS = _COMMON_EXTRA_FIELD_LABELS

_PROMPT_DOC_BUILDER = PromptDocumentBuilder(
    prompt_column=PROMPT_COLUMN,
    solution_column=SOLUTION_COLUMN,
    max_history=PROMPT_MAX_HISTORY,
    title_lookup=default_title_resolver(),
    log_prefix="[KNN]",
    logger=get_logger("knn.features"),
)


def title_for(video_id: str) -> Optional[str]:
    """Return a human-readable title for a YouTube id if available."""

    return _PROMPT_DOC_BUILDER.title_for(video_id)


def viewer_profile_sentence(example: dict) -> str:
    """Return the viewer profile sentence for ``example``."""

    return _PROMPT_DOC_BUILDER.viewer_profile_sentence(example)


def prompt_from_builder(example: dict) -> str:
    """Return the prompt text for ``example`` using the shared builder."""

    return _PROMPT_DOC_BUILDER.prompt_from_builder(example)


def extract_now_watching(example: dict) -> Optional[Tuple[str, str]]:
    """Return the currently watched title/id, if known."""

    return _PROMPT_DOC_BUILDER.extract_now_watching(example)


def extract_slate_items(example: dict) -> List[Tuple[str, str]]:
    """Return the slate as ``(title, video_id)`` pairs."""

    return _PROMPT_DOC_BUILDER.extract_slate_items(example)


def assemble_document(example: dict, extra_fields: Sequence[str] | None = None) -> str:
    """Return concatenated text used to featurise ``example``."""

    return _PROMPT_DOC_BUILDER.assemble_document(example, extra_fields)


def prepare_training_documents(
    train_ds,
    max_train: int,
    seed: int,
    extra_fields: Sequence[str] | None = None,
):
    """Return TF-IDF training documents and associated labels."""

    return _PROMPT_DOC_BUILDER.prepare_training_documents(
        train_ds,
        max_train,
        seed,
        extra_fields,
    )


# ---------------------------------------------------------------------------
# TF-IDF + Word2Vec interfaces
# ---------------------------------------------------------------------------


@dataclass
class Word2VecConfig:
    """Configuration options for Word2Vec embeddings."""

    vector_size: int = 256
    window: int = 5
    min_count: int = 2
    epochs: int = 10
    model_dir: Path = Path("models/knn_word2vec")
    seed: int = 42
    workers: int = 1


class Word2VecFeatureBuilder:
    """Create Word2Vec embeddings from viewer prompts."""

    def __init__(self, config: Optional[Word2VecConfig] = None) -> None:
        """Initialise the builder with optional configuration overrides.

        :param config: Optional :class:`Word2VecConfig` instance.
        """
        self.config = config or Word2VecConfig()
        self._model = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenise input text into whitespace-delimited lower-case tokens.

        :param text: Text to split.
        :returns: List of tokens.
        """
        return text.lower().split()

    def train(self, corpus: Iterable[str]) -> None:
        """Train a Word2Vec model using the provided corpus."""

        if Word2Vec is None:  # pragma: no cover - optional dependency
            raise ImportError("Install gensim to enable Word2Vec embeddings")
        sentences = [self._tokenize(text) for text in corpus]
        self._model = Word2Vec(
            sentences=sentences,
            vector_size=self.config.vector_size,
            window=self.config.window,
            min_count=self.config.min_count,
            sg=1,
            epochs=self.config.epochs,
            seed=self.config.seed,
            workers=self.config.workers,
        )
        self.save(self.config.model_dir)

    def load(self, directory: Path) -> None:
        """Load a previously trained Word2Vec model from disk."""

        if Word2Vec is None:  # pragma: no cover - optional dependency
            raise ImportError("Install gensim to enable Word2Vec embeddings")
        self._model = Word2Vec.load(str(directory / "word2vec.model"))

    def save(self, directory: Path) -> None:
        """Persist the trained model to ``directory``."""

        if self._model is None:
            raise RuntimeError("Word2Vec model must be trained before saving")
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save(str(directory / "word2vec.model"))

    def is_trained(self) -> bool:
        """Return True when the underlying Word2Vec model is ready for inference."""

        return self._model is not None

    def encode(self, text: str) -> np.ndarray:
        """Return the averaged embedding vector for ``text``."""

        if self._model is None:
            raise RuntimeError("Word2Vec model has not been trained/loaded")
        tokens = [token for token in self._tokenize(text) if token in self._model.wv]
        if not tokens:
            return np.zeros(self._model.vector_size, dtype=np.float32)
        return np.asarray(self._model.wv[tokens].mean(axis=0), dtype=np.float32)

    def transform(self, corpus: Sequence[str]) -> np.ndarray:
        """Return stacked embeddings for ``corpus``."""

        if self._model is None:
            raise RuntimeError("Word2Vec model has not been trained/loaded")
        if not corpus:
            return np.zeros((0, self._model.vector_size), dtype=np.float32)
        vectors = [self.encode(text) for text in corpus]
        return np.vstack(vectors).astype(np.float32)


__all__ = [
    "DEFAULT_TITLE_DIRS",
    "EXTRA_FIELD_LABELS",
    "Word2VecConfig",
    "Word2VecFeatureBuilder",
    "assemble_document",
    "extract_now_watching",
    "extract_slate_items",
    "prepare_training_documents",
    "prompt_from_builder",
    "title_for",
    "viewer_profile_sentence",
]
