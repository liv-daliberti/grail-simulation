"""Feature extraction helpers for the refactored KNN baselines."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from numpy.random import default_rng

from common.text import canon_video_id
from common.title_index import TitleResolver
from prompt_builder import build_user_prompt, clean_text, synthesize_viewer_sentence

try:  # pragma: no cover - optional dependency
    from gensim.models import Word2Vec  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Word2Vec = None

from .data import PROMPT_COLUMN, PROMPT_MAX_HISTORY, SOLUTION_COLUMN

# ---------------------------------------------------------------------------
# Title index helpers
# ---------------------------------------------------------------------------

_TITLE_INDEX_ROOT = (
    "/n/fs/similarity/trees/data/results/"
    "capsule-5416997-data/recommendation trees"
)
DEFAULT_TITLE_DIRS = [
    f"{_TITLE_INDEX_ROOT}/trees_gun",
    f"{_TITLE_INDEX_ROOT}/trees_wage",
]

CANON_RE = re.compile(r"[^a-z0-9]+")

_TITLE_RESOLVER = TitleResolver(default_dirs=DEFAULT_TITLE_DIRS)


def title_for(video_id: str) -> Optional[str]:
    """Return a human-readable title for a YouTube id if available."""

    title = _TITLE_RESOLVER.resolve(video_id)
    return title


# ---------------------------------------------------------------------------
# Prompt/document assembly helpers
# ---------------------------------------------------------------------------


def _canon(text: str) -> str:
    return CANON_RE.sub("", (text or "").lower().strip())


def _is_nanlike(value: Any) -> bool:
    if value is None:
        return True
    string = str(value).strip().lower()
    return string in {"", "nan", "none", "null", "na", "n/a"}


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def viewer_profile_sentence(example: dict) -> str:
    """Return the viewer profile sentence for ``example``."""

    sentence = clean_text(example.get("viewer_profile_sentence"))
    if not sentence:
        sentence = clean_text(example.get("viewer_profile"))
    if not sentence:
        try:
            sentence = synthesize_viewer_sentence(example)
        except ValueError:  # pragma: no cover - defensive
            sentence = ""
    return sentence or ""


def prompt_from_builder(example: dict) -> str:
    """Return the prompt text for ``example`` using the shared builder."""

    existing = example.get("state_text") or example.get("prompt")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    try:
        return build_user_prompt(example, max_hist=PROMPT_MAX_HISTORY)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return ""


def _pick_ci(mapping: dict, *alternates: str) -> Optional[str]:
    if not isinstance(mapping, dict):
        return None
    lower = {key.lower(): key for key in mapping.keys()}
    for candidate in alternates:
        original = lower.get(candidate.lower())
        if original:
            value = mapping.get(original)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_now_watching(example: dict) -> Optional[Tuple[str, str]]:
    """Return the current (title, video_id) tuple if present."""

    video_id = _pick_ci(example, "video_id", "videoId")
    if video_id and not _is_nanlike(video_id):
        title = _pick_ci(
            example,
            "current_video_title",
            "now_playing_title",
            "watching_title",
            "currentVideoTitle",
            "nowPlayingTitle",
            "watchingTitle",
            "now_title",
            "current_title",
            "meta_originTitle",
        )
        title = title or title_for(video_id) or ""
        return (title or "(untitled)", video_id)
    title = _pick_ci(
        example,
        "current_video_title",
        "now_playing_title",
        "watching_title",
        "currentVideoTitle",
        "nowPlayingTitle",
        "watchingTitle",
        "now_title",
        "current_title",
    )
    video_id = _pick_ci(
        example,
            "current_video_id",
            "now_playing_id",
            "watching_id",
            "currentVideoId",
            "nowPlayingId",
            "watchingId",
            "now_id",
            "current_id",
            "originId",
            "video_id",
            "videoId",
        )
    if (title and not _is_nanlike(title)) or (video_id and not _is_nanlike(video_id)):
        if _is_nanlike(title) and video_id:
            title = title_for(video_id) or ""
        return (title or "(untitled)", video_id or "")
    return None


def _extract_slate_items(example: dict) -> List[Tuple[str, str]]:
    """Return the slate contents as a list of ``(title, video_id)`` pairs."""

    items: List[Tuple[str, str]] = []
    slate_text = example.get("slate_text")
    if isinstance(slate_text, str) and slate_text.strip():
        for line in slate_text.splitlines():
            token = line.strip()
            if not token:
                continue
            parts = token.split("\t") if "\t" in token else token.split("|", maxsplit=1)
            if len(parts) == 2:
                title, vid = parts
            else:
                title, vid = token, token
            items.append((title.strip(), vid.strip()))
    if items:
        return items
    array = example.get("slate_items") or example.get("options")
    if isinstance(array, list):
        for entry in array:
            if not isinstance(entry, dict):
                continue
            title = entry.get("title") or entry.get("video_title") or entry.get("name") or ""
            video_id = entry.get("id") or entry.get("video_id") or ""
            items.append((title, video_id))
    return items


def extract_slate_items(example: dict) -> List[Tuple[str, str]]:
    """Public wrapper returning the slate as ``(title, video_id)`` pairs."""

    return _extract_slate_items(example)


def extract_now_watching(example: dict) -> Optional[Tuple[str, str]]:
    """Public wrapper returning the currently-watched title/id, if known."""

    return _extract_now_watching(example)


def assemble_document(example: dict, extra_fields: Sequence[str] | None = None) -> str:
    """Return concatenated text used to featurise ``example``."""

    extra_fields = extra_fields or []

    def _good(text: str) -> bool:
        return bool(text and text.lower() not in {"", "nan", "none", "(none)"})

    parts: List[str] = []
    prompt_text = prompt_from_builder(example)
    if not _good(prompt_text):
        fallback_candidates = (
            viewer_profile_sentence(example),
            clean_text(example.get(PROMPT_COLUMN)),
        )
        prompt_text = next((value for value in fallback_candidates if _good(value)), "")
    if prompt_text:
        parts.append(prompt_text)

    now_watching = _extract_now_watching(example)
    if now_watching:
        now_title, now_id = now_watching
        if _good(now_title):
            parts.append(now_title)
        if _good(now_id):
            parts.append(now_id)

    for title, video_id in _extract_slate_items(example):
        surface = (
            title
            if _good(title) and title != "(untitled)"
            else (title_for(video_id) or video_id or "")
        )
        if _good(surface):
            parts.append(surface)

    for field in extra_fields:
        cleaned = clean_text(example.get(field))
        if _good(cleaned):
            parts.append(cleaned)

    return " ".join(parts).strip()


def prepare_training_documents(
    train_ds,
    max_train: int,
    seed: int,
    extra_fields: Sequence[str] | None = None,
):
    """Return TF-IDF training documents and associated labels."""

    n_rows = len(train_ds)
    if n_rows == 0:
        raise RuntimeError("Train split is empty.")
    rng = default_rng(seed)
    if max_train and max_train > 0:
        take = min(max_train, n_rows)
        order = rng.permutation(n_rows)[:take].tolist()
    else:
        order = list(range(n_rows))

    records: List[tuple[str, str, str]] = []
    for index in order:
        record = _record_from_example(train_ds[int(index)], extra_fields)
        if record:
            records.append(record)

    if not records:
        raise RuntimeError(
            "All training documents are empty. Check columns on TRAIN split.\n"
            f"Seen columns: {sorted(list(train_ds.features.keys()))}\n"
            "Fixes: add slate items/current video text or use --knn_text_fields."
        )

    dropped = len(order) - len(records)
    if dropped:
        logging.warning("[KNN] Dropped %d empty docs out of %d.", dropped, len(order))

    filtered_docs = [doc for doc, _, _ in records]
    filtered_labels_id = [label_id for _, label_id, _ in records]
    filtered_labels_title = [label_title for _, _, label_title in records]
    logging.info("[KNN] Assembled %d documents (kept %d non-empty).", len(order), len(records))
    logging.info("[KNN] Example doc: %r", filtered_docs[0][:200])
    return filtered_docs, filtered_labels_id, filtered_labels_title


def _record_from_example(
    example: dict,
    extra_fields: Sequence[str] | None,
) -> Optional[tuple[str, str, str]]:
    document = assemble_document(example, extra_fields).strip()
    if not document:
        return None
    video_id = str(example.get(SOLUTION_COLUMN) or "")
    label_id = canon_video_id(video_id)
    label_title = title_for(video_id) or ""
    return document, label_id, label_title


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


class Word2VecFeatureBuilder:
    """Create Word2Vec embeddings from viewer prompts."""

    def __init__(self, config: Optional[Word2VecConfig] = None) -> None:
        self.config = config or Word2VecConfig()
        self._model = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
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

    def encode(self, text: str) -> List[float]:
        """Return the averaged embedding vector for ``text``."""

        if self._model is None:
            raise RuntimeError("Word2Vec model has not been trained/loaded")
        tokens = [token for token in self._tokenize(text) if token in self._model.wv]
        if not tokens:
            return [0.0] * self._model.vector_size
        return list(self._model.wv[tokens].mean(axis=0))


__all__ = [
    "DEFAULT_TITLE_DIRS",
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
