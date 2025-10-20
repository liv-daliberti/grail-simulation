"""Feature extraction helpers for the refactored KNN baselines."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from prompt_builder.formatters import clean_text
from prompt_builder.prompt import build_user_prompt
from prompt_builder.profiles import synthesize_viewer_sentence

from .data import PROMPT_COLUMN, PROMPT_MAX_HISTORY, SOLUTION_COLUMN

# ---------------------------------------------------------------------------
# Title index helpers
# ---------------------------------------------------------------------------

DEFAULT_TITLE_DIRS = [
    "/n/fs/similarity/trees/data/results/capsule-5416997-data/recommendation trees/trees_gun",
    "/n/fs/similarity/trees/data/results/capsule-5416997-data/recommendation trees/trees_wage",
]

YTID_RE = re.compile(r"([A-Za-z0-9_-]{11})")
CANON_RE = re.compile(r"[^a-z0-9]+")


def _split_env_list(value: str | None) -> List[str]:
    if not value:
        return []
    parts: List[str] = []
    for chunk in re.split(r"[:,\s]+", value):
        token = chunk.strip()
        if token:
            parts.append(token)
    return parts


def _iter_csv_files_from_env() -> List[str]:
    files: List[str] = []
    for path in _split_env_list(os.environ.get("GRAIL_TITLE_CSVS")):
        if os.path.isfile(path):
            files.append(path)
    for directory in _split_env_list(os.environ.get("GRAIL_TITLE_DIRS")):
        if os.path.isdir(directory):
            for root, _, filenames in os.walk(directory):
                for name in filenames:
                    if name.lower().endswith(".csv"):
                        files.append(os.path.join(root, name))
    for pattern in _split_env_list(os.environ.get("GRAIL_TITLE_GLOB")):
        try:
            from glob import glob

            for candidate in glob(pattern):
                if os.path.isfile(candidate):
                    files.append(candidate)
        except Exception:  # pragma: no cover - defensive
            continue
    if not files:
        for directory in DEFAULT_TITLE_DIRS:
            if os.path.isdir(directory):
                for root, _, filenames in os.walk(directory):
                    for name in filenames:
                        if name.lower().endswith(".csv"):
                            files.append(os.path.join(root, name))
    deduped: List[str] = []
    seen: set[str] = set()
    for path in files:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _guess_cols(header: List[str]) -> Tuple[Optional[str], Optional[str]]:
    candidate_ids = [
        "originId",
        "ytid",
        "video_id",
        "youtube_id",
        "videoId",
        "origin_id",
        "id",
    ]
    candidate_titles = ["originTitle", "title", "video_title", "name"]
    lower = {column.lower(): column for column in header}
    id_col = next((lower[name.lower()] for name in candidate_ids if name.lower() in lower), None)
    title_col = next((lower[name.lower()] for name in candidate_titles if name.lower() in lower), None)
    return id_col, title_col


_TITLE_INDEX: Optional[Dict[str, str]] = None


def _build_title_index() -> Dict[str, str]:
    index: Dict[str, str] = {}
    import csv

    for path in _iter_csv_files_from_env():
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                id_col, title_col = _guess_cols(reader.fieldnames)
                if not id_col or not title_col:
                    continue
                for row in reader:
                    vid = _canon_vid(row.get(id_col, "") or "")
                    title = (row.get(title_col, "") or "").strip()
                    if vid and title and vid not in index:
                        index[vid] = title
        except Exception:  # pragma: no cover - defensive
            continue
    return index


def title_for(video_id: str) -> Optional[str]:
    """Return a human-readable title for a YouTube id if available."""

    global _TITLE_INDEX  # pylint: disable=global-statement
    if _TITLE_INDEX is None:
        _TITLE_INDEX = _build_title_index()
        logging.info("[title-index] loaded %d titles from CSV", len(_TITLE_INDEX))
    return _TITLE_INDEX.get(_canon_vid(video_id))


# ---------------------------------------------------------------------------
# Prompt/document assembly helpers
# ---------------------------------------------------------------------------


def _canon(text: str) -> str:
    return CANON_RE.sub("", (text or "").lower().strip())


def _canon_vid(value: str) -> str:
    if not isinstance(value, str):
        return ""
    match = YTID_RE.search(value)
    return match.group(1) if match else value.strip()


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
    sentence = clean_text(example.get("viewer_profile_sentence"))
    if not sentence:
        sentence = clean_text(example.get("viewer_profile"))
    if not sentence:
        try:
            sentence = synthesize_viewer_sentence(example)
        except Exception:  # pragma: no cover - defensive
            sentence = ""
    return sentence or ""


def prompt_from_builder(example: dict) -> str:
    existing = example.get("state_text") or example.get("prompt")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    try:
        return build_user_prompt(example, max_hist=PROMPT_MAX_HISTORY)
    except Exception:  # pragma: no cover - defensive
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


def assemble_document(example: dict, extra_fields: Sequence[str] | None = None) -> str:
    extra_fields = extra_fields or []

    def _good(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return lowered not in {"", "nan", "none", "(none)"}

    parts: List[str] = []
    prompt_text = prompt_from_builder(example)
    used_prompt = False
    if _good(prompt_text):
        parts.append(prompt_text)
        used_prompt = True
    if not used_prompt:
        profile_sentence = viewer_profile_sentence(example)
        if _good(profile_sentence):
            parts.append(profile_sentence)
    if not used_prompt and PROMPT_COLUMN in example:
        state_text = clean_text(example.get(PROMPT_COLUMN))
        if _good(state_text):
            parts.append(state_text)
    current = _extract_now_watching(example)
    if current:
        title, video_id = current
        if _good(title):
            parts.append(title)
        if _good(video_id):
            parts.append(video_id)
    for title, video_id in _extract_slate_items(example):
        surface = title if _good(title) and title != "(untitled)" else (title_for(video_id) or video_id or "")
        if _good(surface):
            parts.append(surface)
    for field in extra_fields:
        value = clean_text(example.get(field))
        if _good(value):
            parts.append(value)
    return " ".join(fragment for fragment in parts if _good(fragment)).strip()


def prepare_training_documents(
    train_ds,
    max_train: int,
    seed: int,
    extra_fields: Sequence[str] | None = None,
):
    from numpy.random import default_rng

    n_rows = len(train_ds)
    if n_rows == 0:
        raise RuntimeError("Train split is empty.")
    rng = default_rng(seed)
    if max_train and max_train > 0:
        take = min(max_train, n_rows)
        order = rng.permutation(n_rows)[:take].tolist()
    else:
        order = list(range(n_rows))

    documents: List[str] = []
    labels_id: List[str] = []
    labels_title: List[str] = []

    for index in order:
        example = train_ds[int(index)]
        document = assemble_document(example, extra_fields)
        documents.append(document)
        video_id = str(example.get(SOLUTION_COLUMN) or "")
        labels_id.append(_canon_vid(video_id))
        try:
            title = title_for(video_id) or ""
        except Exception:  # pragma: no cover - defensive
            title = ""
        labels_title.append(title)

    mask = [bool(doc.strip()) for doc in documents]
    if not any(mask):
        sample_cols = sorted(list(train_ds.features.keys()))
        raise RuntimeError(
            "All training documents are empty. Check columns on TRAIN split.\n"
            f"Seen columns: {sample_cols}\n"
            "Fixes: include slate items / current video, or pass extra fields via --knn_text_fields."
        )
    if sum(mask) < len(documents):
        logging.warning(
            "[KNN] Dropped %d empty docs out of %d.", len(documents) - sum(mask), len(documents)
        )
    filtered_docs = [doc for doc, keep in zip(documents, mask) if keep]
    filtered_labels_id = [label for label, keep in zip(labels_id, mask) if keep]
    filtered_labels_title = [label for label, keep in zip(labels_title, mask) if keep]
    return filtered_docs, filtered_labels_id, filtered_labels_title


# ---------------------------------------------------------------------------
# TF-IDF + Word2Vec interfaces
# ---------------------------------------------------------------------------


@dataclass
class Word2VecConfig:
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
    "prepare_training_documents",
    "prompt_from_builder",
    "title_for",
]
