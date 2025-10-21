"""Shared prompt document assembly helpers used across baselines."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from numpy.random import default_rng

from common import canon_text, canon_video_id, get_logger
from common.title_index import TitleResolver
from prompt_builder import build_user_prompt, clean_text, synthesize_viewer_sentence
from prompt_builder.constants import GUN_FIELD_LABELS, MIN_WAGE_FIELD_LABELS
from prompt_builder.value_maps import format_field_value

TitleLookup = Callable[[Optional[str]], Optional[str]]

_TITLE_INDEX_ROOT = (
    "/n/fs/similarity/trees/data/results/"
    "capsule-5416997-data/recommendation trees"
)
DEFAULT_TITLE_DIRS = [
    f"{_TITLE_INDEX_ROOT}/trees_gun",
    f"{_TITLE_INDEX_ROOT}/trees_wage",
]

EXTRA_FIELD_LABELS: Dict[str, str] = {
    "pid1": "Party identification",
    "pid2": "Party lean",
    "ideo1": "Political ideology",
    "ideo2": "Ideology intensity",
    "pol_interest": "Political interest",
    "religpew": "Religion",
    "freq_youtube": "YouTube frequency",
    "youtube_time": "YouTube time",
    "newsint": "News attention",
    "participant_study": "Participant study",
    "slate_source": "Slate source",
    "educ": "Education level",
    "employ": "Employment status",
    "child18": "Children in household",
    "inputstate": "State",
    "q31": "Household income",
    "income": "Household income",
}
EXTRA_FIELD_LABELS.update(MIN_WAGE_FIELD_LABELS)
EXTRA_FIELD_LABELS.update(GUN_FIELD_LABELS)

_DEFAULT_TITLE_RESOLVER: Optional[TitleResolver] = None


def default_title_resolver() -> TitleResolver:
    """Return a lazily constructed title resolver for the cleaned GRAIL dataset."""

    global _DEFAULT_TITLE_RESOLVER  # pylint: disable=global-statement
    if _DEFAULT_TITLE_RESOLVER is None:
        _DEFAULT_TITLE_RESOLVER = TitleResolver(default_dirs=DEFAULT_TITLE_DIRS)
    return _DEFAULT_TITLE_RESOLVER


def _looks_like_legacy_prompt(prompt_text: str) -> bool:
    legacy_tokens = (
        "PROFILE:",
        "ATTRIBUTES:",
        "CURRENT VIDEO:",
        "RECENTLY WATCHED (NEWEST LAST):",
        "OPTIONS:",
        "SURVEY HIGHLIGHTS:",
    )
    return any(token in prompt_text for token in legacy_tokens)


def _pick_ci(mapping: dict, *alternates: str) -> Optional[str]:
    if not isinstance(mapping, dict):
        return None
    lower = {key.lower(): key for key in mapping.keys()}
    for candidate in alternates:
        original = lower.get(candidate.lower())
        if not original:
            continue
        value = mapping.get(original)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_nanlike(value: object) -> bool:
    if value is None:
        return True
    string = str(value).strip().lower()
    return string in {"", "nan", "none", "null", "na", "n/a"}


def _extract_now_watching(
    example: dict,
    title_lookup: TitleLookup | None,
) -> Optional[Tuple[str, str]]:
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
        if _is_nanlike(title) and title_lookup is not None:
            title = title_lookup(video_id)
        return (title or "(untitled)", str(video_id))
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
        if _is_nanlike(title) and title_lookup is not None:
            title = title_lookup(video_id) or ""
        return (title or "(untitled)", video_id or "")
    return None


def _extract_slate_items(
    example: dict,
    title_lookup: TitleLookup | None,
) -> List[Tuple[str, str]]:
    def _clean_title(value: object) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _clean_id(value: object) -> str:
        if not value:
            return ""
        candidate = canon_video_id(str(value))
        return candidate if len(candidate) == 11 else ""

    def _append_item(title: object, video_id: object) -> None:
        cleaned_title = _clean_title(title)
        cleaned_id = _clean_id(video_id)
        if not cleaned_id and isinstance(title, str):
            possible_id = _clean_id(title)
            if possible_id:
                cleaned_id = possible_id
                cleaned_title = ""
        if not cleaned_title and cleaned_id and title_lookup is not None:
            cleaned_title = title_lookup(cleaned_id) or ""
        if cleaned_title or cleaned_id:
            items.append((cleaned_title or "(untitled)", cleaned_id))

    def _from_structured(array: object) -> List[Tuple[str, str]]:
        structured: List[Tuple[str, str]] = []
        if not isinstance(array, list):
            return structured
        for entry in array:
            if not isinstance(entry, dict):
                continue
            title = (
                entry.get("title")
                or entry.get("video_title")
                or entry.get("name")
                or entry.get("surface")
                or entry.get("text")
                or ""
            )
            video_id = (
                entry.get("id")
                or entry.get("video_id")
                or entry.get("videoId")
                or entry.get("ytid")
                or entry.get("yt_id")
                or entry.get("candidate_id")
                or entry.get("content_id")
                or ""
            )
            structured.append((title, video_id))
        return structured

    items: List[Tuple[str, str]] = []

    for key in ("slate_items", "options", "slate_items_with_meta"):
        structured_items = _from_structured(example.get(key))
        if structured_items:
            for title, video_id in structured_items:
                _append_item(title, video_id)
            break

    if not items:
        slate_text = example.get("slate_text")
        if isinstance(slate_text, str) and slate_text.strip():
            for line in slate_text.splitlines():
                token = line.strip()
                if not token:
                    continue
                token = re.sub(r"^\s*(?:-|\d+\s*[\.\)])\s*", "", token)
                parts = token.split("\t") if "\t" in token else token.split("|", maxsplit=1)
                if len(parts) == 2:
                    title_raw, vid_raw = parts
                else:
                    title_raw, vid_raw = token, ""
                _append_item(title_raw, vid_raw)

    if not items:
        trajectory_json = example.get("trajectory_json")
        if isinstance(trajectory_json, str) and trajectory_json.strip():
            try:
                data = json.loads(trajectory_json)
            except (TypeError, ValueError, json.JSONDecodeError):  # pragma: no cover - defensive
                data = None
            if isinstance(data, dict):
                rows = data.get("order") or data.get("videos") or data.get("history") or []
                if isinstance(rows, list):
                    for entry in rows:
                        if not isinstance(entry, dict):
                            continue
                        raw_id = _pick_ci(
                            entry,
                            "video_id",
                            "id",
                            "videoId",
                            "originId",
                            "content_id",
                        )
                        title = _pick_ci(
                            entry,
                            "title",
                            "video_title",
                            "name",
                            "surface",
                            "text",
                            "videoTitle",
                        )
                        _append_item(title, raw_id)

    if not items:
        return []

    seen: set[str] = set()
    deduped: List[Tuple[str, str]] = []
    for title, video_id in items:
        key = canon_video_id(video_id) or canon_text(title)
        if key:
            if key in seen:
                continue
            seen.add(key)
        deduped.append((title, video_id))
    return deduped


def _format_extra_field(example: dict, field: str) -> str:
    value = example.get(field)
    formatted = format_field_value(field, value)
    if not formatted:
        return ""
    label = EXTRA_FIELD_LABELS.get(field)
    if not label:
        label = field.replace("_", " ").strip().capitalize()
    if field == "child18":
        lowered = formatted.lower()
        if lowered.startswith("no"):
            formatted = "no"
        elif "children" in lowered:
            formatted = "yes"
    return f"{label}: {formatted}"


@dataclass
class PromptDocumentBuilder:
    """Assemble prompt documents and training corpora for baseline models."""

    prompt_column: str
    solution_column: str
    max_history: int
    title_lookup: TitleLookup | None = None
    log_prefix: str = ""
    logger: logging.Logger = field(default_factory=lambda: get_logger("prompt-docs"))

    def _log(self, level: str, message: str, *args: object) -> None:
        log_fn = getattr(self.logger, level)
        prefix = f"{self.log_prefix} " if self.log_prefix else ""
        log_fn(prefix + message, *args)

    def title_for(self, video_id: str) -> Optional[str]:
        if not video_id:
            return None
        if self.title_lookup is None:
            return None
        try:
            return self.title_lookup(video_id)
        except Exception:  # pragma: no cover - defensive
            return None

    def viewer_profile_sentence(self, example: dict) -> str:
        sentence = clean_text(example.get("viewer_profile_sentence"))
        if not sentence:
            sentence = clean_text(example.get("viewer_profile"))
        if not sentence:
            try:
                sentence = synthesize_viewer_sentence(example)
            except ValueError:  # pragma: no cover - defensive
                sentence = ""
        return sentence or ""

    def prompt_from_builder(self, example: dict) -> str:
        existing = example.get(self.prompt_column) or example.get("prompt")
        if isinstance(existing, str):
            stripped = existing.strip()
            if stripped and not _looks_like_legacy_prompt(stripped):
                return stripped
        try:
            return build_user_prompt(example, max_hist=self.max_history)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return ""

    def extract_now_watching(self, example: dict) -> Optional[Tuple[str, str]]:
        return _extract_now_watching(example, self.title_lookup)

    def extract_slate_items(self, example: dict) -> List[Tuple[str, str]]:
        return _extract_slate_items(example, self.title_lookup)

    def assemble_document(
        self,
        example: dict,
        extra_fields: Sequence[str] | None = None,
    ) -> str:
        extra_fields = extra_fields or []

        def _good(text: str) -> bool:
            return bool(text and text.lower() not in {"", "nan", "none", "(none)"})

        parts: List[str] = []
        prompt_text = self.prompt_from_builder(example)
        if not _good(prompt_text):
            fallback_candidates = (
                self.viewer_profile_sentence(example),
                clean_text(example.get(self.prompt_column)),
            )
            prompt_text = next((value for value in fallback_candidates if _good(value)), "")
        if prompt_text:
            parts.append(prompt_text)

        now_watching = self.extract_now_watching(example)
        if now_watching:
            now_title, now_id = now_watching
            if _good(now_title):
                parts.append(now_title)
            if _good(now_id):
                parts.append(now_id)

        for title, video_id in self.extract_slate_items(example):
            surface = (
                title
                if _good(title) and title != "(untitled)"
                else (self.title_for(video_id) or video_id or "")
            )
            if _good(surface):
                parts.append(surface)

        for field in extra_fields:
            formatted = _format_extra_field(example, field)
            if _good(formatted):
                parts.append(formatted)

        return " ".join(parts).strip()

    def _record_from_example(
        self,
        example: dict,
        extra_fields: Sequence[str] | None,
    ) -> Optional[tuple[str, str, str]]:
        document = self.assemble_document(example, extra_fields).strip()
        if not document:
            return None
        video_id = str(example.get(self.solution_column) or "")
        label_id = canon_video_id(video_id)
        label_title = self.title_for(video_id) or ""
        return document, label_id, label_title

    def prepare_training_documents(
        self,
        train_ds: Sequence,
        max_train: int,
        seed: int,
        extra_fields: Sequence[str] | None = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        n_rows = len(train_ds)  # type: ignore[arg-type]
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
            record = self._record_from_example(train_ds[int(index)], extra_fields)
            if record:
                records.append(record)

        if not records:
            raise RuntimeError(
                "All training documents are empty. Check columns on TRAIN split.\n"
                f"Seen columns: {sorted(list(train_ds.features.keys()))}\n"
                "Fixes: add slate items/current video text or pass extra text fields via CLI.",
            )

        dropped = len(order) - len(records)
        if dropped:
            self._log("warning", "Dropped %d empty docs out of %d.", dropped, len(order))

        filtered_docs = [doc for doc, _, _ in records]
        filtered_labels_id = [label_id for _, label_id, _ in records]
        filtered_labels_title = [label_title for _, _, label_title in records]
        self._log("info", "Assembled %d documents (kept %d non-empty).", len(order), len(records))
        self._log("info", "Example doc: %r", filtered_docs[0][:200])
        return filtered_docs, filtered_labels_id, filtered_labels_title


__all__ = [
    "DEFAULT_TITLE_DIRS",
    "EXTRA_FIELD_LABELS",
    "PromptDocumentBuilder",
    "default_title_resolver",
]
