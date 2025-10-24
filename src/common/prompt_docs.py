#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for assembling prompt documents from cleaned GRAIL datasets."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from numpy.random import default_rng

from common import canon_text, canon_video_id, get_logger
from common.prompt_fields import (
    NOW_PLAYING_ID_KEYS,
    NOW_PLAYING_TITLE_KEYS,
    NOW_PLAYING_TITLE_KEYS_WITH_META,
)
from common.title_index import TitleResolver
try:
    from prompt_builder import (
        build_user_prompt,
        clean_text,
        synthesize_viewer_sentence,
        constants as prompt_constants,
        value_maps as prompt_value_maps,
    )
except ImportError as import_error:  # pragma: no cover - handled during import
    raise ImportError("prompt_builder package is required for prompt_docs") from import_error

TitleLookup = Callable[[Optional[str]], Optional[str]]

_TITLE_INDEX_ROOT = (
    "/n/fs/similarity/trees/data/results/"
    "capsule-5416997-data/recommendation trees"
)
DEFAULT_TITLE_DIRS = [
    f"{_TITLE_INDEX_ROOT}/trees_gun",
    f"{_TITLE_INDEX_ROOT}/trees_wage",
]

DEFAULT_EXTRA_TEXT_FIELDS: Tuple[str, ...] = ("viewer_profile", "state_text")

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
EXTRA_FIELD_LABELS.update(prompt_constants.MIN_WAGE_FIELD_LABELS)
EXTRA_FIELD_LABELS.update(prompt_constants.GUN_FIELD_LABELS)

_default_title_resolver_cache: Optional[TitleResolver] = None


def merge_default_extra_fields(extra_fields: Sequence[str] | None) -> Tuple[str, ...]:
    """
    Ensure the default extra text fields are always present.

    :param extra_fields: Caller-provided sequence of extra field names.
    :type extra_fields: Sequence[str] | None
    :returns: Tuple containing the default field list plus any additional ones.
    :rtype: Tuple[str, ...]
    """

    ordered: List[str] = []
    seen: set[str] = set()

    for default_field in DEFAULT_EXTRA_TEXT_FIELDS:
        token = default_field.strip()
        if token and token not in seen:
            ordered.append(token)
            seen.add(token)

    if extra_fields:
        for extra_field_name in extra_fields:
            token = str(extra_field_name or "").strip()
            if token and token not in seen:
                ordered.append(token)
                seen.add(token)

    return tuple(ordered)


def default_title_resolver() -> TitleResolver:
    """

    Return a lazily constructed title resolver for the cleaned GRAIL dataset.



    :returns: Result produced by ``default_title_resolver``.

    :rtype: TitleResolver

    """

    global _default_title_resolver_cache  # pylint: disable=global-statement
    if _default_title_resolver_cache is None:
        _default_title_resolver_cache = TitleResolver(default_dirs=DEFAULT_TITLE_DIRS)
    return _default_title_resolver_cache


def create_prompt_document_builder(
    *,
    prompt_column: str,
    solution_column: str,
    max_history: int,
    log_prefix: str,
    logger_name: str,
) -> "PromptDocumentBuilder":
    """

    Construct a :class:`PromptDocumentBuilder` with shared defaults.



    :param prompt_column: Value provided for ``prompt_column``.

    :type prompt_column: str

    :param solution_column: Value provided for ``solution_column``.

    :type solution_column: str

    :param max_history: Value provided for ``max_history``.

    :type max_history: int

    :param log_prefix: Value provided for ``log_prefix``.

    :type log_prefix: str

    :param logger_name: Value provided for ``logger_name``.

    :type logger_name: str

    :returns: Result produced by ``create_prompt_document_builder``.

    :rtype: 'PromptDocumentBuilder'

    """


    return PromptDocumentBuilder(
        prompt_column=prompt_column,
        solution_column=solution_column,
        max_history=max_history,
        title_lookup=default_title_resolver(),
        log_prefix=log_prefix,
        logger=get_logger(logger_name),
    )


def _looks_like_legacy_prompt(prompt_text: str) -> bool:
    """

    Return ``True`` when ``prompt_text`` matches the legacy prompt layout.



    :param prompt_text: Value provided for ``prompt_text``.

    :type prompt_text: str

    :returns: Result produced by ``_looks_like_legacy_prompt``.

    :rtype: bool

    """


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
    """

    Return the first non-empty value from ``mapping`` matching ``alternates``.



    :param mapping: Value provided for ``mapping``.

    :type mapping: dict

    :param alternates: Value provided for ``alternates``.

    :type alternates: str

    :returns: Result produced by ``_pick_ci``.

    :rtype: Optional[str]

    """


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
    """

    Return ``True`` when ``value`` should be treated as a missing token.



    :param value: Value provided for ``value``.

    :type value: object

    :returns: Result produced by ``_is_nanlike``.

    :rtype: bool

    """


    if value is None:
        return True
    string = str(value).strip().lower()
    return string in {"", "nan", "none", "null", "na", "n/a"}


def load_trajectory_entries(payload: object) -> List[Mapping[str, object]]:
    """

    Return sanitized trajectory entries extracted from ``payload``.



    :param payload: Value provided for ``payload``.

    :type payload: object

    :returns: Result produced by ``load_trajectory_entries``.

    :rtype: List[Mapping[str, object]]

    """


    if isinstance(payload, str) and payload.strip():
        try:
            data = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):  # pragma: no cover - defensive
            return []
    elif isinstance(payload, Mapping):
        data = payload
    else:
        return []

    if not isinstance(data, Mapping):
        return []
    rows = data.get("order") or data.get("videos") or data.get("history") or []
    if not isinstance(rows, Sequence):
        return []

    entries: List[Mapping[str, object]] = []
    for entry in rows:
        if isinstance(entry, Mapping):
            entries.append(entry)
    return entries


def _extract_now_watching(
    example: dict,
    title_lookup: TitleLookup | None,
) -> Optional[Tuple[str, str]]:
    """

    Return ``(title, video_id)`` describing the currently watched video.



    :param example: Value provided for ``example``.

    :type example: dict

    :param title_lookup: Value provided for ``title_lookup``.

    :type title_lookup: TitleLookup | None

    :returns: Result produced by ``_extract_now_watching``.

    :rtype: Optional[Tuple[str, str]]

    """


    video_id = _pick_ci(example, "video_id", "videoId")
    if video_id and not _is_nanlike(video_id):
        title = _pick_ci(example, *NOW_PLAYING_TITLE_KEYS_WITH_META)
        if _is_nanlike(title) and title_lookup is not None:
            title = title_lookup(video_id)
        return (title or "(untitled)", str(video_id))
    title = _pick_ci(example, *NOW_PLAYING_TITLE_KEYS)
    video_id = _pick_ci(example, *NOW_PLAYING_ID_KEYS)
    if (title and not _is_nanlike(title)) or (video_id and not _is_nanlike(video_id)):
        if _is_nanlike(title) and title_lookup is not None:
            title = title_lookup(video_id) or ""
        return (title or "(untitled)", video_id or "")
    return None


@dataclass
class _SlateCollector:
    """Accumulate cleaned slate entries while minimising local state.

    :param title_lookup: Optional callback used to hydrate missing titles.
    :type title_lookup: TitleLookup | None
    :param items: Mutable list of ``(title, video_id)`` pairs collected so far.
    :type items: list[tuple[str, str]]
    """

    title_lookup: TitleLookup | None
    items: List[Tuple[str, str]] = field(default_factory=list)

    def add(self, title: object, video_id: object) -> None:
        """Append a cleaned ``(title, video_id)`` pair to ``items`` when viable.

        :param title: Candidate slate title recovered from the source data.
        :type title: object
        :param video_id: Candidate video identifier to normalise.
        :type video_id: object
        """
        cleaned_id = _normalise_video_id(video_id)
        cleaned_title = _normalise_title(title)
        if not cleaned_id and isinstance(title, str):
            possible_id = _normalise_video_id(title)
            if possible_id:
                cleaned_id = possible_id
                cleaned_title = ""
        if not cleaned_title and cleaned_id and self.title_lookup is not None:
            cleaned_title = self.title_lookup(cleaned_id) or ""
        if cleaned_title or cleaned_id:
            self.items.append((cleaned_title or "(untitled)", cleaned_id))


def _normalise_title(value: object) -> str:
    """Return a stripped title when ``value`` is a string.

    :param value: Raw value that may contain title text.
    :type value: object
    :returns: Cleaned title string or an empty string when unavailable.
    :rtype: str
    """
    return value.strip() if isinstance(value, str) else ""


def _normalise_video_id(value: object) -> str:
    """Return an 11-character YouTube id when ``value`` resembles one.

    :param value: Raw identifier value extracted from the dataset.
    :type value: object
    :returns: Canonical YouTube video id or an empty string if cleaning fails.
    :rtype: str
    """
    if not value:
        return ""
    candidate = canon_video_id(str(value))
    return candidate if len(candidate) == 11 else ""


def _structured_slate_candidates(raw: object) -> List[Tuple[object, object]]:
    """Extract raw ``(title, id)`` pairs from structured slate metadata.

    :param raw: Structured slate metadata, typically a list of dictionaries.
    :type raw: object
    :returns: Sequence of un-normalised ``(title, video_id)`` tuples.
    :rtype: list[tuple[object, object]]
    """
    if not isinstance(raw, list):
        return []
    pairs: List[Tuple[object, object]] = []
    for entry in raw:
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
        pairs.append((title, video_id))
    return pairs


def _collect_structured_items(collector: _SlateCollector, example: Mapping[str, object]) -> bool:
    """Attempt to populate ``collector`` using structured slate arrays.

    :param collector: Aggregator used to store cleaned slate entries.
    :type collector: _SlateCollector
    :param example: Source example containing potential slate metadata.
    :type example: Mapping[str, object]
    :returns: ``True`` when any structured slate entries were ingested.
    :rtype: bool
    """
    for key in ("slate_items", "options", "slate_items_with_meta"):
        candidates = _structured_slate_candidates(example.get(key))
        if not candidates:
            continue
        for title, video_id in candidates:
            collector.add(title, video_id)
        return True
    return False


def _collect_text_items(collector: _SlateCollector, slate_text: object) -> bool:
    """Parse textual slate descriptions and append any recovered pairs.

    :param collector: Aggregator used to store cleaned slate entries.
    :type collector: _SlateCollector
    :param slate_text: Raw textual description of slate candidates.
    :type slate_text: object
    :returns: ``True`` if one or more entries were appended to the collector.
    :rtype: bool
    """
    if not isinstance(slate_text, str) or not slate_text.strip():
        return False
    for line in slate_text.splitlines():
        token = line.strip()
        if not token:
            continue
        token = re.sub(r"^\s*(?:-|\d+\s*[\.\)])\s*", "", token)
        parts = token.split("\t") if "\t" in token else token.split("|", maxsplit=1)
        title_raw, vid_raw = (parts[0], parts[1]) if len(parts) == 2 else (token, "")
        collector.add(title_raw, vid_raw)
    return bool(collector.items)


def _collect_trajectory_items(collector: _SlateCollector, trajectory_json: object) -> None:
    """Fallback that scans trajectory entries when slate metadata is absent.

    :param collector: Aggregator used to store cleaned slate entries.
    :type collector: _SlateCollector
    :param trajectory_json: Raw trajectory payload that may contain slate hints.
    :type trajectory_json: object
    """
    for entry in load_trajectory_entries(trajectory_json):
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
        collector.add(title, raw_id)


def _deduplicate_slate_items(items: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Remove duplicate slate entries while preserving order.

    :param items: Slate entries emitted by the collector.
    :type items: Sequence[tuple[str, str]]
    :returns: Ordered list with duplicate titles or ids removed.
    :rtype: list[tuple[str, str]]
    """
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


def _extract_slate_items(
    example: Mapping[str, object],
    title_lookup: TitleLookup | None,
) -> List[Tuple[str, str]]:
    """Return a list of ``(title, video_id)`` tuples extracted from ``example``.

    :param example: Source record containing slate information in various forms.
    :type example: Mapping[str, object]
    :param title_lookup: Optional callback used to hydrate missing titles.
    :type title_lookup: TitleLookup | None
    :returns: Normalised ``(title, video_id)`` tuples ready for downstream use.
    :rtype: list[tuple[str, str]]
    """

    collector = _SlateCollector(title_lookup)
    if _collect_structured_items(collector, example):
        return _deduplicate_slate_items(collector.items)
    if _collect_text_items(collector, example.get("slate_text")):
        return _deduplicate_slate_items(collector.items)
    _collect_trajectory_items(collector, example.get("trajectory_json"))
    if not collector.items:
        return []
    return _deduplicate_slate_items(collector.items)


def _format_extra_field(example: dict, field_name: str) -> str:
    """

    Return a labelled, human-readable representation of an extra field.



    :param example: Value provided for ``example``.

    :type example: dict

    :param field_name: Value provided for ``field_name``.

    :type field_name: str

    :returns: Result produced by ``_format_extra_field``.

    :rtype: str

    """


    value = example.get(field_name)
    formatted = prompt_value_maps.format_field_value(field_name, value)
    if not formatted:
        return ""
    label = EXTRA_FIELD_LABELS.get(field_name)
    if not label:
        label = field_name.replace("_", " ").strip().capitalize()
    if field_name == "child18":
        lowered = formatted.lower()
        if lowered.startswith("no"):
            formatted = "no"
        elif "children" in lowered:
            formatted = "yes"
    return f"{label}: {formatted}"


@dataclass
class PromptDocumentBuilder:
    """

    Assemble prompt documents and training corpora for baseline models.



    :ivar prompt_column: Attribute ``prompt_column``.

    :vartype prompt_column: str

    :ivar solution_column: Attribute ``solution_column``.

    :vartype solution_column: str

    :ivar max_history: Attribute ``max_history``.

    :vartype max_history: int

    :ivar title_lookup: Attribute ``title_lookup``.

    :vartype title_lookup: TitleLookup | None

    :ivar log_prefix: Attribute ``log_prefix``.

    :vartype log_prefix: str

    :ivar logger: Attribute ``logger``.

    :vartype logger: logging.Logger

    """


    prompt_column: str
    solution_column: str
    max_history: int
    title_lookup: TitleLookup | None = None
    log_prefix: str = ""
    logger: logging.Logger = field(default_factory=lambda: get_logger("prompt-docs"))

    def _log(self, level: str, message: str, *args: object) -> None:
        """

        Emit a log message while respecting the configured prefix.



        :param level: Value provided for ``level``.

        :type level: str

        :param message: Value provided for ``message``.

        :type message: str

        :param args: Value provided for ``args``.

        :type args: object

        :returns: ``None``.

        :rtype: None

        """


        log_fn = getattr(self.logger, level)
        prefix = f"{self.log_prefix} " if self.log_prefix else ""
        log_fn(prefix + message, *args)

    def title_for(self, video_id: str) -> Optional[str]:
        """

        Return the title associated with ``video_id`` when available.



        :param video_id: Value provided for ``video_id``.

        :type video_id: str

        :returns: Result produced by ``title_for``.

        :rtype: Optional[str]

        """


        if not video_id:
            return None
        if self.title_lookup is None:
            return None
        try:
            return self.title_lookup(video_id)
        except (LookupError, RuntimeError, ValueError):  # pragma: no cover - defensive
            return None

    def viewer_profile_sentence(self, example: dict) -> str:
        """

        Return a cleaned viewer profile sentence, synthesising when needed.



        :param example: Value provided for ``example``.

        :type example: dict

        :returns: Result produced by ``viewer_profile_sentence``.

        :rtype: str

        """


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
        """

        Return an existing prompt or fall back to :func:`build_user_prompt`.



        :param example: Value provided for ``example``.

        :type example: dict

        :returns: Result produced by ``prompt_from_builder``.

        :rtype: str

        """


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
        """

        Return the now-watching tuple (title, id) when present.



        :param example: Value provided for ``example``.

        :type example: dict

        :returns: Result produced by ``extract_now_watching``.

        :rtype: Optional[Tuple[str, str]]

        """


        return _extract_now_watching(example, self.title_lookup)

    def extract_slate_items(self, example: dict) -> List[Tuple[str, str]]:
        """

        Return the slate items as a list of ``(title, id)`` pairs.



        :param example: Value provided for ``example``.

        :type example: dict

        :returns: Result produced by ``extract_slate_items``.

        :rtype: List[Tuple[str, str]]

        """


        return _extract_slate_items(example, self.title_lookup)

    def assemble_document(
        self,
        example: dict,
        extra_fields: Sequence[str] | None = None,
    ) -> str:
        """

        Assemble a whitespace-joined prompt document for slate modelling.



        :param example: Value provided for ``example``.

        :type example: dict

        :param extra_fields: Value provided for ``extra_fields``.

        :type extra_fields: Sequence[str] | None

        :returns: Result produced by ``assemble_document``.

        :rtype: str

        """


        # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks,too-many-statements
        extra_fields = extra_fields or []

        def _good(text: str) -> bool:
            """
            Determine whether ``text`` is a meaningful, non-placeholder string.

            :param text: String candidate extracted from the dataset.
            :type text: str
            :returns: ``True`` when ``text`` contains informative content.
            :rtype: bool
            """

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

        for field_name in extra_fields:
            formatted = _format_extra_field(example, field_name)
            if _good(formatted):
                parts.append(formatted)

        return " ".join(parts).strip()

    def _record_from_example(
        self,
        example: dict,
        extra_fields: Sequence[str] | None,
    ) -> Optional[tuple[str, str, str]]:
        """

        Return the document and label tuple extracted from ``example``.



        :param example: Value provided for ``example``.

        :type example: dict

        :param extra_fields: Value provided for ``extra_fields``.

        :type extra_fields: Sequence[str] | None

        :returns: Result produced by ``_record_from_example``.

        :rtype: Optional[tuple[str, str, str]]

        """


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
        """

        Return filtered training documents and labels sampled from ``train_ds``.



        :param train_ds: Value provided for ``train_ds``.

        :type train_ds: Sequence

        :param max_train: Value provided for ``max_train``.

        :type max_train: int

        :param seed: Value provided for ``seed``.

        :type seed: int

        :param extra_fields: Value provided for ``extra_fields``.

        :type extra_fields: Sequence[str] | None

        :returns: Result produced by ``prepare_training_documents``.

        :rtype: Tuple[List[str], List[str], List[str]]

        """


        # pylint: disable=too-many-locals
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
    "DEFAULT_EXTRA_TEXT_FIELDS",
    "EXTRA_FIELD_LABELS",
    "PromptDocumentBuilder",
    "create_prompt_document_builder",
    "default_title_resolver",
    "load_trajectory_entries",
    "merge_default_extra_fields",
]
