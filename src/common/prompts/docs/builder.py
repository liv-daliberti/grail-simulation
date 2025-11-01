"""Prompt document builder facade."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from common import canon_video_id, get_logger
# Import from explicit submodules so static analyzers (pylint) can resolve names.
from prompt_builder.formatters import clean_text
from prompt_builder.prompt import build_user_prompt
from prompt_builder.profiles.render import synthesize_viewer_sentence

from .extra_fields import format_extra_field
from .slate import extract_now_watching, extract_slate_items
from .titles import TitleLookup, default_title_resolver
from ..sampling import collect_selected_examples


def _looks_like_legacy_prompt(prompt_text: str) -> bool:
    """Return ``True`` when ``prompt_text`` matches the legacy prompt layout."""
    legacy_tokens = (
        "PROFILE:",
        "ATTRIBUTES:",
        "CURRENT VIDEO:",
        "RECENTLY WATCHED (NEWEST LAST):",
        "OPTIONS:",
        "SURVEY HIGHLIGHTS:",
    )
    return any(token in prompt_text for token in legacy_tokens)


@dataclass
class PromptDocumentBuilder:
    """
    Assemble prompt documents and training corpora for baseline models.

    :ivar prompt_column: Source column containing raw prompt text.
    :vartype prompt_column: str
    :ivar solution_column: Column storing the ground-truth label identifier.
    :vartype solution_column: str
    :ivar max_history: Maximum number of historical events considered by prompt_builder.
    :vartype max_history: int
    :ivar title_lookup: Optional callback used to hydrate missing titles.
    :vartype title_lookup: TitleLookup | None
    :ivar log_prefix: Human-readable prefix injected ahead of log messages.
    :vartype log_prefix: str
    :ivar logger: Logger instance used for diagnostic output.
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

        :param level: Logging level name such as ``"info"`` or ``"warning"``.
        :type level: str
        :param message: Format string describing the event to log.
        :type message: str
        :param args: Optional positional arguments interpolated into ``message``.
        :type args: object
        """
        log_fn = getattr(self.logger, level)
        prefix = f"{self.log_prefix} " if self.log_prefix else ""
        log_fn(prefix + message, *args)

    def title_for(self, video_id: str) -> Optional[str]:
        """
        Return the title associated with ``video_id`` when available.

        :param video_id: Candidate YouTube identifier to resolve.
        :type video_id: str
        :returns: Human-readable title or ``None`` if unavailable.
        :rtype: Optional[str]
        """
        if not video_id or self.title_lookup is None:
            return None
        try:
            return self.title_lookup(video_id)
        except (LookupError, RuntimeError, ValueError):  # pragma: no cover - defensive
            return None

    def viewer_profile_sentence(self, example: dict) -> str:
        """
        Return a cleaned viewer profile sentence, synthesising when needed.

        :param example: Dataset row providing viewer metadata.
        :type example: dict
        :returns: Canonicalised viewer profile summary or the empty string.
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

        :param example: Dataset row containing prompt and history columns.
        :type example: dict
        :returns: Cleaned prompt string, possibly synthesised from history.
        :rtype: str
        """
        # Prefer generating the full prompt via prompt_builder to ensure
        # consistency across pipelines. Fall back to any existing prompt only
        # if prompt generation fails.
        try:
            built = build_user_prompt(example, max_hist=self.max_history)
            if isinstance(built, str) and built.strip():
                return built.strip()
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
        existing = example.get(self.prompt_column) or example.get("prompt")
        if isinstance(existing, str):
            stripped = existing.strip()
            if stripped and not _looks_like_legacy_prompt(stripped):
                return stripped
        return ""

    def extract_now_watching(self, example: dict) -> Optional[Tuple[str, str]]:
        """
        Return the now-watching tuple ``(title, video_id)`` when present.

        :param example: Dataset row containing candidate now-watching data.
        :type example: dict
        :returns: Two-tuple of title and video id or ``None`` if unavailable.
        :rtype: Optional[Tuple[str, str]]
        """
        return extract_now_watching(example, self.title_lookup)

    def extract_slate_items(self, example: dict) -> List[Tuple[str, str]]:
        """
        Return the slate items as a list of ``(title, video_id)`` pairs.

        :param example: Dataset row containing structured or textual slate metadata.
        :type example: dict
        :returns: Ordered, de-duplicated slate entries ready for downstream use.
        :rtype: List[Tuple[str, str]]
        """
        return extract_slate_items(example, self.title_lookup)

    def assemble_document(
        self,
        example: dict,
        extra_fields: Sequence[str] | None = None,
    ) -> str:
        """
        Assemble a whitespace-joined prompt document for slate modelling.

        :param example: Dataset row providing prompt, slate, and solution fields.
        :type example: dict
        :param extra_fields: Additional viewer attributes to embed in the prompt.
        :type extra_fields: Sequence[str] | None
        :returns: Normalised prompt text suitable for training or inference.
        :rtype: str
        """
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
            # Emit a one-time log of the full prompt text to aid debugging.
            try:
                if not hasattr(self, "_logged_full_prompt_example") or not getattr(
                    self, "_logged_full_prompt_example"
                ):
                    self._log("info", "Full prompt example: %s", prompt_text)
                    setattr(self, "_logged_full_prompt_example", True)
            except Exception:  # pylint: disable=broad-except  # pragma: no cover - best-effort logging
                pass

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
            formatted = format_extra_field(example, field_name)
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

        :param example: Dataset row providing prompt and label columns.
        :type example: dict
        :param extra_fields: Optional sequence of extra field names to include.
        :type extra_fields: Sequence[str] | None
        :returns: Tuple ``(document, label_id, label_title)`` or ``None``.
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

        :param train_ds: HuggingFace-style dataset or sequence of rows to process.
        :type train_ds: Sequence
        :param max_train: Maximum number of training examples to inspect.
        :type max_train: int
        :param seed: Random seed controlling subsampling.
        :type seed: int
        :param extra_fields: Optional extra field names to include in the prompt.
        :type extra_fields: Sequence[str] | None
        :returns: Tuple of documents, label ids, and label titles.
        :rtype: Tuple[List[str], List[str], List[str]]
        :raises RuntimeError: If all candidate prompts are empty after processing.
        """
        indices, records = collect_selected_examples(
            train_ds,
            max_train=max_train,
            seed=seed,
            collect=lambda _, example: self._record_from_example(example, extra_fields),
        )

        if not records:
            raise RuntimeError(
                "All training documents are empty. Check columns on TRAIN split.\n"
                f"Seen columns: {sorted(list(train_ds.features.keys()))}\n"
                "Fixes: add slate items/current video text or pass extra text fields via CLI.",
            )

        dropped = len(indices) - len(records)
        if dropped:
            self._log("warning", "Dropped %d empty docs out of %d.", dropped, len(indices))

        filtered_docs = [doc for doc, _, _ in records]
        filtered_labels_id = [label_id for _, label_id, _ in records]
        filtered_labels_title = [label_title for _, _, label_title in records]
        self._log("info", "Assembled %d documents (kept %d non-empty).", len(indices), len(records))
        self._log("info", "Example doc: %r", filtered_docs[0])
        return filtered_docs, filtered_labels_id, filtered_labels_title


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

    :param prompt_column: Dataset column that stores the raw prompt text.
    :type prompt_column: str
    :param solution_column: Column providing the ground-truth label identifier.
    :type solution_column: str
    :param max_history: Maximum prompt history length considered by prompt_builder.
    :type max_history: int
    :param log_prefix: Prefix added to log messages emitted by the builder.
    :type log_prefix: str
    :param logger_name: Name of the logger to retrieve via :func:`get_logger`.
    :type logger_name: str
    :returns: Fully configured :class:`PromptDocumentBuilder` instance.
    :rtype: PromptDocumentBuilder
    """
    return PromptDocumentBuilder(
        prompt_column=prompt_column,
        solution_column=solution_column,
        max_history=max_history,
        title_lookup=default_title_resolver(),
        log_prefix=log_prefix,
        logger=get_logger(logger_name),
    )


__all__ = ["PromptDocumentBuilder", "create_prompt_document_builder"]
