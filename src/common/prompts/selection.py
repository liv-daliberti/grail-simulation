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

"""Shared prompt selection utilities for baseline recommenders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from common.text import canon_text, canon_video_id
from .docs import PromptDocumentBuilder
from .sampling import collect_selected_examples


@dataclass(frozen=True)
class CandidateMetadata:
    """Lightweight view of slate candidate attributes used for token augmentation."""

    slot: int
    title: str
    video_id: str
    channel_title: Optional[str] = None
    channel_id: Optional[str] = None


def _canon_token(value: object) -> str:
    """
    Return a lowercase alphanumeric token suitable for feature injection.

    :param value: Arbitrary value converted into a token.
    :returns: Normalised token or ``\"\"`` when the value is empty.
    """

    if not value:
        return ""
    return canon_text(str(value))


def _parse_gold_index(raw_value: object) -> int:
    """
    Return ``gold_index`` coerced to ``int`` or ``-1`` when unavailable.

    :param raw_value: Raw gold index extracted from the example payload.
    :returns: Zero-based gold index or ``-1`` when parsing fails.
    """

    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return -1


def _context_tokens(
    example: Mapping[str, object],
    *,
    option_count: int,
    gold_index: int,
) -> List[str]:
    """
    Return tokens describing high-level selection context.

    :param example: Prompt example providing metadata fields.
    :param option_count: Number of slate candidates observed.
    :param gold_index: One-based gold index; values <=0 indicate unknown.
    :returns: List of context tokens highlighting the study and issue.
    """

    tokens: List[str] = []
    if option_count:
        tokens.append(f"options_token_{option_count}")

    issue = _canon_token(example.get("issue"))
    if issue:
        tokens.append(f"issue_token_{issue}")

    study = _canon_token(example.get("participant_study"))
    if study:
        tokens.append(f"study_token_{study}")

    if gold_index > 0:
        tokens.append(f"slot_token_{gold_index}")

    return tokens


def _selected_candidate_tokens(metadata: CandidateMetadata) -> List[str]:
    """
    Return tokens highlighting the selected candidate.

    :param metadata: Candidate metadata describing the selected item.
    :returns: List of tokens without positional slot markers.
    """

    tokens = candidate_feature_tokens(metadata, option_count=None)
    return [token for token in tokens if not token.startswith("slot_token_")]


def _selected_candidate_metadata(
    example: Mapping[str, object],
    candidates: Sequence[CandidateMetadata],
    gold_index: int,
    solution_column: str,
) -> Optional[CandidateMetadata]:
    """
    Return metadata describing the selected candidate when available.

    :param example: Prompt example including the solution column.
    :param candidates: Sequence of known candidate metadata entries.
    :param gold_index: One-based index indicating the correct candidate.
    :param solution_column: Name of the column tracking the selected video ID.
    :returns: Candidate metadata for the gold item, or ``None`` if unknown.
    """

    selected_video = str(example.get(solution_column) or "")

    if 1 <= gold_index <= len(candidates):
        candidate = candidates[gold_index - 1]
        video_id = candidate.video_id or selected_video
        return CandidateMetadata(
            slot=candidate.slot,
            title=candidate.title or "",
            video_id=video_id or "",
            channel_title=candidate.channel_title,
            channel_id=candidate.channel_id,
        )

    if not selected_video:
        return None

    return CandidateMetadata(
        slot=gold_index if gold_index > 0 else 0,
        title="",
        video_id=selected_video,
    )


def candidate_feature_tokens(
    candidate: CandidateMetadata,
    *,
    option_count: int | None = None,
) -> List[str]:
    """
    Generate candidate-specific tokens used during query scoring.

    :param candidate: Metadata describing a slate candidate.
    :param option_count: Optional total number of options available in the slate.
    :returns: List of tokens encoding candidate identifiers and attributes.
    """

    tokens: List[str] = []
    if option_count:
        tokens.append(f"options_token_{option_count}")
    if candidate.slot > 0:
        tokens.append(f"slot_token_{candidate.slot}")

    video_token = canon_video_id(candidate.video_id)
    if video_token:
        tokens.append(f"video_token_{video_token}")

    title_token = _canon_token(candidate.title)
    if title_token:
        tokens.append(f"title_token_{title_token}")

    if candidate.channel_id:
        channel_id_token = _canon_token(candidate.channel_id)
        if channel_id_token:
            tokens.append(f"channelid_token_{channel_id_token}")
    elif candidate.channel_title:
        channel_token = _canon_token(candidate.channel_title)
        if channel_token:
            tokens.append(f"channel_token_{channel_token}")

    return [token for token in tokens if token]


@dataclass
class PromptSelectionHelper:
    """Wrap the prompt document builder with shared selection helpers.

    See :class:`~common.prompts.docs.builder.PromptDocumentBuilder` for details.
    """

    builder: PromptDocumentBuilder

    @property
    def solution_column(self) -> str:
        """
        Expose the solution column tracked by the prompt builder.

        :returns: Name of the solution column used to identify the gold item.
        """

        return self.builder.solution_column

    def title_for(self, video_id: str) -> Optional[str]:
        """
        Look up a human-readable title for ``video_id``.

        :param video_id: Candidate video identifier.
        :returns: Human-friendly title or ``None`` when unavailable.
        """

        return self.builder.title_for(video_id)

    def viewer_profile_sentence(self, example: Mapping[str, object]) -> str:
        """
        Compose the viewer profile sentence associated with ``example``.

        :param example: Prompt example containing viewer metadata.
        :returns: Viewer profile sentence assembled by the builder.
        """

        return self.builder.viewer_profile_sentence(dict(example))

    def prompt_from_builder(self, example: Mapping[str, object]) -> str:
        """
        Assemble the full prompt text for ``example``.

        :param example: Prompt example to convert into prompt text.
        :returns: Full prompt string including contextual tokens.
        """

        return self.builder.prompt_from_builder(dict(example))

    def extract_now_watching(
        self, example: Mapping[str, object]
    ) -> Optional[Tuple[str, str]]:
        """
        Retrieve the currently watched item for ``example``.

        :param example: Prompt example containing slate metadata.
        :returns: Tuple of title and video ID, or ``None`` if absent.
        """

        return self.builder.extract_now_watching(dict(example))

    def extract_slate_items(self, example: Mapping[str, object]) -> List[Tuple[str, str]]:
        """
        Extract the slate of candidate items from ``example``.

        :param example: Prompt example containing slate candidates.
        :returns: List of ``(title, video_id)`` pairs.
        """

        return self.builder.extract_slate_items(dict(example))

    def collect_candidate_metadata(
        self, example: Mapping[str, object]
    ) -> List[CandidateMetadata]:
        """
        Return slate candidate metadata enriched with slot indices and channel info.

        :param example: Prompt example providing slate items and metadata.
        :returns: List of
            :class:`~common.prompts.selection.CandidateMetadata` entries for the slate.
        """

        pairs = self.extract_slate_items(example)
        meta_lookup: dict[str, Mapping[str, object]] = {}
        raw_meta = example.get("slate_items_with_meta")
        if isinstance(raw_meta, Sequence):
            for item in raw_meta:
                if not isinstance(item, Mapping):
                    continue
                vid = str(item.get("id") or "")
                if vid:
                    meta_lookup[vid] = item

        enriched: List[CandidateMetadata] = []
        for slot, (title, video_id) in enumerate(pairs, start=1):
            mapped = meta_lookup.get(video_id, {})
            channel_title = mapped.get("channel_title") or mapped.get("channelTitle")
            channel_id = mapped.get("channel_id") or mapped.get("channelId")
            enriched.append(
                CandidateMetadata(
                    slot=slot,
                    title=title or "",
                    video_id=video_id or "",
                    channel_title=str(channel_title or "") or None,
                    channel_id=str(channel_id or "") or None,
                )
            )
        return enriched

    def candidate_feature_tokens(
        self,
        candidate: CandidateMetadata,
        *,
        option_count: int | None = None,
    ) -> List[str]:
        """
        Delegate to :func:`candidate_feature_tokens` for convenience.

        :param candidate: Slate candidate to transform into feature tokens.
        :param option_count: Optional total number of options available.
        :returns: List of tokens describing the candidate.
        """

        return candidate_feature_tokens(candidate, option_count=option_count)

    def selection_feature_tokens(
        self,
        example: Mapping[str, object],
        candidates: Sequence[CandidateMetadata] | None = None,
    ) -> List[str]:
        """
        Construct additive tokens that highlight the selected candidate within a prompt.

        :param example: Prompt example containing the gold candidate.
        :param candidates: Precomputed candidate metadata, or ``None`` to derive it.
        :returns: List of tokens emphasising the gold candidate.
        """

        if candidates is None:
            candidates = self.collect_candidate_metadata(example)

        gold_index = _parse_gold_index(example.get("gold_index"))
        tokens = _context_tokens(
            example,
            option_count=len(candidates),
            gold_index=gold_index,
        )

        selected = _selected_candidate_metadata(
            example,
            candidates,
            gold_index,
            self.solution_column,
        )
        if selected is not None:
            candidate_tokens = _selected_candidate_tokens(selected)
            if candidate_tokens:
                tokens.extend(candidate_tokens)

        return tokens

    def assemble_document(
        self,
        example: Mapping[str, object],
        extra_fields: Sequence[str] | None = None,
    ) -> str:
        """
        Concatenate prompt components and selection tokens for feature extraction.

        :param example: Prompt example containing base fields and slate metadata.
        :param extra_fields: Optional additional fields injected into the prompt.
        :returns: Document string suitable for downstream vectorisation.
        """

        base_document = self.builder.assemble_document(dict(example), extra_fields)
        candidates = self.collect_candidate_metadata(example)
        tokens = self.selection_feature_tokens(example, candidates)
        if tokens:
            token_text = " ".join(tokens)
            if base_document:
                return f"{base_document}\n{token_text}"
            return token_text
        return base_document

    def prepare_training_documents(
        self,
        train_ds,
        max_train: int,
        seed: int,
        extra_fields: Sequence[str] | None = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Prepare prompt documents, labels, and metadata for TF-IDF style training.

        :param train_ds: Training dataset consumed by :func:`collect_selected_examples`.
        :param max_train: Maximum number of training examples to include.
        :param seed: Random seed controlling subsampling determinism.
        :param extra_fields: Optional additional fields appended to the prompt.
        :returns: Tuple of documents, label IDs, and label titles.
        :raises RuntimeError: If no eligible documents are generated.
        """

        # pylint: disable=too-many-locals
        def _collect(_, example: Mapping[str, object]) -> Tuple[str, str, str] | None:
            """
            Transform a raw example into a prompt document and label triple.

            :param _: Index of the example within the dataset (unused).
            :param example: Example containing slate and viewer metadata.
            :returns: Tuple of document text, video ID, and label title, or ``None``.
            """
            document = self.assemble_document(example, extra_fields)
            if not document.strip():
                return None

            video_raw = example.get(self.solution_column)
            video_id = canon_video_id(video_raw)
            if not video_id:
                return None

            candidates = self.collect_candidate_metadata(example)
            gold_index = _parse_gold_index(example.get("gold_index"))

            label_title = self.title_for(video_id) or ""
            if 1 <= gold_index <= len(candidates):
                selected = candidates[gold_index - 1]
                if selected.title and not label_title:
                    label_title = selected.title

            return document, video_id, label_title

        indices, triples = collect_selected_examples(
            train_ds,
            max_train=max_train,
            seed=seed,
            collect=_collect,
        )

        if not triples:
            raise RuntimeError("No eligible documents were generated for training.")

        documents, label_ids, label_titles = zip(*triples)

        # Emit a concise summary and an example prompt to aid debugging.
        try:
            dropped = len(indices) - len(triples)
            prefix = f"{self.builder.log_prefix} " if self.builder.log_prefix else ""
            self.builder.logger.info(
                "%sAssembled %d selection-aware documents (kept %d non-empty).",
                prefix,
                len(indices),
                len(triples),
            )
            if documents:
                sample = str(documents[0])
                self.builder.logger.info("%sExample doc: %r", prefix, sample)
            if dropped > 0:
                self.builder.logger.warning(
                    "%sDropped %d empty docs out of %d.",
                    prefix,
                    dropped,
                    len(indices),
                )
        except (
            TypeError,
            ValueError,
            AttributeError,
            RuntimeError,
            IndexError,
        ):
            # pragma: no cover - best-effort logging
            # Best-effort logging; continue silently if unexpected formatting issues occur.
            pass

        return list(documents), list(label_ids), list(label_titles)

    def prepare_prompt_documents(
        self,
        train_ds,
        max_train: int,
        seed: int,
        extra_fields: Sequence[str] | None = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Convenience wrapper mirroring :meth:`prepare_training_documents`.

        :param train_ds: Training dataset consumed by :func:`collect_selected_examples`.
        :param max_train: Maximum number of training examples to include.
        :param seed: Random seed controlling subsampling determinism.
        :param extra_fields: Optional additional fields appended to the prompt.
        :returns: Tuple of documents, label IDs, and label titles.
        """

        return self.prepare_training_documents(
            train_ds,
            max_train=max_train,
            seed=seed,
            extra_fields=extra_fields,
        )


def _unbound_export(name: str) -> RuntimeError:
    """
    Return a helpful error when prompt-selection helpers are not yet bound.

    :param name: Name of the helper attribute expected by the caller.
    :returns: RuntimeError describing the missing binding.
    """

    return RuntimeError(
        f"Prompt selection helper '{name}' is not bound. "
        "Call 'bind_prompt_selection_exports' to attach runtime implementations."
    )


def title_for(_video_id: str) -> Optional[str]:
    """
    Placeholder bound via :func:`bind_prompt_selection_exports`.

    :param _video_id: Video identifier requested by the caller.
    :returns: Title string once bound; raises otherwise.
    """

    raise _unbound_export("title_for") from None


def viewer_profile_sentence(_example: Mapping[str, object]) -> str:
    """
    Placeholder bound via :func:`bind_prompt_selection_exports`.

    :param _example: Prompt example passed to the helper.
    :returns: Viewer profile sentence once bound; raises otherwise.
    """

    raise _unbound_export("viewer_profile_sentence") from None


def prompt_from_builder(_example: Mapping[str, object]) -> str:
    """
    Placeholder bound via :func:`bind_prompt_selection_exports`.

    :param _example: Prompt example passed to the helper.
    :returns: Prompt string once bound; raises otherwise.
    """

    raise _unbound_export("prompt_from_builder") from None


def extract_now_watching(
    _example: Mapping[str, object],
) -> Optional[Tuple[str, str]]:
    """
    Placeholder bound via :func:`bind_prompt_selection_exports`.

    :param _example: Prompt example passed to the helper.
    :returns: Tuple of title and video ID once bound; raises otherwise.
    """

    raise _unbound_export("extract_now_watching") from None


def extract_slate_items(_example: Mapping[str, object]) -> List[Tuple[str, str]]:
    """
    Placeholder bound via :func:`bind_prompt_selection_exports`.

    :param _example: Prompt example passed to the helper.
    :returns: List of ``(title, video_id)`` pairs once bound; raises otherwise.
    """

    raise _unbound_export("extract_slate_items") from None


def collect_candidate_metadata(_example: Mapping[str, object]) -> List[CandidateMetadata]:
    """
    Placeholder bound via :func:`bind_prompt_selection_exports`.

    :param _example: Prompt example passed to the helper.
    :returns: List of
        :class:`~common.prompts.selection.CandidateMetadata` entries once bound;
        raises otherwise.
    """

    raise _unbound_export("collect_candidate_metadata") from None


def selection_feature_tokens(
    _example: Mapping[str, object],
    _candidates: Sequence[CandidateMetadata] | None = None,
) -> List[str]:
    """
    Placeholder bound via :func:`bind_prompt_selection_exports`.

    :param _example: Prompt example passed to the helper.
    :param _candidates: Optional candidate metadata sequence.
    :returns: List of feature tokens once bound; raises otherwise.
    """

    raise _unbound_export("selection_feature_tokens") from None


def assemble_document(
    _example: Mapping[str, object],
    _extra_fields: Sequence[str] | None = None,
) -> str:
    """
    Placeholder bound via :func:`bind_prompt_selection_exports`.

    :param _example: Prompt example passed to the helper.
    :param _extra_fields: Optional additional prompt fields.
    :returns: Prompt document once bound; raises otherwise.
    """

    raise _unbound_export("assemble_document") from None


def prepare_training_documents(
    _train_ds: object,
    _max_train: int,
    _seed: int,
    _extra_fields: Sequence[str] | None = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Placeholder bound via :func:`bind_prompt_selection_exports`.

    :param _train_ds: Training dataset passed to the helper.
    :param _max_train: Maximum number of training examples requested.
    :param _seed: Random seed used during subsampling.
    :param _extra_fields: Optional additional prompt fields.
    :returns: Tuple of documents, label IDs, and label titles once bound.
    """

    raise _unbound_export("prepare_training_documents") from None


def prepare_prompt_documents(
    _train_ds: object,
    _max_train: int,
    _seed: int,
    _extra_fields: Sequence[str] | None = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Placeholder bound via :func:`bind_prompt_selection_exports`.

    :param _train_ds: Training dataset passed to the helper.
    :param _max_train: Maximum number of training examples requested.
    :param _seed: Random seed used during subsampling.
    :param _extra_fields: Optional additional prompt fields.
    :returns: Tuple of documents, label IDs, and label titles once bound.
    """

    raise _unbound_export("prepare_prompt_documents") from None


PROMPT_SELECTION_EXPORT_ATTRS = (
    "title_for",
    "viewer_profile_sentence",
    "prompt_from_builder",
    "extract_now_watching",
    "extract_slate_items",
    "collect_candidate_metadata",
    "selection_feature_tokens",
    "assemble_document",
    "prepare_training_documents",
    "prepare_prompt_documents",
)


def bind_prompt_selection_exports(
    helper: PromptSelectionHelper,
    *,
    include_candidate_tokens: bool = False,
) -> Dict[str, object]:
    """Return a mapping of standard prompt-selection helpers exposed by modules.

    :param helper: Instance wrapping a
        :class:`~common.prompts.docs.builder.PromptDocumentBuilder`.
    :param include_candidate_tokens: When ``True`` include ``candidate_feature_tokens``.
    :returns: Dictionary suitable for ``globals().update(...)``.
    """

    exports: Dict[str, object] = {
        name: getattr(helper, name) for name in PROMPT_SELECTION_EXPORT_ATTRS
    }
    if include_candidate_tokens:
        exports["candidate_feature_tokens"] = helper.candidate_feature_tokens
    return exports


__all__ = [
    "CandidateMetadata",
    "PROMPT_SELECTION_EXPORT_ATTRS",
    "PromptSelectionHelper",
    "bind_prompt_selection_exports",
    "candidate_feature_tokens",
    *PROMPT_SELECTION_EXPORT_ATTRS,
]
