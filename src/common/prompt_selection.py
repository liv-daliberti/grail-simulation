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
from typing import List, Mapping, Optional, Sequence, Tuple

from numpy.random import default_rng

from common.prompt_docs import PromptDocumentBuilder
from common.text import canon_text, canon_video_id


@dataclass(frozen=True)
class CandidateMetadata:
    """Lightweight view of slate candidate attributes used for token augmentation."""

    slot: int
    title: str
    video_id: str
    channel_title: Optional[str] = None
    channel_id: Optional[str] = None


def _canon_token(value: object) -> str:
    """Return a lowercase alphanumeric token suitable for feature injection."""

    if not value:
        return ""
    return canon_text(str(value))


def _parse_gold_index(raw_value: object) -> int:
    """Return ``gold_index`` coerced to ``int`` or ``-1`` when unavailable."""

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
    """Return tokens describing high-level selection context."""

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
    """Return tokens highlighting the selected candidate."""

    tokens = candidate_feature_tokens(metadata, option_count=None)
    return [token for token in tokens if not token.startswith("slot_token_")]


def _selected_candidate_metadata(
    example: Mapping[str, object],
    candidates: Sequence[CandidateMetadata],
    gold_index: int,
    solution_column: str,
) -> Optional[CandidateMetadata]:
    """Return metadata describing the selected candidate when available."""

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
    """Generate candidate-specific tokens used during query scoring."""

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
    """Wrap :class:`PromptDocumentBuilder` with shared selection helpers."""

    builder: PromptDocumentBuilder

    @property
    def solution_column(self) -> str:
        """Expose the solution column tracked by the prompt builder."""

        return self.builder.solution_column

    def title_for(self, video_id: str) -> Optional[str]:
        """Look up a human-readable title for ``video_id``."""

        return self.builder.title_for(video_id)

    def viewer_profile_sentence(self, example: Mapping[str, object]) -> str:
        """Compose the viewer profile sentence associated with ``example``."""

        return self.builder.viewer_profile_sentence(dict(example))

    def prompt_from_builder(self, example: Mapping[str, object]) -> str:
        """Assemble the full prompt text for ``example``."""

        return self.builder.prompt_from_builder(dict(example))

    def extract_now_watching(self, example: Mapping[str, object]) -> Optional[Tuple[str, str]]:
        """Retrieve the currently watched item for ``example``."""

        return self.builder.extract_now_watching(dict(example))

    def extract_slate_items(self, example: Mapping[str, object]) -> List[Tuple[str, str]]:
        """Extract the slate of candidate items from ``example``."""

        return self.builder.extract_slate_items(dict(example))

    def collect_candidate_metadata(self, example: Mapping[str, object]) -> List[CandidateMetadata]:
        """Return slate candidate metadata enriched with slot indices and channel info."""

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
        """Delegate to :func:`candidate_feature_tokens` for convenience."""

        return candidate_feature_tokens(candidate, option_count=option_count)

    def selection_feature_tokens(
        self,
        example: Mapping[str, object],
        candidates: Sequence[CandidateMetadata] | None = None,
    ) -> List[str]:
        """Construct additive tokens that highlight the selected candidate within a prompt."""

        if candidates is None:
            candidates = self.collect_candidate_metadata(example)

        gold_index = _parse_gold_index(example.get("gold_index"))
        tokens = _context_tokens(example, option_count=len(candidates), gold_index=gold_index)

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
        """Concatenate prompt components and selection tokens for feature extraction."""

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
        """Prepare prompt documents, labels, and metadata for TF-IDF style training."""

        # pylint: disable=too-many-locals
        n_rows = len(train_ds)  # type: ignore[arg-type]
        if n_rows == 0:
            raise RuntimeError("Train split is empty.")

        rng = default_rng(seed)
        if max_train and max_train > 0:
            take = min(max_train, n_rows)
            indices = rng.permutation(n_rows)[:take].tolist()
        else:
            indices = list(range(n_rows))

        documents: List[str] = []
        label_ids: List[str] = []
        label_titles: List[str] = []

        for index in indices:
            example = train_ds[int(index)]
            document = self.assemble_document(example, extra_fields)
            if not document.strip():
                continue

            video_raw = example.get(self.solution_column)
            video_id = canon_video_id(video_raw)
            if not video_id:
                continue

            candidates = self.collect_candidate_metadata(example)
            gold_index = _parse_gold_index(example.get("gold_index"))

            label_title = self.title_for(video_id) or ""
            if 1 <= gold_index <= len(candidates):
                selected = candidates[gold_index - 1]
                if selected.title and not label_title:
                    label_title = selected.title

            documents.append(document)
            label_ids.append(video_id)
            label_titles.append(label_title)

        if not documents:
            raise RuntimeError("No eligible documents were generated for training.")

        return documents, label_ids, label_titles

    def prepare_prompt_documents(
        self,
        train_ds,
        max_train: int,
        seed: int,
        extra_fields: Sequence[str] | None = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Convenience wrapper mirroring :meth:`prepare_training_documents`."""

        return self.prepare_training_documents(
            train_ds,
            max_train=max_train,
            seed=seed,
            extra_fields=extra_fields,
        )


__all__ = [
    "CandidateMetadata",
    "PromptSelectionHelper",
    "candidate_feature_tokens",
]
