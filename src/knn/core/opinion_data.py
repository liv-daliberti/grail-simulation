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

"""Dataset helpers for opinion-regression studies."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from numpy.random import default_rng

from common.opinion import (
    DEFAULT_SPECS,
    OpinionSpec,
    float_or_none,
    make_opinion_example_from_values,
)
from common.prompts.docs import merge_default_extra_fields

from .features import assemble_document, viewer_profile_sentence
from .opinion_models import OpinionExample

LOGGER = logging.getLogger("knn.opinion")


def find_spec(key: str) -> OpinionSpec:
    """
    Return the :class:`OpinionSpec` matching ``key``.

    :param key: Dictionary key identifying the current record.
    :type key: str
    :returns: Matching opinion specification.
    :rtype: OpinionSpec
    :raises KeyError: If ``key`` does not correspond to a known spec.
    """
    normalised = key.strip().lower()
    for spec in DEFAULT_SPECS:
        if spec.key.lower() == normalised:
            return spec
    expected_keys = [spec.key for spec in DEFAULT_SPECS]
    raise KeyError(
        f"Unknown opinion study '{key}'. Expected one of {expected_keys!r}"
    )


def collect_examples(
    dataset,
    *,
    spec: OpinionSpec,
    extra_fields: Sequence[str],
    max_examples: int | None,
    seed: int,
) -> List[OpinionExample]:
    """
    Collapse the dataset split into participant-level opinion rows.

    :param dataset: Dataset split providing raw participant interactions.
    :type dataset: datasets.Dataset | Sequence[Mapping[str, Any]]
    :param spec: Opinion study specification describing the target columns.
    :type spec: OpinionSpec
    :param extra_fields: Additional prompt columns appended to each document.
    :type extra_fields: Sequence[str]
    :param max_examples: Optional cap on participants retained from the split.
    :type max_examples: int | None
    :param seed: Random seed applied when subsampling participants.
    :type seed: int
    :returns: Participant-level examples combining prompts and opinion values.
    :rtype: List[OpinionExample]
    """
    # The KNN opinion pipeline accepts ``extra_fields`` for parity with prompt
    # builders but deliberately avoids using them to reduce target leakage.
    del extra_fields

    LOGGER.info(
        "[OPINION] Collapsing dataset split for study=%s issue=%s rows=%d",
        spec.key,
        spec.issue,
        len(dataset),
    )
    collapsed, sample_doc = _collapse_dataset_split(dataset, spec)

    if sample_doc:
        LOGGER.info("[OPINION] Example prompt: %r", sample_doc)

    if max_examples and 0 < max_examples < len(collapsed):
        rng = default_rng(seed)
        order = rng.permutation(len(collapsed))[:max_examples]
        collapsed = [collapsed[i] for i in order]
        LOGGER.info(
            "[OPINION] Sampled %d participants (max=%d).",
            len(collapsed),
            max_examples,
        )

    return collapsed


def _collapse_dataset_split(
    dataset, spec: OpinionSpec
) -> Tuple[List[OpinionExample], Optional[str]]:
    """
    Collapse the raw dataset rows into per-participant examples for ``spec``.

    :param dataset: Dataset split providing raw participant interactions.
    :param spec: Opinion study specification describing the target columns.
    :returns: Tuple of (collapsed_examples, example_prompt).
    """
    per_participant: Dict[Tuple[str, str], OpinionExample] = {}
    sample_doc: Optional[str] = None

    for idx in range(len(dataset)):
        example = dataset[int(idx)]
        if example.get("issue") != spec.issue or example.get("participant_study") != spec.key:
            continue
        before = float_or_none(example.get(spec.before_column))
        after = float_or_none(example.get(spec.after_column))
        if before is None or after is None:
            continue
        document = _build_opinion_document(example)
        if not document:
            continue
        if sample_doc is None:
            sample_doc = document
        step_index = _parse_step_index(example.get("step_index"))
        participant_id = str(example.get("participant_id") or "")
        key = (participant_id, spec.key)
        existing = per_participant.get(key)
        candidate = _make_opinion_candidate(
            spec=spec,
            example=example,
            participant_id=participant_id,
            document=document,
            scores=(before, after),
            step_index=step_index,
        )
        if existing is None or step_index >= existing.step_index:
            per_participant[key] = candidate

    collapsed = list(per_participant.values())
    LOGGER.info(
        "[OPINION] Collapsed to %d unique participants (from %d rows).",
        len(collapsed),
        len(dataset),
    )
    return collapsed, sample_doc


def _build_opinion_document(example) -> str:
    """
    Build a sanitised opinion document that avoids target leakage.

    Uses only the viewer profile and state text; falls back to
    :func:`assemble_document` with a restricted field set when inputs are sparse.
    """
    viewer_profile = viewer_profile_sentence(example)
    state = str(example.get("state_text") or "").strip()
    document = " ".join(token for token in (viewer_profile, state) if token).strip()
    if document:
        return document
    return assemble_document(example, ("viewer_profile", "state_text"))


def _parse_step_index(raw) -> int:
    """Parse a step index value robustly, defaulting to -1 when invalid."""
    try:
        return int(raw or -1)
    except (TypeError, ValueError):
        return -1


def _make_opinion_candidate(
    *,
    spec: OpinionSpec,
    example,
    participant_id: str,
    document: str,
    scores: Tuple[float, float],
    step_index: int,
) -> OpinionExample:
    """
    Construct an :class:`OpinionExample` from the raw ``example`` and precomputed fields.
    """
    session_id = example.get("session_id")
    return make_opinion_example_from_values(
        spec,
        participant_id,
        document,
        scores=scores,
        factory=OpinionExample,
        step_index=step_index,
        session_id=str(session_id) if session_id is not None else None,
    )


def _resolve_requested_specs(args) -> List[OpinionSpec]:
    """
    Return the requested opinion specs based on CLI arguments.

    :param args: Parsed CLI arguments.
    :type args: Any
    :returns: Sequence of opinion specifications to evaluate.
    :rtype: List[OpinionSpec]
    """
    raw = getattr(args, "opinion_studies", "") or ""
    if raw:
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
    else:
        tokens = [spec.key for spec in DEFAULT_SPECS]
    return [find_spec(token) for token in tokens]


def _extra_fields_from_args(args) -> Sequence[str]:
    """
    Extract extra prompt fields from CLI arguments.

    :param args: Parsed CLI arguments.
    :type args: Any
    :returns: Sequence of additional prompt fields.
    :rtype: Sequence[str]
    """
    raw_fields = getattr(args, "knn_text_fields", "") or ""
    tokens = [token.strip() for token in raw_fields.split(",") if token.strip()]
    return merge_default_extra_fields(tokens)


__all__ = [
    "_extra_fields_from_args",
    "_resolve_requested_specs",
    "collect_examples",
    "find_spec",
]
