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
    # pylint: disable=too-many-locals

    # The KNN opinion pipeline accepts ``extra_fields`` for parity with prompt
    # builders but deliberately avoids using them to reduce target leakage.
    del extra_fields

    LOGGER.info(
        "[OPINION] Collapsing dataset split for study=%s issue=%s rows=%d",
        spec.key,
        spec.issue,
        len(dataset),
    )
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
        # Build a sanitised opinion document to avoid target leakage.
        # For opinion regression, avoid using the full prompt_builder output
        # since it can include post-study survey fields (e.g., wave-2 indices)
        # that directly encode the regression target. Instead, restrict the
        # document to the viewer profile sentence and state text.
        viewer_profile = viewer_profile_sentence(example)
        state = str(example.get("state_text") or "").strip()
        document = " ".join(token for token in (viewer_profile, state) if token).strip()
        if not document:
            # Fallback to the generic assembler (still avoids extra_fields that
            # may include target values) to reduce empty-doc drops if inputs
            # are unexpectedly sparse.
            document = assemble_document(example, ("viewer_profile", "state_text"))
        if not document:
            continue
        if sample_doc is None:
            sample_doc = document
        try:
            step_index = int(example.get("step_index") or -1)
        except (TypeError, ValueError):
            step_index = -1
        participant_id = str(example.get("participant_id") or "")
        key = (participant_id, spec.key)
        existing = per_participant.get(key)
        session_id = example.get("session_id")
        candidate = make_opinion_example_from_values(
            spec,
            participant_id,
            document,
            scores=(before, after),
            factory=OpinionExample,
            step_index=step_index,
            session_id=str(session_id) if session_id is not None else None,
        )
        if existing is None or step_index >= existing.step_index:
            per_participant[key] = candidate

    collapsed = list(per_participant.values())
    LOGGER.info(
        "[OPINION] Collapsed to %d unique participants (from %d rows).",
        len(collapsed),
        len(dataset),
    )
    if sample_doc:
        LOGGER.info("[OPINION] Example prompt: %r", sample_doc)

    if max_examples and 0 < max_examples < len(collapsed):
        rng = default_rng(seed)
        order = rng.permutation(len(collapsed))[:max_examples]
        collapsed = [collapsed[i] for i in order]
        LOGGER.info("[OPINION] Sampled %d participants (max=%d).", len(collapsed), max_examples)

    return collapsed


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
