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

"""Probability helpers used by the XGBoost evaluation pipeline."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from .evaluation_types import ProbabilityContext
from .utils import canon_video_id


def candidate_probabilities(
    slate: Sequence[tuple[str, str]],
    probability_map: Dict[str, float],
) -> Tuple[Dict[int, float], Dict[int, str]]:
    """Return per-option probabilities and known candidate ids.

    :param slate: Ordered ``(title, video_id)`` pairs representing the slate.
    :param probability_map: Mapping from canonical video ids to predicted probabilities.
    :returns: Tuple of ``({index: probability}, {index: canonical_id})`` keyed by 1-based index.
    """

    candidate_probs = {
        slate_idx + 1: probability_map.get(canon_video_id(candidate_id), 0.0)
        for slate_idx, (_, candidate_id) in enumerate(slate)
    }
    known_candidates = {
        slate_idx + 1: canon_video_id(candidate_id)
        for slate_idx, (_, candidate_id) in enumerate(slate)
        if canon_video_id(candidate_id) in probability_map
    }
    return candidate_probs, known_candidates


def probability_context(
    *,
    prediction_idx: Optional[int],
    candidate_probs: Dict[int, float],
    known_candidates: Dict[int, str],
    gold_id_canon: str,
) -> ProbabilityContext:
    """Summarise probability metadata for the predicted candidate.

    :param prediction_idx: 1-based predicted option index, or ``None`` when abstaining.
    :param candidate_probs: Mapping of slate indices to predicted probabilities.
    :param known_candidates: Slate indices whose canonical ids were observed during training.
    :param gold_id_canon: Canonicalised gold video identifier for correctness checks.
    :returns: :class:`ProbabilityContext` with best probability and hit flags.
    """

    best_probability = (
        candidate_probs.get(prediction_idx, 0.0)
        if prediction_idx is not None
        else 0.0
    )
    record_probability = bool(prediction_idx and prediction_idx in known_candidates)
    known_candidate_hit = bool(
        record_probability
        and prediction_idx is not None
        and known_candidates[prediction_idx] == gold_id_canon
    )
    return ProbabilityContext(
        best_probability=best_probability,
        record_probability=record_probability,
        known_candidate_hit=known_candidate_hit,
    )


__all__ = ["candidate_probabilities", "probability_context"]
