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

"""Opinion evaluation settings parsing for GPT-4o pipelines."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from common.opinion import DEFAULT_SPECS

from ..config import DATASET_NAME
from .models import OpinionFilters, OpinionLimits, OpinionRuntime, OpinionSettings


def parse_tokens(raw: str | None) -> Tuple[List[str], set[str]]:
    """Return the requested token list and a lowercase set for fast membership tests."""
    tokens: List[str] = []
    for segment in (raw or "").split(","):
        candidate = segment.strip()
        if candidate:
            tokens.append(candidate)
    lowered = {token.lower() for token in tokens}
    if "all" in lowered:
        lowered.clear()
    return tokens, lowered


def resolve_spec_keys(raw: str | None) -> List[str]:
    """Return the ordered opinion study keys to evaluate."""
    if not raw:
        return [spec.key for spec in DEFAULT_SPECS]
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return [spec.key for spec in DEFAULT_SPECS]
    return tokens


def build_settings(args) -> OpinionSettings:
    """Construct :class:`OpinionSettings` from CLI or programmatic arguments."""
    dataset_name = str(getattr(args, "dataset", "") or DATASET_NAME)
    cache_dir = getattr(args, "cache_dir", None)

    issues_raw = str(getattr(args, "issues", "") or "")
    studies_raw = str(getattr(args, "studies", "") or "")
    _, issue_filter = parse_tokens(issues_raw)
    _, study_filter = parse_tokens(studies_raw)
    filters = OpinionFilters(issues=issue_filter, studies=study_filter)

    requested_specs: Sequence[str] = resolve_spec_keys(getattr(args, "opinion_studies", None))

    eval_max = getattr(args, "opinion_max_participants", 0) or getattr(args, "eval_max", 0)
    direction_tolerance = float(
        getattr(
            args,
            "opinion_direction_tolerance",
            getattr(args, "direction_tolerance", 1e-6),
        )
    )
    overwrite = bool(getattr(args, "overwrite", False))
    limits = OpinionLimits(
        eval_max=int(eval_max or 0),
        direction_tolerance=direction_tolerance,
        overwrite=overwrite,
    )

    runtime = OpinionRuntime(
        temperature=float(getattr(args, "temperature", 0.0)),
        max_tokens=int(getattr(args, "max_tokens", 32)),
        top_p=getattr(args, "top_p", None),
        deployment=getattr(args, "deployment", None),
        retries=max(1, int(getattr(args, "request_retries", 5) or 5)),
        retry_delay=max(0.0, float(getattr(args, "request_retry_delay", 1.0) or 0.0)),
    )

    return OpinionSettings(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        filters=filters,
        requested_specs=requested_specs,
        limits=limits,
        runtime=runtime,
    )


__all__ = ["build_settings", "parse_tokens", "resolve_spec_keys"]
