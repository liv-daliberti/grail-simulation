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

"""Shared helpers for the modular KNN report generation pipeline."""

from __future__ import annotations

import logging
import shlex
from typing import Iterable, Sequence, Tuple, Union

from common.pipeline_formatters import safe_int

LOGGER = logging.getLogger("knn.pipeline.reports")


def _format_shell_command(bits: Sequence[str]) -> str:
    """Join CLI arguments into a shell-friendly command."""
    return " ".join(shlex.quote(str(bit)) for bit in bits if str(bit))


def _feature_space_heading(feature_space: str) -> str:
    """Return a Markdown heading for the target feature space."""
    if feature_space == "tfidf":
        return "## TF-IDF Feature Space"
    if feature_space == "word2vec":
        return "## Word2Vec Feature Space"
    if feature_space == "sentence_transformer":
        return "## Sentence-Transformer Feature Space"
    return f"## {feature_space.replace('_', ' ').title()} Feature Space"

def parse_k_sweep(k_sweep: Union[str, Iterable[int]]) -> Tuple[int, ...]:
    """Convert the stored ``k`` sweep representation into integers."""
    if isinstance(k_sweep, str):
        tokens = [
            token.strip()
            for token in k_sweep.replace("/", ",").split(",")
            if token.strip()
        ]
        values = []
        for token in tokens:
            parsed = safe_int(token)
            if parsed is None:
                LOGGER.warning("Skipping non-numeric k sweep token: %s", token)
                continue
            values.append(parsed)
        return tuple(values)

    values = []
    for value in k_sweep:
        parsed = safe_int(value)
        if parsed is None:
            LOGGER.warning("Skipping non-numeric k sweep token: %s", value)
            continue
        values.append(parsed)
    return tuple(values)


__all__ = ["LOGGER", "_feature_space_heading", "_format_shell_command", "parse_k_sweep"]
