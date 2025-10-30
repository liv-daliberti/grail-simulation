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

"""Shared data models and parsing utilities for the GPT-4o pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping


@dataclass(frozen=True)
class SweepConfig:
    """Describe a GPT-4o configuration evaluated during sweeps."""

    temperature: float
    max_tokens: int
    top_p: float

    def label(self) -> str:
        """
        Build a filesystem-friendly identifier that encodes the configuration.

        :returns: Label composed of temperature, max tokens, and top-p values.
        """

        temp_token = f"temp{self.temperature:g}".replace(".", "p")
        tok_token = f"tok{self.max_tokens}"
        top_p_token = f"tp{self.top_p:g}".replace(".", "p")
        return f"{temp_token}_{tok_token}_{top_p_token}"

    def cli_args(self) -> List[str]:
        """
        Produce CLI arguments that represent this configuration.

        :returns: Sequence of argument tokens suitable for ``gpt4o.cli``.
        """

        return [
            "--temperature",
            str(self.temperature),
            "--max_tokens",
            str(self.max_tokens),
            "--top_p",
            str(self.top_p),
        ]


@dataclass
class SweepOutcome:
    """Capture metrics gathered during a sweep run."""

    config: SweepConfig
    accuracy: float
    parsed_rate: float
    format_rate: float
    metrics_path: Path
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class PipelinePaths:
    """Convenience container aggregating resolved output directories."""

    out_dir: Path
    final_out_dir: Path
    opinion_dir: Path
    sweep_dir: Path
    reports_dir: Path
    cache_dir: str


def coerce_float(value: object, default: float = 0.0) -> float:
    """
    Coerce ``value`` to ``float`` while providing a fallback.

    :param value: Arbitrary value that should represent a numeric quantity.
    :param default: Replacement value when coercion fails.
    :returns: Parsed float or ``default`` when conversion raises ``TypeError`` or ``ValueError``.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _decode_label_token(
    token: str,
    prefix: str,
    *,
    cast: Callable[[str], float | int],
) -> float | int:
    """
    Parse a token extracted from a configuration label.

    :param token: Candidate token pulled from the label string.
    :param prefix: Expected prefix describing the token semantics (e.g. ``"temp"``).
    :param cast: Callable used to convert the token payload into the desired type.
    :returns: Parsed value produced by ``cast``.
    :raises ValueError: If the token is malformed or missing the required prefix.
    """
    if not token.startswith(prefix):
        raise ValueError(f"Token '{token}' missing prefix '{prefix}'.")
    raw = token[len(prefix) :]
    if not raw:
        raise ValueError(f"Token '{token}' missing value.")
    if cast is int:
        return cast(raw)
    normalised = raw.replace("p", ".")
    return cast(normalised)


def parse_config_label(label: str) -> SweepConfig:
    """
    Reconstruct a :class:`SweepConfig` from a label string.

    :param label: Filesystem label produced by :meth:`SweepConfig.label`.
    :returns: Configuration object mirroring the encoded parameters.
    :raises ValueError: If the label format cannot be parsed.
    """
    parts = label.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unrecognised sweep label '{label}'.")
    temperature = float(_decode_label_token(parts[0], "temp", cast=float))
    max_tokens = int(_decode_label_token(parts[1], "tok", cast=int))
    top_p = float(_decode_label_token(parts[2], "tp", cast=float))
    return SweepConfig(temperature=temperature, max_tokens=max_tokens, top_p=top_p)


__all__ = ["SweepConfig", "SweepOutcome", "PipelinePaths", "coerce_float", "parse_config_label"]
