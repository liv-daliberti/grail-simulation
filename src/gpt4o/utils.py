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

"""Utility helpers shared across the GPT-4o baseline implementation."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from common.text import (
    canon_text as _canon_text,
    canon_video_id as _canon_video_id,
    resolve_paths_from_env as _resolve_paths_from_env,
    split_env_list as _split_env_list,
)

from .client import ds_call

ANS_TAG = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
THINK_BLOCK = re.compile(r"(?si)<think>.+?</think>")
ANSWER_BLOCK = re.compile(r"(?si)<answer>.+?</answer>")
INDEX_ONLY = re.compile(r"^\s*(?:option\s*)?(\d+)\s*$", re.I)


@dataclass(frozen=True)
class InvocationParams:
    """Parameters controlling a single GPT-4o invocation."""

    max_tokens: int
    temperature: float
    top_p: float | None = None
    deployment: str | None = None


@dataclass(frozen=True)
class RetryPolicy:
    """Retry behaviour applied to GPT-4o calls."""

    attempts: int = 5
    delay: float = 1.0
    validator: Callable[[str], bool] | None = None


def canon_text(value: str | None) -> str:
    """Normalise ``value`` using the shared canonical text helper.

    :param value: Raw text value to canonicalise.
    :returns: Canonicalised string (possibly empty).
    """

    return _canon_text(value)


def canon_video_id(value: str | None) -> str:
    """Extract a canonical YouTube id from ``value``.

    :param value: Raw identifier or text containing a video id.
    :returns: Canonical 11-character video id or empty string.
    """

    return _canon_video_id(value)


def split_env_list(raw: str | None) -> list[str]:
    """Split ``raw`` using the separators understood by the common helper.

    :param raw: Environment variable value listing multiple items.
    :returns: List of trimmed tokens.
    """

    return _split_env_list(raw)


def resolve_paths_from_env(env_vars: list[str]) -> list[str]:
    """Return resolved filesystem paths aggregated from ``env_vars``.

    :param env_vars: Ordered list of environment variable names.
    :returns: Resolved filesystem paths aggregated from the variables.
    """

    return _resolve_paths_from_env(env_vars)


def is_nan_like(value: object | None) -> bool:
    """Return ``True`` when the provided value should be treated as missing.

    :param value: Candidate value to inspect.
    :returns: ``True`` if the value represents a missing entry.
    """

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null", "na", "n/a"}
    return False


def truthy(value: object | None) -> bool:
    """Return ``True`` for typical boolean truthy values used in the dataset.

    :param value: Raw value to interpret as a boolean.
    :returns: Boolean interpretation of ``value``.
    """

    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    string_value = str(value).strip().lower()
    return string_value in {"1", "true", "t", "yes", "y"}


def repo_root() -> Path:
    """Return the repository root used by the GPT-4o modules.

    :returns: Repository root path.
    """

    return Path(__file__).resolve().parents[2]


def qa_log_path_for(run_dir: Path, *, filename: str = "qa.log") -> Path:
    """Return the canonical QA log path under logs/gpt mirroring ``run_dir``.

    :param run_dir: Model output directory whose structure is mirrored.
    :param filename: QA log filename to materialise.
    :returns: Destination path for the QA log file.
    """

    repo = repo_root()
    logs_root = repo / "logs" / "gpt"
    try:
        relative = run_dir.relative_to(repo / "models" / "gpt-4o")
    except ValueError:
        try:
            relative = run_dir.relative_to(repo)
        except ValueError:
            relative = Path(run_dir.name)
    destination_dir = logs_root / relative
    destination_dir.mkdir(parents=True, exist_ok=True)
    return destination_dir / filename


LOGGER = logging.getLogger("gpt4o.utils")


def _has_required_reasoning_tags(text: str) -> bool:
    """Return ``True`` when ``text`` includes both <think>…</think> and <answer>…</answer>.

    :param text: Model response to validate.
    :returns: ``True`` if the response contains the required tags.
    """

    if not isinstance(text, str) or not text.strip():
        return False
    return bool(THINK_BLOCK.search(text)) and bool(ANSWER_BLOCK.search(text))


def call_gpt4o_with_retries(
    messages,
    *,
    invocation: InvocationParams,
    retry: RetryPolicy | None = None,
    logger: logging.Logger | None = None,
):
    """
    Invoke GPT-4o with retry semantics, propagating the first successful response.

    :param messages: Chat messages formatted for the Azure OpenAI API.
    :param invocation: Model invocation parameters (temperature, max tokens, etc.).
    :param retry: Retry policy controlling attempts and delay.
    :param logger: Optional logger for warnings.
    :returns: Model response text when successful.
    :raises RuntimeError: After exhausting retries without a non-empty response.
    """

    policy = retry or RetryPolicy()
    attempts = max(1, int(policy.attempts or 1))
    sleep_seconds = max(0.0, float(policy.delay or 0.0))
    log = logger or LOGGER
    last_exc: Exception | None = None
    check = policy.validator or _has_required_reasoning_tags

    for attempt in range(1, attempts + 1):
        try:
            response = ds_call(
                messages,
                max_tokens=invocation.max_tokens,
                temperature=invocation.temperature,
                top_p=invocation.top_p,
                deployment=invocation.deployment,
            )
            if response and response.strip() and check(response):
                return response
            raise RuntimeError("Malformed GPT-4o response (missing required tags).")
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            if attempt < attempts:
                log.warning(
                    "GPT-4o call failed (attempt %d/%d): %s", attempt, attempts, exc
                )
                if sleep_seconds:
                    time.sleep(sleep_seconds)
            else:
                break

    if last_exc is None:
        last_exc = RuntimeError("Empty response from GPT-4o.")
    raise RuntimeError(
        f"GPT-4o call failed after {attempts} attempts: {last_exc}"
    ) from last_exc
