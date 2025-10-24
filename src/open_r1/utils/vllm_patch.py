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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import requests


# ─────────────────── helper to parse non-stream JSON ─────────────────────────
def _parse_nonstream_json(data: dict, tokenizer=None) -> List[List[str]]:
    """Normalise various vLLM response schemas into `List[List[str]]`."""
    # OpenAI route
    if "choices" in data:
        return [[c["text"] for c in data["choices"]]]
    # Plain /generate route (newer default)
    if "results" in data:
        return [[r["text"] for r in data["results"]]]
    # vLLM 0.8.x batched output
    if "text" in data and isinstance(data["text"], list):
        return [[t] for t in data["text"]]
    # vLLM 0.8.x token-ID output
    if "completion_ids" in data:
        if tokenizer is None:
            raise RuntimeError(
                "Server returned token IDs but no tokenizer was supplied to safe_generate()."
            )
        return [[tokenizer.decode(ids, skip_special_tokens=True)] for ids in data["completion_ids"]]
    raise RuntimeError(f"Unknown vLLM response format: {data}")


def _consume_stream_response(response, prompt_count: int) -> List[List[str]]:
    """Aggregate streamed JSON lines into per-prompt completions."""
    texts = [[] for _ in range(prompt_count)]
    for line in response.iter_lines():
        if not line:
            continue
        row = json.loads(line.decode())
        idx = row.get("prompt_index", 0)
        texts[idx].append(row["text"])
    return [["".join(parts)] for parts in texts]


# ─────────────────── POST /generate helper ────────────────────────────────────
def safe_generate(
    *,
    prompts: List[str],
    url: str = "http://localhost:8000/generate",
    tokenizer: Any | None = None,  # ← optional tokenizer for token-ID responses
    **options: Any,
) -> List[List[str]]:
    """Robust call to /generate with retry + schema-agnostic decoding."""
    defaults: Dict[str, Any] = {
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "stream": False,
        "max_retries": 3,
        "backoff": 1.0,
        "timeout": 30.0,
    }
    unknown = set(options) - set(defaults)
    if unknown:
        raise TypeError(f"safe_generate received unexpected options: {sorted(unknown)}")
    settings = {**defaults, **options}

    stream = settings["stream"]
    payload = {
        "prompts": prompts,
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "n": settings["n"],
        "max_tokens": settings["max_tokens"],
        "stream": stream,
    }

    for attempt in range(settings["max_retries"]):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=settings["timeout"],
                stream=stream,
            )
            response.raise_for_status()
            if stream:
                return _consume_stream_response(response, len(prompts))
            return _parse_nonstream_json(response.json(), tokenizer)
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.HTTPError,
            RuntimeError,
        ) as exc:
            if attempt < settings["max_retries"] - 1:
                time.sleep(settings["backoff"] * (2**attempt))
            else:
                raise RuntimeError(f"safe_generate failed: {exc}") from exc

    raise RuntimeError("safe_generate exhausted retries without a response")
