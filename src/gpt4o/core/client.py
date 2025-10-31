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

"""Client helper for interacting with the Azure-hosted GPT-4o deployment."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List
from importlib import import_module

from .config import (
    DEPLOYMENT_NAME,
    SANDBOX_API_KEY,
    SANDBOX_API_VER,
    SANDBOX_ENDPOINT,
    ensure_azure_env,
)


def _require_openai() -> Any:
    """Return the Azure OpenAI class, raising an informative error if missing.

    :returns: Azure OpenAI client class exposed by the ``openai`` package.
    :rtype: Any
    :raises ImportError: If the ``openai`` package (with Azure support) is unavailable.
    """

    try:  # pragma: no cover - optional dependency
        mod = import_module("openai")
        client_cls = getattr(mod, "AzureOpenAI")
    except (ImportError, AttributeError) as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'openai' package is required to use Azure OpenAI client helpers. "
            "Install it with `pip install openai`."
        ) from exc
    return client_cls


@lru_cache(maxsize=1)
def _cached_client(
    api_key: str, endpoint: str, api_version: str
) -> Any:
    """Return a cached Azure OpenAI client instance.

    :param api_key: Azure OpenAI API key used for authentication.
    :type api_key: str
    :param endpoint: Fully-qualified Azure OpenAI endpoint URL.
    :type endpoint: str
    :param api_version: API version string negotiated with the Azure service.
    :type api_version: str
    :returns: Lazily cached Azure OpenAI client.
    :rtype: openai.AzureOpenAI
    """

    client_cls = _require_openai()
    return client_cls(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def get_client() -> Any:
    """Construct or reuse the singleton Azure OpenAI client.

    :returns: Cached client configured with environment or default credentials.
    :rtype: openai.AzureOpenAI
    """

    ensure_azure_env()
    api_key = os.environ.get("SANDBOX_API_KEY", SANDBOX_API_KEY)
    endpoint = os.environ.get("SANDBOX_ENDPOINT", SANDBOX_ENDPOINT)
    api_version = os.environ.get("SANDBOX_API_VER", SANDBOX_API_VER)
    return _cached_client(api_key, endpoint, api_version)


def ds_call(
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float | None = None,
    deployment: str | None = None,
) -> str:
    """Execute a chat completion call and return the trimmed text output.

    :param messages: Chat messages formatted for the OpenAI API.
    :type messages: list[dict[str, str]]
    :param max_tokens: Maximum number of tokens the model may generate.
    :type max_tokens: int
    :param temperature: Sampling temperature forwarded to the model.
    :type temperature: float
    :param deployment: Optional Azure deployment override.
    :type deployment: str | None
    :returns: Trimmed textual completion returned by the model.
    :rtype: str
    """

    client = get_client()
    completion_kwargs: Dict[str, object] = {
        "model": deployment or DEPLOYMENT_NAME,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if top_p is not None:
        completion_kwargs["top_p"] = top_p
    response = client.chat.completions.create(**completion_kwargs)
    return response.choices[0].message.content.strip()
