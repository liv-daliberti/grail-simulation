#!/usr/bin/env python
"""Client helper for interacting with the Azure-hosted GPT-4o deployment."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, TYPE_CHECKING

from .config import (
    DEPLOYMENT_NAME,
    SANDBOX_API_KEY,
    SANDBOX_API_VER,
    SANDBOX_ENDPOINT,
    ensure_azure_env,
)

if TYPE_CHECKING:  # pragma: no cover
    from openai import AzureOpenAI as AzureOpenAIType
else:
    AzureOpenAIType = Any  # type: ignore[assignment]

_AZURE_OPENAI_IMPORT_ERROR: ImportError | None = None

try:  # pragma: no cover - optional dependency
    from openai import AzureOpenAI as _AzureOpenAI  # type: ignore  # pylint: disable=import-error
except ImportError as exc:  # pragma: no cover - optional dependency
    _AzureOpenAI = None  # type: ignore[assignment]
    _AZURE_OPENAI_IMPORT_ERROR = exc
else:  # pragma: no cover
    _AZURE_OPENAI_IMPORT_ERROR = None


def _require_openai() -> Any:
    """Return the Azure OpenAI class, raising an informative error if missing.

    :returns: Azure OpenAI client class exposed by the ``openai`` package.
    :rtype: Any
    :raises ImportError: If the ``openai`` package (with Azure support) is unavailable.
    """

    if _AzureOpenAI is None:
        raise ImportError(
            "The 'openai' package is required to use Azure OpenAI client helpers. "
            "Install it with `pip install openai`."
        ) from _AZURE_OPENAI_IMPORT_ERROR
    return _AzureOpenAI


@lru_cache(maxsize=1)
def _cached_client(
    api_key: str, endpoint: str, api_version: str
) -> AzureOpenAIType:
    """Return a cached Azure OpenAI client instance.

    :param api_key: Azure OpenAI API key used for authentication.
    :type api_key: str
    :param endpoint: Fully-qualified Azure OpenAI endpoint URL.
    :type endpoint: str
    :param api_version: API version string negotiated with the Azure service.
    :type api_version: str
    :returns: Lazily cached Azure OpenAI client.
    :rtype: AzureOpenAIType
    """

    client_cls = _require_openai()
    return client_cls(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def get_client() -> AzureOpenAIType:
    """Construct or reuse the singleton Azure OpenAI client.

    :returns: Cached client configured with environment or default credentials.
    :rtype: AzureOpenAIType
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
    response = client.chat.completions.create(
        model=deployment or DEPLOYMENT_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    return response.choices[0].message.content.strip()
