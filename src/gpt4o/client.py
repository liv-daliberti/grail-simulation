"""Azure OpenAI client helpers for GPT-4o evaluation."""

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

try:  # pragma: no cover - optional dependency
    from openai import AzureOpenAI as _AzureOpenAI  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    _AzureOpenAI = None  # type: ignore[assignment]
    _azure_openai_import_error = exc
else:  # pragma: no cover
    _azure_openai_import_error = None


def _require_openai() -> Any:
    """Return the Azure OpenAI class, raising an informative error if missing."""

    if _AzureOpenAI is None:
        raise ImportError(
            "The 'openai' package is required to use Azure OpenAI client helpers. "
            "Install it with `pip install openai`."
        ) from _azure_openai_import_error
    return _AzureOpenAI


@lru_cache(maxsize=1)
def _cached_client(
    api_key: str, endpoint: str, api_version: str
) -> AzureOpenAIType:
    """Return a cached Azure OpenAI client instance."""

    client_cls = _require_openai()
    return client_cls(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def get_client() -> AzureOpenAIType:
    """Construct or reuse the singleton Azure OpenAI client."""

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
    """Execute a chat completion call and return the trimmed text output."""

    client = get_client()
    response = client.chat.completions.create(
        model=deployment or DEPLOYMENT_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    return response.choices[0].message.content.strip()
