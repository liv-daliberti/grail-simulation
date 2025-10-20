"""Azure OpenAI client helpers for GPT-4o evaluation."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

from openai import AzureOpenAI

from .config import (
    DEPLOYMENT_NAME,
    SANDBOX_API_KEY,
    SANDBOX_API_VER,
    SANDBOX_ENDPOINT,
    ensure_azure_env,
)


@lru_cache(maxsize=1)
def _cached_client(
    api_key: str, endpoint: str, api_version: str
) -> AzureOpenAI:
    """Return a cached Azure OpenAI client instance."""

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def get_client() -> AzureOpenAI:
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
