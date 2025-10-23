"""Utilities shared by router-based execution helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


def build_router_payload(
    scripts: Sequence[str],
    languages: Optional[Sequence[str]],
    *,
    timeout: int | float | None,
    request_timeout: int | float | None,
) -> Tuple[List[str], Dict[str, object]]:
    """Return normalised languages and the request payload for router APIs."""

    language_list = list(languages) if languages is not None else ["python"] * len(scripts)
    payload: Dict[str, object] = {
        "scripts": list(scripts),
        "languages": language_list,
        "timeout": timeout,
        "request_timeout": request_timeout,
    }
    return language_list, payload


__all__ = ["build_router_payload"]
