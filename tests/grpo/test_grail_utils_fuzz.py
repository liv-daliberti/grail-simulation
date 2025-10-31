"""Property-style fuzz tests for GRAIL text helpers."""

from __future__ import annotations

import random
import string
from typing import Any

import pytest

from grail.grail_gail import _render_disc_text
from grail.grail_utils import _completion_text


def _random_string(rng: random.Random, min_len: int = 0, max_len: int = 12) -> str:
    length = rng.randint(min_len, max_len)
    alphabet = string.ascii_letters + string.digits + " \n\t"
    return "".join(rng.choice(alphabet) for _ in range(length))


def _random_payload(rng: random.Random, depth: int = 0) -> Any:
    choice = rng.random()
    if depth > 2:
        return _random_string(rng)
    if choice < 0.3:
        return _random_string(rng)
    if choice < 0.6:
        return {
            "role": rng.choice(["system", "user", "assistant"]),
            "content": _random_string(rng),
            "extra": rng.randint(0, 100),
        }
    size = rng.randint(0, 4)
    return [_random_payload(rng, depth + 1) for _ in range(size)]


@pytest.mark.parametrize("seed", range(10))
def test_render_disc_text_handles_random_inputs(monkeypatch: pytest.MonkeyPatch, seed: int) -> None:
    rng = random.Random(seed)
    show_ids = "1" if seed % 2 else "0"
    monkeypatch.setenv("GRAIL_DISC_SHOW_IDS", show_ids)

    items = []
    for _ in range(rng.randint(0, 5)):
        entry = {}
        if rng.random() < 0.7:
            entry["title"] = _random_string(rng, 1, 8)
        if rng.random() < 0.7:
            entry["id"] = _random_string(rng, 1, 6)
        items.append(entry)

    action_surface = items[rng.randrange(len(items))].get("title", "") if items else ""
    action_id = items[rng.randrange(len(items))].get("id") if items else None

    text = _render_disc_text(
        viewer=_random_string(rng),
        state_text=_random_string(rng),
        slate_items=items,
        action_surface=action_surface,
        action_id=action_id,
    )

    assert "VIEWER:" in text
    assert "STATE:" in text
    assert "SLATE (names):" in text
    assert "ACTION_NAME:" in text
    if show_ids == "1":
        assert "ACTION_ID:" in text


@pytest.mark.parametrize("seed", range(20))
def test_completion_text_returns_string_for_random_payloads(seed: int) -> None:
    rng = random.Random(seed)
    payload = _random_payload(rng)
    result = _completion_text(payload)
    assert isinstance(result, str)
