"""Shared pytest fixtures."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

# Single transparent 1x1 PNG (base64-encoded)
_PNG_DATA = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    """Return the path to a tiny PNG image for embedding in reports."""

    path = tmp_path / "sample.png"
    path.write_bytes(_PNG_DATA)
    return path
