"""Shared pytest fixtures and test-wide configuration."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

from .helpers.datasets_stub import ensure_datasets_stub
from .helpers.graphviz_stub import ensure_graphviz_stub

# Ensure project modules and the datasets stub are available for imports.
ensure_datasets_stub()
ensure_graphviz_stub()
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
