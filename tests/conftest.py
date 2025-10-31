"""Shared pytest fixtures and test-wide configuration."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import pytest

from .helpers.datasets_stub import ensure_datasets_stub
from .helpers.graphviz_stub import ensure_graphviz_stub
from .helpers.openai_stub import ensure_openai_stub

# Ensure project modules and the datasets stub are available for imports.
ensure_datasets_stub()
ensure_graphviz_stub()
ensure_openai_stub()
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_CACHE_ROOT = ROOT / ".cache"
_HF_CACHE = ROOT / ".hf_cache"
_TMP_ROOT = ROOT / ".tmp"
_PYTHON_CACHE = ROOT / ".cache" / "pyc"
_TORCHINDUCTOR_CACHE = ROOT / ".torchinductor"
_TRITON_CACHE = ROOT / ".triton"

_ENV_PATHS: dict[str, Path] = {
    "XDG_CACHE_HOME": _CACHE_ROOT,
    "HF_HOME": _HF_CACHE,
    "HF_HUB_CACHE": _CACHE_ROOT / "huggingface" / "transformers",
    "HF_DATASETS_CACHE": _CACHE_ROOT / "huggingface" / "datasets",
    "TMPDIR": _TMP_ROOT,
    "PIP_CACHE_DIR": _CACHE_ROOT / "pip",
    "PIP_BUILD_DIR": _CACHE_ROOT / "pip" / "build",
    "PYTHONPYCACHEPREFIX": _PYTHON_CACHE,
    "TORCHINDUCTOR_CACHE_DIR": _TORCHINDUCTOR_CACHE,
    "TRITON_CACHE_DIR": _TRITON_CACHE,
}

for env_var, path in _ENV_PATHS.items():
    os.environ.setdefault(env_var, str(path))
    path.mkdir(parents=True, exist_ok=True)

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
