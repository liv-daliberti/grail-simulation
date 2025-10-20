"""Pytest configuration ensuring project modules are importable."""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from tests.helpers.datasets_stub import ensure_datasets_stub  # noqa: E402
from tests.helpers.graphviz_stub import ensure_graphviz_stub  # noqa: E402
from tests.helpers.openai_stub import ensure_openai_stub  # noqa: E402
from tests.helpers.pandas_stub import ensure_pandas_stub  # noqa: E402
from tests.helpers.torch_stub import ensure_torch_stub  # noqa: E402

ensure_datasets_stub()
ensure_pandas_stub()
ensure_graphviz_stub()
ensure_openai_stub()
ensure_torch_stub()
